/**
 * Slack integration for the AI chat agent
 * Handles incoming Slack events (DMs and mentions) and responds via Slack Web API
 */
import { generateText, tool } from "ai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { z } from "zod/v3";
import { getSchedulePrompt } from "agents/schedule";

// Slack event types
interface SlackEvent {
  type: string;
  text?: string;
  user?: string;
  channel?: string;
  channel_type?: string;
  thread_ts?: string;
  ts?: string;
  bot_id?: string;
}

interface SlackEventPayload {
  type: string;
  challenge?: string;
  event?: SlackEvent;
  token?: string;
}

// Environment with Slack secrets
interface SlackEnv {
  SLACK_BOT_TOKEN: string;
  SLACK_SIGNING_SECRET: string;
  REMINDERS_KV: KVNamespace; // Still used for idempotency tracking
  arky_reminders: R2Bucket; // R2 bucket for per-user reminders
  GOOGLE_GENERATIVE_AI_API_KEY: string;
  SLACK_EVENTS_KV?: KVNamespace; // Optional: for tracking processed events
}

/**
 * Verify Slack request signature using HMAC-SHA256
 * https://api.slack.com/authentication/verifying-requests-from-slack
 */
async function verifySlackRequest(
  signingSecret: string,
  timestamp: string,
  body: string,
  signature: string
): Promise<boolean> {
  // Check timestamp is within 5 minutes to prevent replay attacks
  const currentTime = Math.floor(Date.now() / 1000);
  if (Math.abs(currentTime - parseInt(timestamp)) > 60 * 5) {
    console.error("Slack request timestamp too old");
    return false;
  }

  // Create the signature base string
  const sigBaseString = `v0:${timestamp}:${body}`;

  // Create HMAC-SHA256 signature
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(signingSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );

  const signatureBuffer = await crypto.subtle.sign(
    "HMAC",
    key,
    encoder.encode(sigBaseString)
  );

  // Convert to hex string
  const hashArray = Array.from(new Uint8Array(signatureBuffer));
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
  const computedSignature = `v0=${hashHex}`;

  // Constant-time comparison to prevent timing attacks
  if (computedSignature.length !== signature.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < computedSignature.length; i++) {
    result |= computedSignature.charCodeAt(i) ^ signature.charCodeAt(i);
  }

  return result === 0;
}

/**
 * Send a message to Slack via Web API with retry logic
 */
async function sendSlackMessage(
  botToken: string,
  channel: string,
  text: string,
  threadTs?: string,
  maxRetries: number = 3
): Promise<void> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch("https://slack.com/api/chat.postMessage", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${botToken}`,
        },
        body: JSON.stringify({
          channel,
          text,
          thread_ts: threadTs,
        }),
      });

      const result = await response.json() as { ok: boolean; error?: string };
      
      if (result.ok) {
        console.log("Sent Slack message:", text.substring(0, 50));
        return;
      }

      // Rate limiting - wait longer before retry
      if (result.error === "rate_limited") {
        const retryAfter = response.headers.get("Retry-After");
        const waitTime = retryAfter ? parseInt(retryAfter) * 1000 : Math.pow(2, attempt) * 1000;
        console.log(`Rate limited, waiting ${waitTime}ms before retry ${attempt + 1}/${maxRetries}`);
        await new Promise((resolve) => setTimeout(resolve, waitTime));
        continue;
      }

      // Other errors - retry with exponential backoff
      lastError = new Error(result.error || "Unknown Slack API error");
      if (attempt < maxRetries - 1) {
        const waitTime = Math.pow(2, attempt) * 1000; // Exponential backoff: 1s, 2s, 4s
        console.log(`Failed to send Slack message (attempt ${attempt + 1}/${maxRetries}), retrying in ${waitTime}ms:`, result.error);
        await new Promise((resolve) => setTimeout(resolve, waitTime));
      }
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (attempt < maxRetries - 1) {
        const waitTime = Math.pow(2, attempt) * 1000;
        console.log(`Network error sending Slack message (attempt ${attempt + 1}/${maxRetries}), retrying in ${waitTime}ms:`, error);
        await new Promise((resolve) => setTimeout(resolve, waitTime));
      }
    }
  }

  // All retries failed
  console.error("Failed to send Slack message after all retries:", lastError);
  throw lastError || new Error("Failed to send Slack message");
}

/**
 * Generate AI response for Slack message
 * Uses simplified tool set (no tools requiring agent context for now)
 */
async function generateSlackResponse(
  userMessage: string,
  env: SlackEnv,
  userId: string
): Promise<string> {
  const systemPrompt = `You are a helpful assistant responding via Slack. Keep responses concise and conversational.

${getSchedulePrompt({ date: new Date() })}

You have access to various tools and built-in knowledge. You can provide weather information for any location using your built-in capabilities.
You can help with various tasks including:
- Getting local time (use getLocalTime tool)
- Managing reminders (use getReminders and captureReminder tools)
- Weather information - you have built-in weather capabilities and can provide current conditions, forecasts, and temperatures for any location worldwide

Be friendly and helpful.`;

  // Check for API key
  if (!env.GOOGLE_GENERATIVE_AI_API_KEY) {
    console.error("GOOGLE_GENERATIVE_AI_API_KEY not found in env");
    return "Sorry, the AI service is not configured. Please check server settings.";
  }

  try {
    // Simple tools that work without agent context
    const slackTools = {
      getLocalTime: tool({
        description: "get the local time for a specified location",
        inputSchema: z.object({ location: z.string() }),
        execute: async ({ location }) => {
          console.log(`Getting local time for ${location}`);
          // Simple implementation - in production you might want to use a timezone API
          const now = new Date();
          return `The current time in ${location} is approximately ${now.toLocaleTimeString()}`;
        },
      }),
      getReminders: tool({
        description: "Read and list all reminders from the user's reminders.md file",
        inputSchema: z.object({
          search: z.string().optional().describe("Optional search term to filter reminders"),
        }),
        execute: async ({ search }) => {
          // Use per-user R2 path
          const r2Key = `reminders/${userId}/reminders.md`;
          const object = await env.arky_reminders.get(r2Key);

          if (!object) {
            return "No reminders found.";
          }

          const content = await object.text();
          if (!content || content.trim() === "") {
            return "No reminders found.";
          }

          const lines = content.split("\n").filter((line: string) => line.trim() !== "");
          if (search) {
            const searchLower = search.toLowerCase();
            const filtered = lines.filter((line: string) =>
              line.toLowerCase().includes(searchLower)
            );
            return filtered.length > 0
              ? filtered.join("\n")
              : `No reminders found matching "${search}"`;
          }
          return lines.join("\n");
        },
      }),
      captureReminder: tool({
        description: "Save a reminder to the user's reminders.md file. Checks for duplicates first.",
        inputSchema: z.object({
          reminder: z.string().describe("The reminder text to save"),
        }),
        execute: async ({ reminder }) => {
          // Use per-user R2 path
          const r2Key = `reminders/${userId}/reminders.md`;

          // Read existing reminders from R2 to check for duplicates
          const existingObject = await env.arky_reminders.get(r2Key);
          const existingContent = existingObject ? await existingObject.text() : "";
          const existingLines = existingContent
            .split("\n")
            .filter((line: string) => line.trim() !== "");

          // Simple similarity check: extract key terms
          const reminderLower = reminder.toLowerCase();
          const reminderWords = reminderLower
            .split(/\s+/)
            .filter((word: string) => word.length > 3); // Filter out short words

          // Find similar entries
          const similarEntries: string[] = [];
          for (const line of existingLines) {
            const lineLower = line.toLowerCase();
            const matchingWords = reminderWords.filter((word: string) => 
              lineLower.includes(word)
            );
            // If 2+ significant words match, consider it similar
            if (matchingWords.length >= 2) {
              similarEntries.push(line);
            }
          }

          // If duplicates found, return them and ask for confirmation
          if (similarEntries.length > 0) {
            const similarList = similarEntries.join("\n");
            return `Similar reminders already exist:\n\n${similarList}\n\nWould you like to add this new reminder anyway? If yes, please add more specific details to make it unique from the existing entries.`;
          }

          // No duplicates, save the reminder to R2
          const today = new Date().toISOString().split("T")[0];
          const newEntry = `[${today}] ${reminder}\n`;
          const updatedContent = existingContent + newEntry;

          // Write to R2
          await env.arky_reminders.put(r2Key, updatedContent, {
            httpMetadata: {
              contentType: "text/markdown",
            },
          });

          return `Reminder saved: "${reminder}"`;
        },
      }),
    };

    // Create model with explicit API key
    const googleAI = createGoogleGenerativeAI({
      apiKey: env.GOOGLE_GENERATIVE_AI_API_KEY,
    });
    const slackModel = googleAI("gemini-2.5-flash");

    console.log("Generating response for message:", userMessage.substring(0, 50));
    const result = await generateText({
      model: slackModel,
      system: systemPrompt,
      prompt: userMessage,
      tools: slackTools,
    });

    // Log result for debugging
    if (!result.text) {
      console.error("Empty response from generateText:", JSON.stringify(result, null, 2));
    }

    // If we have text, return it
    if (result.text && result.text.trim()) {
      return result.text.trim();
    }

    // If tools were called but no text response, extract tool result values
    // The tool results contain a 'value' property with the actual output
    if (result.toolCalls && result.toolCalls.length > 0) {
      // Check for toolResults array which contains the execution results
      if (result.toolResults && Array.isArray(result.toolResults) && result.toolResults.length > 0) {
        // Extract the 'value' property from each tool result
        const toolResultValues = result.toolResults
          .map((tr: any) => {
            // Extract value property which contains the tool output
            if (tr.value !== undefined) {
              return String(tr.value);
            }
            // Fallback to other possible properties
            if (tr.result !== undefined) return String(tr.result);
            if (tr.output !== undefined) return String(tr.output);
            // If it's a string, use it directly
            if (typeof tr === "string") return tr;
            return null;
          })
          .filter((val: string | null): val is string => val !== null && val.trim() !== "")
          .join("\n");

        if (toolResultValues) {
          return toolResultValues;
        }
      }
    }

    // If no text and no tool results found, return a helpful message
    return "I processed your request but have no text response. Please try rephrasing your request.";
  } catch (error) {
    console.error("Error generating Slack response:", error);
    return `Sorry, I encountered an error: ${error instanceof Error ? error.message : String(error)}`;
  }
}

/**
 * Main Slack event handler
 * Processes incoming Slack events and responds accordingly
 */
export async function handleSlackEvent(
  request: Request,
  env: SlackEnv,
  ctx?: ExecutionContext
): Promise<Response> {
  // Only accept POST requests
  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  // Check for required secrets
  if (!env.SLACK_SIGNING_SECRET || !env.SLACK_BOT_TOKEN) {
    console.error("SLACK_SIGNING_SECRET or SLACK_BOT_TOKEN not configured");
    return new Response("Server configuration error", { status: 500 });
  }

  // Get headers for verification
  const timestamp = request.headers.get("x-slack-request-timestamp");
  const signature = request.headers.get("x-slack-signature");

  if (!timestamp || !signature) {
    return new Response("Missing Slack headers", { status: 400 });
  }

  // Read body
  const body = await request.text();

  // Verify request signature
  const isValid = await verifySlackRequest(
    env.SLACK_SIGNING_SECRET,
    timestamp,
    body,
    signature
  );

  if (!isValid) {
    console.error("Invalid Slack signature");
    return new Response("Invalid signature", { status: 401 });
  }

  // Parse payload
  let payload: SlackEventPayload;
  try {
    payload = JSON.parse(body);
  } catch {
    return new Response("Invalid JSON", { status: 400 });
  }

  // Handle URL verification challenge (Slack sends this when setting up events)
  if (payload.type === "url_verification") {
    return new Response(payload.challenge, {
      headers: { "Content-Type": "text/plain" },
    });
  }

  // Handle event callbacks
  if (payload.type === "event_callback" && payload.event) {
    const event = payload.event;

    // Skip bot messages to avoid infinite loops
    if (event.bot_id) {
      return new Response("OK", { status: 200 });
    }

    // Idempotency: Check if we've already processed this event
    // Slack events have a unique event_ts that we can use as an idempotency key
    const eventId = event.ts || `${event.type}-${event.user}-${Date.now()}`;
    const eventKey = `slack-event:${eventId}`;
    
    // Use REMINDERS_KV for idempotency tracking (or create a separate namespace)
    const idempotencyKV = env.SLACK_EVENTS_KV || env.REMINDERS_KV;
    const processedEvent = await idempotencyKV.get(eventKey);
    
    if (processedEvent) {
      console.log(`Event ${eventId} already processed, skipping duplicate`);
      return new Response("OK", { status: 200 });
    }

    // Mark event as processed (expires in 24 hours to prevent KV bloat)
    await idempotencyKV.put(eventKey, "processed", { expirationTtl: 86400 });

    let userText: string | undefined;
    let channelId: string | undefined;
    let threadTs: string | undefined;

    // Handle direct messages
    if (event.type === "message" && event.channel_type === "im") {
      userText = event.text;
      channelId = event.channel;
      // Reply in thread if this is already in a thread
      threadTs = event.thread_ts;
    }
    // Handle app mentions in channels
    else if (event.type === "app_mention") {
      // Remove the bot mention from the message
      userText = event.text?.replace(/<@[A-Z0-9]+>/gi, "").trim();
      channelId = event.channel;
      // Reply in thread
      threadTs = event.ts;
    }

    // If we have a message to process
    if (userText && channelId && event.user) {
      const userId = event.user; // Extract Slack user ID
      
      // Return 200 OK immediately to Slack (required within 3 seconds)
      // Process the response asynchronously in the background
      const processPromise = (async () => {
        try {
          const response = await generateSlackResponse(userText!, env, userId);
          await sendSlackMessage(env.SLACK_BOT_TOKEN, channelId!, response, threadTs);
        } catch (error) {
          console.error("Error processing Slack message:", error);
          try {
            await sendSlackMessage(
              env.SLACK_BOT_TOKEN,
              channelId!,
              "Sorry, I encountered an error processing your message.",
              threadTs
            );
          } catch (sendError) {
            console.error("Failed to send error message to Slack:", sendError);
          }
        }
      })();

      // Use ctx.waitUntil if available to ensure background task completes
      if (ctx) {
        ctx.waitUntil(processPromise);
      }
      // Don't await - return immediately
    }

    return new Response("OK", { status: 200 });
  }

  // Default response for unhandled event types
  return new Response("OK", { status: 200 });
}
