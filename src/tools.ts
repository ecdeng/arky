/**
 * Tool definitions for the AI chat agent
 * Tools can either require human confirmation or execute automatically
 */
import { tool, type ToolSet, generateText } from "ai";
import { z } from "zod/v3";
import { google } from "@ai-sdk/google";

import type { Chat } from "./server";
import { getCurrentAgent } from "agents";
import { scheduleSchema } from "agents/schedule";

/**
 * Weather information tool that requires human confirmation
 * When invoked, this will present a confirmation dialog to the user
 */
const getWeatherInformation = tool({
  description: "show the weather in a given city to the user",
  inputSchema: z.object({ city: z.string() })
  // Omitting execute function makes this tool require human confirmation
});

/**
 * Local time tool that executes automatically
 * Since it includes an execute function, it will run without user confirmation
 * This is suitable for low-risk operations that don't need oversight
 */
const getLocalTime = tool({
  description: "get the local time for a specified location",
  inputSchema: z.object({ location: z.string() }),
  execute: async ({ location }) => {
    console.log(`Getting local time for ${location}`);
    return "10am";
  }
});

const scheduleTask = tool({
  description: "A tool to schedule a task to be executed at a later time",
  inputSchema: scheduleSchema,
  execute: async ({ when, description }) => {
    // we can now read the agent context from the ALS store
    const { agent } = getCurrentAgent<Chat>();

    function throwError(msg: string): string {
      throw new Error(msg);
    }
    if (when.type === "no-schedule") {
      return "Not a valid schedule input";
    }
    const input =
      when.type === "scheduled"
        ? when.date // scheduled
        : when.type === "delayed"
          ? when.delayInSeconds // delayed
          : when.type === "cron"
            ? when.cron // cron
            : throwError("not a valid schedule input");
    try {
      agent!.schedule(input!, "executeTask", description);
    } catch (error) {
      console.error("error scheduling task", error);
      return `Error scheduling task: ${error}`;
    }
    return `Task scheduled for type "${when.type}" : ${input}`;
  }
});

/**
 * Tool to list all scheduled tasks
 * This executes automatically without requiring human confirmation
 */
const getScheduledTasks = tool({
  description: "List all tasks that have been scheduled",
  inputSchema: z.object({}),
  execute: async () => {
    const { agent } = getCurrentAgent<Chat>();

    try {
      const tasks = agent!.getSchedules();
      if (!tasks || tasks.length === 0) {
        return "No scheduled tasks found.";
      }
      return tasks;
    } catch (error) {
      console.error("Error listing scheduled tasks", error);
      return `Error listing scheduled tasks: ${error}`;
    }
  }
});

/**
 * Tool to cancel a scheduled task by its ID
 * This executes automatically without requiring human confirmation
 */
const cancelScheduledTask = tool({
  description: "Cancel a scheduled task using its ID",
  inputSchema: z.object({
    taskId: z.string().describe("The ID of the task to cancel")
  }),
  execute: async ({ taskId }) => {
    const { agent } = getCurrentAgent<Chat>();
    try {
      await agent!.cancelSchedule(taskId);
      return `Task ${taskId} has been successfully canceled.`;
    } catch (error) {
      console.error("Error canceling scheduled task", error);
      return `Error canceling task ${taskId}: ${error}`;
    }
  }
});

/**
 * Tool to capture and store reminders with enriched information
 * Automatically enriches the reminder with additional context (e.g., restaurant details, addresses)
 * and appends to a persistent reminders.md file
 */
const captureReminder = tool({
  description: "Capture a reminder from the user, enrich it with additional details (like restaurant addresses, cuisine types, etc.), and save it to a persistent reminders.md file",
  inputSchema: z.object({
    reminder: z.string().describe("The reminder text to capture and enrich")
  }),
  execute: async ({ reminder }) => {
    const { agent } = getCurrentAgent<Chat>();
    
    if (!agent) {
      return "Error: Agent context not available";
    }

    try {
      // Access KV namespace from agent's env
      const kv = (agent as any).env?.REMINDERS_KV;
      if (!kv) {
        return "Error: REMINDERS_KV binding not configured. Please add a KV namespace binding in wrangler.jsonc";
      }

      // Enrich the reminder using AI
      const enrichmentModel = google("gemini-2.5-flash");
      const enrichmentPrompt = `You are a helpful assistant that enriches reminders with useful details.

Given this reminder: "${reminder}"

If this mentions a business (restaurant, venue, shop, etc.), look up and include:
- Full business name
- Address (if available)
- Price range (if applicable)
- Type/cuisine (if applicable)
- Neighborhood/area (if applicable)

Create a concise 1-2 sentence summary that includes the enriched information. 
If no business is mentioned, just create a clear 1-2 sentence summary of the reminder.

Only output the enriched summary text, nothing else.`;

      const enrichmentResult = await generateText({
        model: enrichmentModel,
        prompt: enrichmentPrompt,
      });

      const enrichedText = enrichmentResult.text.trim();

      // Get current date in YYYY-MM-DD format
      const today = new Date().toISOString().split('T')[0];

      // Read existing reminders.md from KV to check for duplicates
      const existingContent = await kv.get("reminders.md");
      const currentContent = existingContent || "";

      // Check for similar/duplicate entries
      const existingLines = currentContent
        .split("\n")
        .filter((line: string) => line.trim() !== "");

      // Simple similarity check: extract key terms from enriched text
      // (e.g., business name, location, person's name)
      const enrichedLower = enrichedText.toLowerCase();
      const enrichedWords = enrichedLower
        .split(/\s+/)
        .filter((word: string) => word.length > 3); // Filter out short words

      // Find similar entries (containing same key terms)
      const similarEntries: string[] = [];
      for (const line of existingLines) {
        const lineLower = line.toLowerCase();
        // Count matching significant words
        const matchingWords = enrichedWords.filter((word: string) => 
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

      // Format the new reminder entry
      const newEntry = `[${today}] ${enrichedText}\n`;

      // Append new entry
      const updatedContent = currentContent + newEntry;

      // Write back to KV
      await kv.put("reminders.md", updatedContent);

      return `Reminder captured and saved! Added to reminders.md: "${enrichedText}"`;
    } catch (error) {
      console.error("Error capturing reminder", error);
      return `Error capturing reminder: ${error instanceof Error ? error.message : String(error)}`;
    }
  }
});

/**
 * Tool to read and list all reminders from reminders.md
 * This executes automatically without requiring human confirmation
 */
const getReminders = tool({
  description: "Read and list all reminders from the reminders.md file. Can optionally filter by search term",
  inputSchema: z.object({
    search: z.string().optional().describe("Optional search term to filter reminders (e.g., 'LA', 'restaurant', 'Margaret')")
  }),
  execute: async ({ search }) => {
    const { agent } = getCurrentAgent<Chat>();
    
    if (!agent) {
      return "Error: Agent context not available";
    }

    try {
      // Access KV namespace from agent's env
      const kv = (agent as any).env?.REMINDERS_KV;
      if (!kv) {
        return "Error: REMINDERS_KV binding not configured. Please add a KV namespace binding in wrangler.jsonc";
      }

      // Read reminders.md from KV
      const content = await kv.get("reminders.md");
      
      if (!content || content.trim() === "") {
        return "No reminders found. The reminders.md file is empty.";
      }

      // Split into individual reminder lines
      const lines = content.split("\n").filter((line: string) => line.trim() !== "");
      
      // Filter by search term if provided
      let filteredLines = lines;
      if (search) {
        const searchLower = search.toLowerCase();
        filteredLines = lines.filter((line: string) => line.toLowerCase().includes(searchLower));
      }

      if (filteredLines.length === 0) {
        return search 
          ? `No reminders found matching "${search}"`
          : "No reminders found.";
      }

      // Format the output
      const result = filteredLines.join("\n");
      
      return result;
    } catch (error) {
      console.error("Error reading reminders", error);
      return `Error reading reminders: ${error instanceof Error ? error.message : String(error)}`;
    }
  }
});

/**
 * Export all available tools
 * These will be provided to the AI model to describe available capabilities
 */
export const tools = {
  getWeatherInformation,
  getLocalTime,
  scheduleTask,
  getScheduledTasks,
  cancelScheduledTask,
  captureReminder,
  getReminders
} satisfies ToolSet;

/**
 * Implementation of confirmation-required tools
 * This object contains the actual logic for tools that need human approval
 * Each function here corresponds to a tool above that doesn't have an execute function
 */
export const executions = {
  getWeatherInformation: async ({ city }: { city: string }) => {
    console.log(`Getting weather information for ${city}`);
    return `The weather in ${city} is sunny`;
  }
};
