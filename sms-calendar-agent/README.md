# Local SMS Listing Coordination Agent

This is a small local Flask app that connects a Twilio SMS number to an OpenAI Agents SDK agent. Arky helps coordinate the work needed to list a home, including availability by text. It supports direct SMS and, when configured through Twilio Conversations, group MMS. It does not connect to a calendar or create events. SMS conversation history is stored under `data/conversations.db`; coordination state is stored under `data/coordination.db`.

## 1. Install

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set at least:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-6-luna
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
APP_TIMEZONE=America/Los_Angeles
```

The app defaults to `gpt-6-luna` for both Arky and its availability summarizer. Set
`OPENAI_MODEL` to another model ID if you want to change the cost/quality tradeoff.

## 2. Run locally

```bash
python app.py
```

Check that it is running:

```bash
curl http://127.0.0.1:5000/health
```

## 3. Expose the webhook during development

In another terminal, expose port 5000 with a tunnel such as ngrok:

```bash
ngrok http 5000
```

Copy the HTTPS forwarding URL, then configure your Twilio phone number:

1. Open the phone number in the Twilio Console.
2. Under Messaging, set **A message comes in** to **Webhook**.
3. Use `https://YOUR-NGROK-DOMAIN/sms` with HTTP POST.
4. For the first test, leave `TWILIO_VALIDATE_REQUESTS=false`.
5. After the webhook works, set it to `true`, restart the app, and make sure `TWILIO_AUTH_TOKEN` is set.

## 4. Enable group chats and availability coordination

The agent can respond in a Twilio Conversations group, privately text selected phone participants for availability, collect their replies, then post a summary back to the originating group. For example: “Arky, find a 30-minute time for Eric and Aaron next week.” The agent only texts phone numbers present in that active group. It suggests times based only on their text replies; it cannot inspect calendars or create events.

Twilio's ordinary Programmable Messaging webhook does not support native Group MMS. Group MMS is supported through Twilio Conversations (classic) for US/Canada. It is not a way to join arbitrary iMessage, WhatsApp, or other app-based groups; the group must be a supported Twilio Conversations Group MMS conversation. See Twilio's [group MMS/autocreation guide](https://www.twilio.com/docs/conversations-classic/inbound-autocreation).

1. In Twilio, configure a Conversations (classic) service for Group MMS and the Twilio number/Messaging Service you will use. Enable inbound autocreation for the number so both the group and private replies become Conversations events. Follow Twilio's setup carefully: it changes how inbound messages for numbers in that Messaging Service are routed.
2. In the Conversations service's Webhooks settings, set the post-event webhook URL to `https://YOUR-NGROK-DOMAIN/conversation-webhook` and enable the `onMessageAdded` event. The app queues this event and processes it in the background because Conversations webhooks have a short response timeout. See [Twilio's webhook configuration docs](https://www.twilio.com/docs/conversations-classic/conversations-webhooks).
3. Set `TWILIO_CONVERSATIONS_MODE=true` in `.env` and point the number-level incoming-message webhook to `/sms` as well (the app acknowledges but ignores it in this mode). Twilio may fire the number webhook independently of the Conversations event; this setting ensures the Conversations event is the single processing path. The `/sms` route remains active for direct SMS-only mode when this setting is false.
4. Add the Twilio number to a supported group MMS conversation and include the bot in the chat. Send a test message mentioning a scheduling request.

For direct SMS and Conversations event delivery to work together, the number and Messaging Service settings must match the Twilio routing setup. Start with a test number/group; don't enable autocreation broadly on a production Messaging Service without understanding its routing effects.

## 5. Try texting it

Examples:

```text
Help me plan the remaining tasks to list the house.
What do we still need to do before photography?
Arky, coordinate a 30-minute time with Eric and Aaron next week for photos.
Arky, ask Eric when he can finish retouching the photos.
```

Send `RESET` to clear the conversation history for your phone number.

In a group, the agent can be asked to coordinate availability for named people. It will privately text those people, accept their replies, and share a combined summary in the original group. Each person must reply before the summary is posted. If a message cannot be delivered, the summary flags that person as unreachable.

## Important scope note

Arky coordinates scheduling through texts only. It cannot see anyone's availability unless they provide it by text, and it does not add agreed times to a calendar or send future reminders. Group coordination requires each selected person to reply before a combined summary is posted. The local app's in-process background worker is intended for prototyping; work can be lost if the process stops while handling a group event.
