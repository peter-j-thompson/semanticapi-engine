# ⚡ Semantic API Engine

**Natural language interface to any API.**

Talk to APIs like you talk to a person. No SDKs, no docs, no boilerplate — just say what you want.

```python
from semanticapi import AgenticProcessor

processor = AgenticProcessor()
processor.add_provider("stripe", api_key="sk_live_...")

result = processor.process("What's my Stripe balance and show my last 3 payments")
print(result.response)
# Your Stripe balance is $4,285.92 (available) with $1,200.00 pending.
#
# Last 3 payments:
# 1. $299.00 — succeeded (Jan 15)
# 2. $49.99 — succeeded (Jan 14)
# 3. $150.00 — refunded (Jan 12)
```

## What It Does

You describe what you want in plain English. The AI agent figures out which APIs to call, calls them, handles pagination and errors, and gives you a human-readable answer.

```
"send an SMS to +1555-123-4567 saying 'Meeting at 3pm'"  →  Twilio API
"list my open GitHub issues in the react repo"            →  GitHub API
"create a $50 payment for customer cus_abc123"            →  Stripe API
"search my Notion for project planning docs"              →  Notion API
```

Works with **8 built-in providers** and any API you add via a simple JSON file.

## Quick Start

### Install

```bash
pip install semanticapi
```

Or clone and install:

```bash
git clone https://github.com/petermtj/semanticapi-engine.git
cd semanticapi-engine
pip install -e ".[all]"
```

### Run

```bash
# Set your LLM key (pick one)
export ANTHROPIC_API_KEY=sk-ant-...    # Claude (recommended)
# export OPENAI_API_KEY=sk-...         # GPT-4
# export GROQ_API_KEY=gsk_...          # Groq (fast + free tier)

# Start the server
uvicorn semanticapi.server:app --port 8080
```

### Use

```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "get my stripe balance",
    "credentials": {
      "stripe": {"api_key": "sk_live_..."}
    }
  }'
```

## Docker

```bash
# Copy and edit .env
cp .env.example .env

# Run
docker compose up
```

The server starts on port 8080 with all 8 providers loaded.

## Usage

### Python SDK

```python
from semanticapi import SemanticAPI

api = SemanticAPI()
api.add_provider("stripe", api_key="sk_live_...")
api.add_provider("twilio", account_sid="AC...", auth_token="...", from_number="+15551234567")

# Fetch data
payments = api.fetch("get my last 5 stripe payments")
print(payments.data)

# Execute actions
result = api.execute("send SMS to +15559876543 saying 'Hello!'")
print(result.success)
```

### Agentic Processor (Recommended)

The agentic processor uses an LLM to reason through multi-step queries:

```python
from semanticapi import AgenticProcessor

processor = AgenticProcessor(
    ai_provider="anthropic",   # or "openai", "groq", "ollama"
    debug=True,
)

processor.add_provider("stripe", api_key="sk_live_...")
processor.add_provider("github", access_token="ghp_...")

# Multi-step queries — the AI figures out what to call
result = processor.process("What's my Stripe balance? Also show recent payments over $100")
print(result.response)

# It asks for clarification when needed
result = processor.process("send a message to Slack")
if result.status == "needs_input":
    print(result.question)  # "Which channel should I send the message to?"
```

### One-Liner

```python
from semanticapi import process_query

result = process_query("get my stripe balance", stripe={"api_key": "sk_live_..."})
print(result.response)
```

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/health` | GET | Health check |
| `/api/query` | POST | Process a natural language query |
| `/api/providers` | GET | List loaded providers |
| `/api/providers/{name}/configure` | POST | Set provider credentials |
| `/docs` | GET | Interactive API docs (Swagger) |

## Built-in Providers

| Provider | Capabilities | Auth Type |
|----------|-------------|-----------|
| **Stripe** | Balance, payments, customers, subscriptions, refunds, invoices | API Key |
| **Twilio** | SMS, voice calls, WhatsApp, phone lookup, verification | Basic Auth |
| **OpenAI** | Chat, images, embeddings, transcription, TTS, vision | API Key |
| **GitHub** | Repos, issues, PRs, notifications, user profile | OAuth |
| **Shopify** | Products, orders, customers, inventory | Access Token |
| **Slack** | Messages, channels, users, search, status | OAuth |
| **Notion** | Search, pages, databases, blocks, users | OAuth |
| **Gmail** | Send email, read inbox, search messages | OAuth |

## Adding Custom Providers

Create a JSON file — no code needed:

```json
{
  "provider": "weatherapi",
  "name": "Weather API",
  "description": "Real-time weather data",
  "base_url": "https://api.weatherapi.com/v1",
  "auth": {
    "type": "bearer",
    "prefix": "Bearer"
  },
  "capabilities": [
    {
      "id": "current_weather",
      "name": "Get Current Weather",
      "description": "Get weather conditions for a location",
      "semantic_tags": ["weather", "temperature", "forecast"],
      "endpoint": {
        "method": "GET",
        "path": "/current.json",
        "params": {
          "q": {
            "type": "string",
            "required": true,
            "description": "City name, zip code, or coordinates"
          }
        }
      }
    }
  ]
}
```

Drop it in the `providers/` directory and restart. Done.

## Architecture

```
┌──────────────────────────────────────────┐
│              User Query                   │
│   "get my last 5 stripe payments"        │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│          Agentic Processor                │
│                                           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐ │
│  │  LLM    │→ │  Tool    │→ │  HTTP   │ │
│  │ (Claude │  │  Calling │  │ Client  │ │
│  │  GPT-4  │  │  Loop    │  │         │ │
│  │  Groq)  │  │          │  │         │ │
│  └─────────┘  └──────────┘  └─────────┘ │
│       ↑              │             │      │
│       └──────────────┘             │      │
│        Reason about results        │      │
└────────────────────────────────────┼──────┘
                                     │
                 ┌───────────────────┼───────────┐
                 ▼                   ▼           ▼
          ┌──────────┐      ┌──────────┐  ┌──────────┐
          │  Stripe  │      │  GitHub  │  │  Slack   │
          │  API     │      │  API     │  │  API     │
          └──────────┘      └──────────┘  └──────────┘
```

The engine works in a loop:

1. **Parse** — LLM understands the user's intent
2. **Plan** — Selects which API tools to call
3. **Execute** — Makes the HTTP request
4. **Reason** — Examines the result, decides if more calls are needed
5. **Respond** — Synthesizes a human-friendly answer

## Supported LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| **Anthropic** | Claude 4, Claude 3.5 Sonnet | Recommended. Best tool calling. |
| **OpenAI** | GPT-4o, GPT-4o-mini | Great alternative. |
| **Groq** | Llama 3.3 70B, Mixtral | Fast inference, free tier. |
| **Ollama** | Llama 3.2, Qwen, etc. | Local, free, private. |

```python
# Use any provider
processor = AgenticProcessor(ai_provider="groq")
processor = AgenticProcessor(ai_provider="ollama", model="llama3.2")
processor = AgenticProcessor(ai_provider="openai", model="gpt-4o")
```

## Hosted Version

Don't want to self-host? Use the managed version at **[semanticapi.dev](https://semanticapi.dev)** — includes auto-discovery of any API, OAuth flows, and a dashboard.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

The easiest way to contribute is adding a new provider — it's just a JSON file.

## License

[AGPL-3.0](LICENSE) — Free to use, modify, and self-host. If you modify the engine and offer it as a service, you must open-source your changes.
