---
layout: lesson
lesson_id: "0007"
chapter: 1
chapter_title: "Foundations of AI Engineering"
title: "The Anthropic API — Claude and how it differs"
description: "30–40 min read · Hands-on coding"
prev: "P002-document-summarizer.html"
prev_title: "Build a document summarizer"
next: "0047-open-models-ollama-huggingface.html"
next_title: "Open models — running LLMs without an API key"
prereqs:
  - "[Lesson 2](0002-openai-api-basics.html): the OpenAI API call pattern — you will learn by analogy"
  - "[Lesson 4](0004-structured-outputs.html): Pydantic models, `model_validate_json()`"
  - "An Anthropic API key — sign up at [console.anthropic.com](https://console.anthropic.com). Add it to your `.env` file as `ANTHROPIC_API_KEY`."
  - "Install: `uv add anthropic`"
assignment:
  article:
    title: "How To Choose an LLM"
    url: "https://www.aihero.dev/how-to-choose-an-llm"
    author: "Matt Pocock (aihero.dev)"
    time: "about 10 minutes"
    why: "A practical decision framework for choosing between providers and model tiers. Directly applicable to every project in this curriculum where you have to pick a model."
  task:
    description: "Add Anthropic support to your CLI assistant from Project 1."
    steps:
      - "Add `ANTHROPIC_API_KEY` to your `.env` file"
      - "Run `uv add anthropic`"
      - "Implement the `LLMClient` abstraction from this lesson (both `OpenAIClient` and `AnthropicClient`)"
      - "Add a `--provider` flag to `assistant.py`: `--provider openai` (default) or `--provider anthropic`"
      - "Test both providers with the same question and compare responses"
    expected: "`uv run assistant.py --provider anthropic \"Explain what RAG is in one sentence.\"` returns a Claude response."
    why: "The provider-swap pattern is used in real products to A/B test models, fall back when one provider is down, and route different task types to the best model."
knowledge_check:
  - q: "What are the two structural differences between the Anthropic and OpenAI API call?"
    a: "1. The system message is a separate top-level `system=` parameter in the Anthropic API, not an item in the `messages` list. 2. `max_tokens` is required by Anthropic — the call fails without it. The OpenAI API makes it optional."
    section: "#anthropic-api"
    section_title: "The Anthropic API"
  - q: "How do you get the text of Claude's reply from the response object?"
    a: "`message.content[0].text`. Anthropic's response has a `content` list (like OpenAI's `choices`), and the first item has a `text` field (not a `message.content` nested object like OpenAI)."
    section: "#anthropic-api"
    section_title: "The Anthropic API"
  - q: "What is the recommended approach for getting structured JSON output from Claude?"
    a: "Use `client.messages.parse()` with `output_format=YourPydanticClass` — Claude now supports the same native structured outputs interface as OpenAI. A system-prompt-plus-`model_validate_json()` approach also works as a fallback. Never use output prefilling — it returns a 400 error in Claude 4.6+."
    section: "#structured-claude"
    section_title: "Structured outputs with Claude"
  - q: "What practical advantage does Claude's 200K context window give over GPT-4o's 128K?"
    a: "You can fit significantly larger inputs — entire codebases, long legal contracts, or full research papers — into a single call. This avoids chunking strategies and lets the model see the full document when reasoning, which improves accuracy on tasks that require cross-document understanding."
    section: "#claude-vs-gpt"
    section_title: "Claude vs GPT"
additional_resources:
  - title: "Anthropic Python SDK on GitHub"
    url: "https://github.com/anthropics/anthropic-sdk-python"
    desc: "Source and full reference; especially useful for tool-calling and streaming examples"
  - title: "Anthropic Messages API reference"
    url: "https://docs.anthropic.com/en/api/messages"
    desc: "All parameters, including vision, tool use, and extended thinking"
---

## Motivation

Production AI engineering teams rarely commit to a single provider. OpenAI and Anthropic both power real products, often in the same codebase — one model for document analysis, another for conversation. A job posting that says "experience with LLM APIs" means both. If you can only call GPT-4o, you are half-equipped.

Claude is Anthropic's model family, used at companies like Slack, Notion, and Quora. It handles long documents exceptionally well, is considered among the safest production models for regulated industries, and powers Claude Code — the tool you are using right now. This lesson teaches you to call it and to swap providers cleanly when a project demands it.

{% include prereqs.html %}

## Claude vs GPT — the practical differences {#claude-vs-gpt}

Both are capable LLMs that take text in and return text out. For most tasks, either works. The differences that actually matter in production:

| Dimension | GPT-4o / gpt-4o-mini | Claude Sonnet 4.6 |
|---|---|---|
| Context window | 128K tokens | 200K tokens |
| Long document handling | Good | Excellent — better at retrieving from the middle of long docs |
| Instruction following | Very good | Very good — particularly strong with structured XML prompts |
| Structured outputs | Native `.parse()` support | Native `.parse()` via `output_format` (Sonnet 4.6+) |
| Safety / refusals | Moderate | Conservative — prefers to decline ambiguous requests |
| Pricing (mid-tier model) | ~$2.50 / $10 per M tokens | ~$3.00 / $15 per M tokens (Sonnet 4.6) |

The 200K context window is Claude's most distinctive edge — you can fit entire codebases, legal contracts, or research papers into a single call.

## The Anthropic API — Messages format {#anthropic-api}

Anthropic's Python SDK is called `anthropic`. The API is similar to OpenAI's but with two structural differences: the system message is a separate top-level parameter (not in the messages list), and `max_tokens` is required (not optional).

```python
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful assistant. Be concise.",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(message.content[0].text)
```

Two differences from the OpenAI call:

- `system=` is a separate parameter — not an item in `messages`
- `max_tokens=` is required — the call will fail without it
- The response text is at `message.content[0].text`, not `message.choices[0].message.content`

## The Claude model family {#model-family}

Anthropic uses a tiered naming system — Haiku (fast/cheap), Sonnet (balanced), Opus (most capable). As of 2026, the current generation is Claude 4:

| Model ID | Best for | Rough cost (in/out per M) |
|---|---|---|
| `claude-haiku-4-5-20251001` | High-volume, simple tasks; classification; fast responses | $0.80 / $4 |
| `claude-sonnet-4-6` | The workhorse — most tasks; code generation; document analysis | $3 / $15 |
| `claude-opus-4-8` | Most capable; complex reasoning; research-grade tasks | $15 / $75 |

Use `claude-sonnet-4-6` as your default Anthropic model — the same role that `gpt-4o-mini` plays on the OpenAI side.

## Structured outputs with Claude {#structured-claude}

Claude supports native structured outputs through `client.messages.parse()` with a Pydantic `output_format` — the same interface you used for OpenAI in Lesson 4:

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    currency: str

response = client.messages.parse(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Invoice INV-001, total $500 USD."}],
    output_format=Invoice,
)

invoice = response.parsed_output
print(invoice.invoice_number)  # INV-001
print(invoice.total_amount)    # 500.0
```

`response.parsed_output` is a real `Invoice` instance with typed fields — identical to what OpenAI returns at `response.choices[0].message.parsed`.

### System prompt fallback

If you need explicit schema control or want to understand how structured outputs work under the hood, the system prompt approach is still valid: generate a JSON Schema from your Pydantic model, embed it in the system prompt, and validate the response with `model_validate_json()`:

```python
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    currency: str

schema = Invoice.model_json_schema()

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=512,
    system=f"""Extract invoice details from the text.
Respond with a JSON object matching this schema exactly. No extra text.

Schema:
{json.dumps(schema, indent=2)}""",
    messages=[
        {"role": "user", "content": "Invoice INV-001, total $500 USD."}
    ]
)

raw_json = message.content[0].text
invoice = Invoice.model_validate_json(raw_json)
print(invoice.invoice_number)  # INV-001
print(invoice.total_amount)    # 500.0
```

Claude also supports structured output through its **tool-calling API** — define a "tool" whose input schema is your Pydantic model, and Claude will call it with structured arguments. The tool-calling API is covered in depth in Chapter 2 (Lesson 12).

<div class="callout warn">
<strong>Do not use output prefilling.</strong> Claude 4.6+ returns a 400 error if you include a pre-filled assistant message (e.g. starting the assistant turn with <code>{"</code>). Use <code>client.messages.parse()</code> with <code>output_format</code>, or the system prompt approach shown above.
</div>

## Swapping providers cleanly {#swapping-providers}

A well-designed AI application abstracts the LLM provider behind a simple interface — so you can switch from OpenAI to Anthropic without changing any of the calling code. Here is a clean pattern:

```python
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        ...

class OpenAIClient(LLMClient):
    def __init__(self):
        from openai import OpenAI
        import os
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self):
        import anthropic
        import os
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

# Usage — identical regardless of provider:
def get_client(provider: str) -> LLMClient:
    if provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"Unknown provider: {provider}")

client = get_client("anthropic")
result = client.complete(
    system="You are a helpful assistant.",
    user="What is the capital of Japan?",
)
print(result)
```

This pattern appears throughout Chapter 2 and Chapter 3. The calling code never imports `openai` or `anthropic` directly — it only uses the `LLMClient` interface. Swapping providers is a one-line change.
