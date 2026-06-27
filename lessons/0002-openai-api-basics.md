---
layout: lesson
lesson_id: "0002"
chapter: 1
chapter_title: "Foundations of AI Engineering"
title: "Your first LLM API call — how OpenAI works"
description: "30–40 min read · Hands-on coding"
prev: "0001-ai-dev-environment.html"
prev_title: "Setting up your AI engineering environment"
next: "0003-chat-and-history.html"
next_title: "Conversations and chat history"
prereqs:
  - "[Lesson 1](0001-ai-dev-environment.html): uv project created, `openai` and `python-dotenv` installed, `.env` file set up"
  - "A real OpenAI API key — sign up at [platform.openai.com](https://platform.openai.com) and create one under API Keys"
  - "A small amount of API credit ($5 is more than enough for this entire chapter)"
assignment:
  article:
    title: "What Is an LLM?"
    url: "https://www.aihero.dev/what-is-an-llm"
    author: "Matt Pocock (aihero.dev)"
    time: "about 10 minutes"
    why: "Written by a practitioner, not a researcher. It explains what LLMs are in terms that map to your job as an engineer: what you give them, what you get back, and where their limits are. Read it after making your first API call so the vocabulary clicks."
  task:
    description: "Write a script that takes a question from the command line, sends it to `gpt-4o-mini`, and prints the answer plus the token count."
    steps:
      - "In your `ai-foundations` project, edit `main.py`"
      - "Use `import sys` and `sys.argv[1]` to read the question from the command line"
      - "Call the API with that question and print the response text"
      - "Below the response, print: `Tokens used: N`"
      - "Add a `try/except` that catches `AuthenticationError` and prints a helpful message"
    expected: "`uv run main.py \"What are three uses of LLMs in production software?\"`"
    why: "Command-line argument handling and token logging are skills you will use in every project throughout the curriculum. Building them now makes later projects faster."
knowledge_check:
  - q: "What is a token, and why do tokens matter for AI engineering?"
    a: "A token is the unit LLMs use to measure and process text — roughly four characters or three-quarters of a word. Tokens matter because LLM providers charge per token: the more text you send and receive, the more it costs. Token counts also determine whether your input fits within a model's context window."
    section: "#what-is-an-llm"
    section_title: "What is an LLM?"
  - q: "What is the `client` object and when is it created?"
    a: "The `client` is an `OpenAI()` instance — a Python object that holds your API key and knows how to send requests to OpenAI's API. It is created once at the top of the script (or module) and reused for every API call. Creating it is not the same as making an API call."
    section: "#authentication"
    section_title: "Authentication"
  - q: "What two things does every chat completions call require?"
    a: "Every call to `client.chat.completions.create()` requires a `model` name (which LLM to use) and a `messages` list (the conversation, as a list of dictionaries with `role` and `content`)."
    section: "#chat-completions"
    section_title: "The chat completions endpoint"
  - q: "How do you extract the text of the model's reply from the response object?"
    a: "`response.choices[0].message.content`. The response contains a `choices` list (usually one item), and each choice contains a `message` object with a `content` field holding the text."
    section: "#reading-response"
    section_title: "Reading the response"
  - q: "Which exception should you catch to detect a wrong API key?"
    a: "`AuthenticationError`, imported from the `openai` package. This is raised when the API key is missing, invalid, or revoked. Always catch it separately from generic `OpenAIError` so you can give the user a clear, actionable error message."
    section: "#handling-errors"
    section_title: "Handling errors"
additional_resources:
  - title: "What Are LLMs Used For?"
    url: "https://www.aihero.dev/what-are-llms-used-for"
    desc: "Companion to the assignment read; covers the range of production use cases you will build in this curriculum"
  - title: "OpenAI Python SDK on GitHub"
    url: "https://github.com/openai/openai-python"
    desc: "Source code and full API reference for the `openai` package"
  - title: "OpenAI Cookbook"
    url: "https://cookbook.openai.com/"
    desc: "Curated examples of real API usage patterns; useful as a reference after you have the basics down"
  - title: "Tiktokenizer"
    url: "https://tiktokenizer.vercel.app/"
    desc: "Interactive tokenizer — paste any text to see exactly how OpenAI models split it into tokens and how many tokens it costs"
---

## Motivation

The most common task in AI engineering is deceptively simple: send text in, get text back. A customer sends a support message — your code sends it to an LLM, gets a draft reply, and routes it to the right team. A user uploads a contract — your code sends the text to an LLM, gets a structured summary, and stores it in a database.

Everything else in this curriculum — agents, RAG, production deployments — is built on top of that one primitive: call the LLM, read the response. This lesson teaches you exactly how that works, from the first line of code to handling errors gracefully.

{% include prereqs.html %}

## What is an LLM? {#what-is-an-llm}

A **Large Language Model** — LLM for short — is a type of AI model that takes text as input and produces text as output. When you type a question into ChatGPT and get an answer, you are using an LLM. When GitHub Copilot suggests the next line of code, that is an LLM. When Google translates a document, that is an LLM.

Under the hood, an LLM is a massive mathematical function — billions of numbers (called parameters) arranged so that given some input text, the model can predict what text should come next. It does this one small piece at a time. Those small pieces are called **tokens**.

A **token** is the unit an LLM uses to measure and process text. A token is roughly four characters — about three-quarters of an average English word. (Words average five or more characters plus spaces, so one token covers slightly less than one word — four characters per token × 1,000,000 tokens = 4,000,000 characters; at five-plus characters per word, that is roughly 750,000 words.) The word "hello" is one token. The word "unfortunately" is four tokens. A full paragraph might be 80–100 tokens. Tokens matter because LLM providers charge you per token — more text means more cost.

As an AI engineer, you will never train or modify an LLM. These models take months and millions of dollars to train, and that work is done by companies like OpenAI and Anthropic. Your job is to call them via an API — the same way a web developer calls a weather API, not writes the meteorological simulation.

## What is an API? {#what-is-an-api}

An **API** (Application Programming Interface) is a way for one piece of software to talk to another over the internet. When your app needs today's weather, it sends a request to a weather service's API and receives a structured response. Your code did not compute the weather — it asked a service that already knows it.

The OpenAI API works the same way. Your Python code sends a request to OpenAI's servers. Those servers run the LLM, generate a response, and send it back to your code. The whole round trip takes one to five seconds for most requests.

OpenAI's API is a standard web API — it works over HTTP, the same protocol your browser uses to load websites. The Python `openai` package is a library that handles all the HTTP details for you: it formats the request, sends it, waits for the response, and returns a clean Python object.

## Authentication — proving who you are {#authentication}

Every request you make to the OpenAI API must include your API key. OpenAI's servers check the key, identify your account, and charge any usage to you. Without the key, the request is rejected.

The `openai` library reads your API key when you create a client object. The pattern looks like this:

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

The `OpenAI()` call creates a client — an object that knows your API key and knows how to send requests. You create it once at the top of your script and reuse it for every API call. The key comes from the environment (loaded by `load_dotenv()`), never from the source code itself.

<div class="callout info">
<strong>Shortcut:</strong> If your API key is in the environment variable <code>OPENAI_API_KEY</code>, the <code>openai</code> library picks it up automatically — you can write <code>client = OpenAI()</code> with no argument. Explicit is clearer for learning, so we pass it explicitly here.
</div>

## Models — which LLM are you calling? {#models}

OpenAI offers several different LLMs, and you specify which one you want on every API call. Each LLM is called a **model**. Different models have different capabilities, speeds, and costs.

The main models you will use in this curriculum:

| Model | Best for | Cost |
|-------|----------|------|
| `gpt-4o-mini` | Learning, prototyping, simple tasks | Very cheap (~$0.15 per million tokens in) |
| `gpt-4o` | Complex reasoning, production tasks | Moderate (~$2.50 per million tokens in) |

Start with `gpt-4o-mini` for all your experiments. It is fast, inexpensive, and capable enough for almost every task in this chapter. You can always upgrade to `gpt-4o` for a specific task that needs more reasoning power.

<div class="callout info">
A million tokens is about 750,000 words — roughly ten full novels. At those prices, the coding tasks in this curriculum will cost you less than $1 total.
</div>

## The chat completions endpoint {#chat-completions}

Every call to an LLM goes through the **chat completions endpoint**. "Endpoint" is the API term for "the specific URL and action you are calling" — think of it like a function name for the API. The chat completions endpoint is OpenAI's main interface: you send a conversation, and the model replies.

A chat completions call requires two things: the model name and a list of messages. The messages list represents the conversation. Here is the simplest possible call:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
```

Each message in the list is a dictionary with two keys:

- `role` — who is speaking. For a single question you only need `"user"`. Two other roles exist: `"assistant"` (the model's previous replies) and `"system"` (background instructions) — both are introduced in Lesson 3 when you build multi-turn conversations.
- `content` — the text of that message.

For a single question with no history, the messages list contains just one entry with role `"user"`.

## Reading the response {#reading-response}

The API call returns a Python object — a structured response with several fields. The text you want lives at a specific path inside it:

```python
text = response.choices[0].message.content
print(text)
```

Why such a long path? Because the response is designed to support several responses at once (the `choices` list), and each choice contains a full message object (just like the ones you send). For now, there is always exactly one choice, at index `[0]`.

The response also tells you how many tokens the call used — useful for tracking cost:

```python
print(response.usage.prompt_tokens)      # tokens in your messages
print(response.usage.completion_tokens)  # tokens in the reply
print(response.usage.total_tokens)       # sum of both
```

### Complete working example

Here is every piece together — a complete script that makes a real API call and prints the result:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain what an API is in two sentences."}
    ]
)

text = response.choices[0].message.content
print(text)
print(f"\nTokens used: {response.usage.total_tokens}")
```

Run it with:

```bash
uv run main.py
```

You should see the model's explanation of APIs, followed by the token count. If you see that, your entire environment — key, package, network, billing — is working.

## Handling errors {#handling-errors}

API calls fail. Networks drop. Keys expire. Rate limits get hit. A professional script handles these cases instead of crashing with a confusing stack trace.

The `openai` package raises specific exception types for different failure modes. The three you will encounter most often:

| Exception | What it means | What to do |
|-----------|---------------|------------|
| `AuthenticationError` | API key is wrong or missing | Check your `.env` file — copy the key fresh from the dashboard |
| `RateLimitError` | Too many requests in too short a time | Wait a moment and retry; add a small delay in automated scripts |
| `OpenAIError` | Any other API error (catch-all) | Log the error message and surface it to the user or a monitoring system |

Wrap your API calls in a `try/except` block:

```python
from openai import OpenAI, AuthenticationError, RateLimitError, OpenAIError

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

except AuthenticationError:
    print("Error: invalid API key. Check your .env file.")
except RateLimitError:
    print("Error: rate limit hit. Wait a moment and try again.")
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
```

In production AI systems, errors go to a structured logging system rather than `print()` — but the pattern is the same. Catch the specific exceptions you can handle, catch the generic one as a fallback.

## Anatomy of a complete, production-ready call

Here is what a well-written API call looks like when all the pieces come together. This is the pattern you will follow throughout the curriculum:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, RateLimitError, OpenAIError

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask(question: str) -> str:
    """Send a single question to gpt-4o-mini and return the text reply."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    except AuthenticationError:
        raise RuntimeError("Invalid API key — check your .env file.")
    except RateLimitError:
        raise RuntimeError("Rate limit hit — wait a moment and retry.")
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI error: {e}") from e


if __name__ == "__main__":
    answer = ask("What is a token in the context of LLMs?")
    print(answer)
```

A few design choices worth noting:

- The API call is wrapped in a function (`ask`) so the rest of your code does not know or care about HTTP details.
- Errors are re-raised as `RuntimeError` with plain English messages — the caller does not need to import `openai` exceptions to handle them.
- The `if __name__ == "__main__"` guard lets this file be imported by other modules without immediately making an API call.
