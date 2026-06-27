---
layout: lesson
lesson_id: "0004"
chapter: 1
chapter_title: "Foundations of AI Engineering"
title: "Structured outputs and function calling"
description: "30–40 min read · Hands-on coding"
prev: "0003-chat-and-history.html"
prev_title: "Conversations and chat history"
next: "P001-cli-assistant.html"
next_title: "Build a CLI AI assistant"
prereqs:
  - "[Lesson 2](0002-openai-api-basics.html): `client.chat.completions.create()`, reading responses"
  - "[Lesson 3](0003-chat-and-history.html): system messages, the messages list"
  - "Familiarity with Python classes — you will define a Pydantic model (a class). Install: `uv add pydantic`"
assignment:
  article:
    title: "Structured Outputs from LLMs"
    url: "https://www.boundaryml.com/blog/structured-output-from-llms"
    author: "BoundaryML blog"
    time: "about 10 minutes"
    why: "A clear practitioner explanation of why structured outputs matter and how different approaches compare — including why schema-constrained generation beats prompt-only approaches for reliability."
  task:
    description: "Build a CV (résumé) parser using Structured Outputs."
    steps:
      - "Define a `CVSummary` Pydantic model with fields: `name`, `email`, `years_of_experience` (int), `top_skills` (list of str, max 5), `most_recent_role` (str), `seniority_level` (str: \"junior\", \"mid\", \"senior\", \"lead\")"
      - "Write a `parse_cv(text: str) -> CVSummary` function using `.parse()`"
      - "Test it with a short made-up CV (paste a few sentences of fake experience)"
      - "Print all six fields from the returned object"
    expected: "Six lines, one per field, with properly typed values (e.g. `years_of_experience` is an `int`, not a string)."
    why: "CV parsing is a real AI engineering use case (ATS systems, recruiters). It also tests every feature from this lesson: required fields, typed values, and a list field."
knowledge_check:
  - q: "Why do free-text LLM responses break production code?"
    a: "Because models are non-deterministic — the same prompt can produce differently worded responses. Any code that parses prose with string operations will break when phrasing changes. Structured outputs constrain the model to a guaranteed schema, making the response machine-readable every time."
    section: "#why-free-text-breaks"
    section_title: "Why free-text responses break production code"
  - q: "What method do you use for Structured Outputs with OpenAI, and how does it differ from `.create()`?"
    a: "`client.chat.completions.parse()`. You pass a Pydantic class as `response_format`. Instead of a string in `response.choices[0].message.content`, the result is a Pydantic object at `response.choices[0].message.parsed` — already typed and validated."
    section: "#structured-outputs-api"
    section_title: "OpenAI Structured Outputs"
  - q: "How do you mark a field as optional in a Pydantic model?"
    a: "Use `type | None` syntax and give the field a default of `None`: e.g. `vendor_name: str | None = None`. The model will set it to `None` if the information is not present in the input."
    section: "#optional-fields"
    section_title: "Optional fields and nested models"
  - q: "What changed in Claude 4.6+ that makes the output-prefilling trick unusable?"
    a: "Claude 4.6+ removed the ability to prefill the assistant's reply (starting the reply with `{\"` to force JSON). Attempting it now returns a 400 error. The correct approaches are: the native `client.messages.parse()` with `output_format`, Anthropic's `output_config` API, or the tool-calling API."
    section: "#claude-note"
    section_title: "What about Claude?"
additional_resources:
  - title: "OpenAI: Structured Outputs guide"
    url: "https://platform.openai.com/docs/guides/structured-outputs"
    desc: "Full reference including JSON Schema mode, supported types, and known limitations"
  - title: "Pydantic documentation"
    url: "https://docs.pydantic.dev/latest/"
    desc: "Validators, field constraints (`Field(max_length=...)`), and model serialisation"
---

## Motivation

Free-text LLM responses are useless to a program. If your backend asks a model to extract the sender's name from an email, you cannot save a paragraph to a database column. You need a specific field, reliably formatted, every time. Production AI systems almost always need structured data — JSON objects with predictable fields — not prose.

Structured outputs are what let you chain an LLM into a real system: extract a name → look it up in a database → send a reply. Every step expects a specific shape. This lesson teaches the standard approach used in 2026 production code.

{% include prereqs.html %}

## Why free-text responses break production code {#why-free-text-breaks}

Suppose you ask a model: "Extract the invoice number and total amount from this email." With a plain text call, you might get:

```text
"The invoice number is INV-2024-0042 and the total amount is $1,250.00."
```

To use this in code, you would need to parse that sentence — find the number after "invoice number is", find the amount after "total amount is". This parsing breaks the moment the model rephrases the sentence, adds an extra word, or changes punctuation. And models do rephrase — they are non-deterministic by nature.

What you actually want is:

```python
{"invoice_number": "INV-2024-0042", "total_amount": 1250.00}
```

A Python dictionary with predictable keys and typed values. No parsing. No fragility. This is what structured outputs give you.

## Pydantic — defining the shape you want {#pydantic}

**Pydantic** is a Python library for defining data structures with types and validation. You describe the shape of the data you want, and Pydantic ensures that any data you receive matches that shape — raising a clear error if it does not.

A Pydantic model is a Python class that inherits from `BaseModel`. Each field is an attribute with a type annotation:

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    currency: str
    due_date: str
```

This class says: "I expect a dict with exactly these four fields, with these types." Pydantic validates incoming data against it and converts compatible types automatically (e.g. a string `"1250.00"` becomes a `float`).

Pydantic is already installed if you have `openai` (it is a dependency), but add it explicitly so it is in your project's requirements: `uv add pydantic`.

## OpenAI Structured Outputs {#structured-outputs-api}

OpenAI's Structured Outputs feature lets you pass a Pydantic model directly to the API. The model is converted to a JSON Schema, sent to the API as a constraint, and the response is *guaranteed* to match that schema. You get back a Pydantic object, not a string.

The method is `client.chat.completions.parse()` — note `.parse()` instead of `.create()`. You pass your Pydantic class as `response_format`:

```python
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    currency: str

email_text = """
Hi, please find attached invoice INV-2024-0042.
Total due: $1,250.00 USD by 30 January 2025.
"""

response = client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract invoice details from the text."},
        {"role": "user",   "content": email_text}
    ],
    response_format=Invoice,
)

invoice = response.choices[0].message.parsed
print(invoice.invoice_number)   # INV-2024-0042
print(invoice.total_amount)     # 1250.0
print(invoice.currency)         # USD
```

`response.choices[0].message.parsed` returns a real `Invoice` instance — a Python object with proper types. You can access fields with dot notation. No JSON parsing, no key errors, no type casting.

## Optional fields and nested models {#optional-fields}

Real data is messy — not every field will always be present. Mark a field as optional using `type | None` and give it a default of `None`:

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    currency: str
    due_date: str | None = None    # may not be in every email
    vendor_name: str | None = None
```

You can also nest Pydantic models for structured sub-objects:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    billing_address: Address | None = None
```

The API handles nested models correctly — `billing_address` will be an `Address` instance or `None`, never a raw dict.

## Lists in structured output

When the output is naturally a collection — a list of extracted items, a list of categories, a list of action items — use Python's `list` type in the model:

```python
from pydantic import BaseModel

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float

class Invoice(BaseModel):
    invoice_number: str
    line_items: list[LineItem]
    total_amount: float
```

`invoice.line_items` will be a Python list of `LineItem` objects — fully typed, fully iterable.

## What about Claude? {#claude-note}

Claude (covered in depth in Lesson 7) now supports native structured outputs through the same `.parse()` pattern. Pass a Pydantic class as `output_format` and read the result from `response.parsed_output`:

```python
from pydantic import BaseModel
from anthropic import Anthropic

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    currency: str

client = Anthropic()
response = client.messages.parse(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Invoice INV-001, total $500 USD."}],
    output_format=Invoice,
)
print(response.parsed_output.invoice_number)  # INV-001
```

The pattern is nearly identical to OpenAI's. The key differences (covered fully in Lesson 7): `max_tokens` is required, and the system message is a separate `system=` parameter rather than an item in the `messages` list.

<div class="callout warn">
<strong>Claude 4.6+ removed output prefilling.</strong> Older tutorials show a technique where you start the assistant's reply with <code>{"</code> to force JSON output. This now returns a 400 error. Use the native <code>client.messages.parse()</code> with <code>output_format</code>, Anthropic's <code>output_config</code> API, or the tool-calling API instead.
</div>

## Complete example: email triage

Here is a real-world pattern — extracting structured data from an unstructured email and using it in code:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SupportTicket(BaseModel):
    subject: str
    priority: str           # "low", "medium", "high", "urgent"
    category: str           # "billing", "technical", "account", "other"
    customer_name: str | None = None
    one_line_summary: str

def triage_email(email_body: str) -> SupportTicket:
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a support triage assistant. "
                    "Extract structured information from support emails."
                ),
            },
            {"role": "user", "content": email_body},
        ],
        response_format=SupportTicket,
    )
    return response.choices[0].message.parsed


email = """
Hi, I've been charged twice for my subscription this month.
My name is Sarah Chen and I'm really frustrated.
This needs to be fixed immediately — I can't afford these duplicate charges.
"""

ticket = triage_email(email)
print(f"Priority: {ticket.priority}")
print(f"Category: {ticket.category}")
print(f"Customer: {ticket.customer_name}")
print(f"Summary:  {ticket.one_line_summary}")
```
