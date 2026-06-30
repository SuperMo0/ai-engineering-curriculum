---
layout: lesson
lesson_id: "0016"
chapter: 2
chapter_title: "AI System Design"
title: "PydanticAI — type-safe AI workflows"
description: "30–40 min read · Hands-on coding"
prev: "P005-agent-from-scratch.html"
prev_title: "Project: Build a research agent from scratch"
next: "0017-langchain.html"
next_title: "LangChain — the agent framework"
prereqs:
  - "[Lesson 4](0004-structured-outputs.html): Pydantic BaseModel — PydanticAI extends this"
  - "[Lesson 14](0014-agentic-loops.html): agentic loops — PydanticAI agents implement this pattern"
  - "Install: `uv add pydantic-ai`"
assignment:
  article:
    title: "PydanticAI: Agents"
    url: "https://ai.pydantic.dev/agents/"
    author: "ai.pydantic.dev"
    time: "about 12 minutes (\"Introduction\" and \"Agents\" pages)"
    why: "The official documentation is unusually readable and example-driven for an official docs site. The dependency injection design is explained with worked examples that are clearer than any blog post."
  task:
    description: "Build a PydanticAI support triage agent."
    steps:
      - "Define a `TriageResult` Pydantic model: `category`, `priority`, `suggested_reply` (str), `escalate` (bool), `confidence`"
      - "Define a `Deps` dataclass with `customer_tier: str` (e.g. \"free\", \"pro\", \"enterprise\") and `open_ticket_count: int`"
      - "Create an agent with three tools: `get_customer_tier()` (returns `ctx.deps.customer_tier`), `get_open_tickets()` (returns `ctx.deps.open_ticket_count`), and `get_known_issues() -> str` (returns a hardcoded list of current known issues)"
      - "Run the agent on three test messages with different dep values and print the structured results"
    expected: "Three `TriageResult` objects printed as JSON, each correctly using dep values (e.g. enterprise customer gets higher priority)."
    why: "The dependency injection pattern is the key thing PydanticAI contributes over raw API code. Building it with injected context shows how to make agents testable and context-aware at the same time."
knowledge_check:
  - q: "What happens if a PydanticAI agent's result does not match the declared result type?"
    a: "PydanticAI automatically retries with a correction prompt — it tells the model what the validation error was and asks it to fix the output. This happens transparently; you get a validated result object or an exception after the retry budget is exhausted. You never receive an unvalidated string."
    section: "#basic-agent"
    section_title: "A basic PydanticAI agent"
  - q: "What is the purpose of dependencies in PydanticAI, and how do tools access them?"
    a: "Dependencies are a typed dataclass passed when running the agent — they hold runtime resources like database connections, API clients, or user-specific context. Tools access them via a `RunContext[Deps]` first argument: `ctx.deps.your_field`. This avoids globals and makes the agent fully testable with injected mocks."
    section: "#dependencies"
    section_title: "Dependencies"
  - q: "When is PydanticAI a better choice than LangGraph?"
    a: "When building a backend service or API endpoint that needs clean typed interfaces, dependency injection for testability, and Python-idiomatic code. PydanticAI is better when you value type safety and testing over visual workflow graphs. LangGraph wins when you need explicit branching, state persistence, or human-in-the-loop."
    section: "#pydantic-ai-vs-langgraph"
    section_title: "PydanticAI vs LangGraph"
additional_resources:
  - title: "PydanticAI documentation"
    url: "https://ai.pydantic.dev/"
    desc: "Full reference including multi-agent orchestration, testing utilities, and streaming"
---

## Motivation

LangGraph gives you stateful graph workflows. PydanticAI gives you something different: agents defined as typed Python objects where the result type, dependency type, and every tool input and output are all enforced at runtime. If your agent returns the wrong type or a tool is called with a missing argument, Python catches it immediately — not after a production incident.

PydanticAI is built by the Pydantic team and integrates natively with Pydantic validation. It is the cleaner choice for backend services where type safety and testability are priorities over visual workflow graphs.

{% include prereqs.html %}

## Core concepts {#core-concepts}

PydanticAI has three central ideas that differ from raw API code:

- **Typed result:** an agent is declared with a result type (`Agent[Deps, ResultType]`). The agent only returns when it produces a value that validates against that type.
- **Typed dependencies:** data your agent needs at runtime (database connections, API clients, config) is passed as a typed **dependency** object — not a global or an environment variable.
- **Typed tools:** tool functions are registered on the agent; their parameters are automatically converted to tool schemas from Python type annotations.

## A basic PydanticAI agent {#basic-agent}

The simplest agent: no tools, no dependencies — just a typed result:

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class SentimentResult(BaseModel):
    sentiment: str       # "positive" | "neutral" | "negative"
    score: float         # 0.0 (negative) to 1.0 (positive)
    explanation: str

agent = Agent(
    "openai:gpt-4o-mini",
    result_type=SentimentResult,
    system_prompt="Analyse the sentiment of the given text.",
)

result = agent.run_sync("I absolutely love this product! Best purchase of the year.")
print(result.data.sentiment)     # positive
print(result.data.score)         # 0.92
print(result.data.explanation)   # "Strongly positive language..."
```

`result.data` is a validated `SentimentResult` instance — not a string, not a dict. If the model returns something that cannot be parsed as `SentimentResult`, PydanticAI automatically retries with a correction prompt.

## Adding tools {#tools}

Tools are registered with the `@agent.tool` decorator. The function's parameters become the tool's schema automatically — no manual JSON Schema writing:

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
import datetime

class ResearchResult(BaseModel):
    answer: str
    sources_consulted: list[str]
    confidence: str   # "high" | "medium" | "low"

agent = Agent(
    "openai:gpt-4o-mini",
    result_type=ResearchResult,
    system_prompt="You are a research assistant. Use tools to gather information.",
)

@agent.tool_plain   # tool_plain = no RunContext needed
def get_current_date() -> str:
    """Return today's date in ISO format."""
    return datetime.date.today().isoformat()

@agent.tool_plain
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Example: '2 + 2 * 10'"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

@agent.tool_plain
def search_web(query: str) -> str:
    """Search the web for information. Returns relevant snippets."""
    # Replace with Tavily or another search API in production
    return f"[Mock results for: {query}]"

result = agent.run_sync("What is today's date and what is 144 / 12?")
print(result.data.answer)
```

The docstring on each tool function becomes the tool's description — the same text the model uses to decide when to call it. Write docstrings that explain the trigger condition, not just what the function does.

## Dependencies — injecting runtime context {#dependencies}

Dependencies are the PydanticAI way to pass runtime resources to tools without using globals. Define a dataclass for the dependencies, pass it when running the agent, and receive it in tools via `RunContext`:

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class Deps:
    user_id: str
    db_connection: object   # your real DB client

class SupportResponse(BaseModel):
    reply: str
    escalate: bool
    ticket_id: str

agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=Deps,
    result_type=SupportResponse,
    system_prompt="You are a support agent with access to customer data.",
)

@agent.tool
async def get_customer_orders(ctx: RunContext[Deps], limit: int = 5) -> str:
    """Retrieve the customer's recent orders from the database."""
    # ctx.deps gives access to the injected dependencies
    orders = await ctx.deps.db_connection.fetch(
        "SELECT * FROM orders WHERE user_id = ? LIMIT ?",
        ctx.deps.user_id, limit
    )
    return str(orders)

# Run the agent with injected dependencies
import asyncio

async def handle_support(user_id: str, message: str, db) -> SupportResponse:
    deps = Deps(user_id=user_id, db_connection=db)
    result = await agent.run(message, deps=deps)
    return result.data
```

Dependencies make agents testable: in tests, inject a mock database and assert on the structured result without real API calls.

## PydanticAI vs LangGraph — choosing between them {#pydantic-ai-vs-langgraph}

| | PydanticAI | LangGraph |
|---|---|---|
| Mental model | Typed Python agent | Explicit state graph |
| Workflow visibility | Implicit (inside agent loop) | Explicit (nodes and edges) |
| Type safety | Excellent — all inputs/outputs enforced | Good — state is TypedDict |
| Testability | Excellent — dependency injection | Good — can test nodes in isolation |
| Human-in-the-loop | Manual implementation | Native with `interrupt()` |
| Best for | Backend services, clean typed APIs | Complex multi-step workflows with branching |

Neither is universally better. PydanticAI is the right choice when you are building a backend service that needs clean typed interfaces and easy testing. LangGraph is the right choice when you need visible, debuggable graph execution with branching and persistence.
