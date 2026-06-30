---
layout: lesson
lesson_id: "0015"
chapter: 2
chapter_title: "AI System Design"
title: "Building agents from scratch — no framework needed"
description: "35–45 min read · Hands-on coding"
prev: "P004-document-pipeline.html"
prev_title: "Project: Multi-step document processing pipeline"
next: "P005-agent-from-scratch.html"
next_title: "Project: Build a research agent from scratch"
prereqs:
  - "[Lesson 12](0012-tool-calling.html): tool calling — the agent loop is built on the two-phase tool exchange"
  - "[Lesson 14](0014-agentic-loops.html): agentic loops and the ReAct pattern"
  - "[Lesson 13](0013-evaluator-router.html): evaluator-optimizer and routers — both appear as sub-components of an agent"
assignment:
  article:
    title: "Learning About AI Agents"
    url: "https://www.answer.ai/posts/2024-09-12-learning-about-ai-agents.html"
    author: "Answer.AI"
    time: "about 12 minutes"
    why: "Jeremy Howard's team at Answer.AI built agents from scratch while learning the space. This post documents what they discovered is actually needed vs what frameworks provide — a practitioner's honest comparison that mirrors exactly what this lesson teaches."
  task:
    description: "Extend the `Agent` class with an evaluator-optimizer tool."
    steps:
      - "Copy the `Agent` and `Tool` classes from this lesson"
      - "Add a `self_critique` tool that the agent can call when it wants to evaluate its own draft answer: it sends the draft to a second LLM call and returns a quality score (1–10) and improvement suggestions"
      - "Give the agent a system prompt that says: \"When answering complex questions, always call self_critique on your draft before giving the final answer. Only accept a draft with score ≥ 7.\""
      - "Test on a complex question (e.g. \"Explain the tradeoffs between microservices and monoliths for a 5-person startup\") and observe whether the agent self-critiques and improves its answer"
      - "Log each iteration including whether self_critique was called and what score it returned"
    expected: "An answer to the test question preceded by a log showing the iteration sequence and the self-critique score."
    why: "Self-critique is a real pattern used in production (it's the evaluator-optimizer from Lesson 14, implemented as a tool the agent itself decides to call). Building it into the raw agent class shows how frameworks compose these patterns transparently."
knowledge_check:
  - q: "What four things does an agent framework do that you must implement yourself in a raw agent?"
    a: "1. **Assemble the messages list** — build the system prompt and history. 2. **Run the loop** — call LLM, inspect for tool calls, execute, append results. 3. **Manage state** — store evolving context between iterations. 4. **Handle stopping** — decide when the loop ends (no tool calls, max iterations, error)."
    section: "#what-a-framework-actually-does"
    section_title: "What a framework actually does"
  - q: "How does the `Agent` class detect that the LLM wants to give a final answer instead of calling a tool?"
    a: "It checks `msg.tool_calls` on the response message. If `tool_calls` is empty or `None`, the model has no more tools to call — the loop exits and returns `msg.content` as the final answer."
    section: "#complete-agent"
    section_title: "A complete agent implementation"
  - q: "How does the `ConversationalAgent` maintain history across multiple `chat()` calls?"
    a: "It maintains a `self._history` list of past user and assistant messages. Before each run it prepends the system prompt + history to the messages, and after each run it appends the new user message and answer to the history. It trims to the last 40 messages (20 turns) to stay within the context window."
    section: "#adding-memory"
    section_title: "Adding persistent memory"
additional_resources:
  - title: "OpenAI function calling reference"
    url: "https://platform.openai.com/docs/guides/function-calling"
    desc: "The exact API contract the agent loop depends on"
  - title: "ReAct: Synergizing Reasoning and Acting in Language Models"
    url: "https://arxiv.org/abs/2210.03629"
    desc: "The original 2022 research paper introducing the ReAct pattern"
---

## Motivation

LangGraph, PydanticAI, and CrewAI — the orchestration frameworks you will explore in the next lessons — are built on top of the same raw API calls you have been writing since Lesson 2. Before you meet those frameworks, it pays to build what they build, by hand, once. That way you will know exactly what heavy lifting they are doing for you, and when a framework behaves unexpectedly, you will be able to reason about what is happening underneath rather than guessing.

This lesson walks through a complete, production-shaped agent implementation in plain Python with nothing but the OpenAI SDK and Pydantic — the exact foundation every agent framework builds on.

{% include prereqs.html %}

## What a framework actually does {#what-a-framework-actually-does}

When you write a LangGraph workflow or a PydanticAI agent, the framework does exactly four things for you:

1. **Assembles the messages list** — it builds the system prompt and history that goes to the API.
2. **Runs the loop** — it calls the LLM, inspects the response for tool calls, executes them, and appends results.
3. **Manages state** — it stores the evolving context (history, tool results, flags) somewhere between iterations.
4. **Handles stopping** — it decides when the loop ends (no more tool calls, max iterations, error).

None of these require a framework. They are straightforward Python. A hand-rolled agent is often *shorter* than a framework-based one for simple use cases, and always more debuggable.

## A complete agent implementation {#complete-agent}

The following is a production-shaped agent class. It wraps everything you built across Lessons 12–14 into a single, reusable component:

```python
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Tool definition helpers ────────────────────────────────────

@dataclass
class Tool:
    """Wraps a Python function as an agent tool."""
    name: str
    description: str
    parameters: dict          # JSON Schema for the arguments
    fn: Callable[..., Any]

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def execute(self, arguments_json: str) -> str:
        args = json.loads(arguments_json)
        try:
            result = self.fn(**args)
            return str(result)
        except Exception as e:
            return f"Tool error: {e}"


# ── Agent state ────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable state for one agent run."""
    messages: list     = field(default_factory=list)
    iterations: int    = 0
    total_tokens: int  = 0
    elapsed_s: float   = 0.0
    finished: bool     = False
    error: str | None  = None


# ── Agent ──────────────────────────────────────────────────────

class Agent:
    """
    A ReAct agent over the OpenAI API.

    Instantiate once; call run() for each new task.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful agent. Use tools to accomplish the task, then provide a final answer.",
        tools: list[Tool] = None,
        max_iterations: int = 10,
        max_tokens_per_run: int = 50_000,
        timeout_s: float = 120.0,
    ):
        self.model             = model
        self.system_prompt     = system_prompt
        self.tools             = tools or []
        self.max_iterations    = max_iterations
        self.max_tokens        = max_tokens_per_run
        self.timeout_s         = timeout_s
        self._tool_map         = {t.name: t for t in self.tools}
        self._openai_schemas   = [t.to_openai_schema() for t in self.tools]

    # ── Run ───────────────────────────────────────────────────

    def run(self, user_message: str, verbose: bool = False) -> str:
        state = AgentState()
        state.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_message},
        ]
        start = time.monotonic()

        while True:
            # ── Stopping conditions ────────────────────────────
            if state.iterations >= self.max_iterations:
                return self._abort(state, "max iterations reached")
            if state.total_tokens >= self.max_tokens:
                return self._abort(state, "token budget exceeded")
            if time.monotonic() - start > self.timeout_s:
                return self._abort(state, "timeout")

            # ── LLM call ──────────────────────────────────────
            response = client.chat.completions.create(
                model=self.model,
                messages=state.messages,
                tools=self._openai_schemas or None,
            )
            state.iterations  += 1
            state.total_tokens += response.usage.total_tokens

            msg = response.choices[0].message

            # ── No tool calls → final answer ──────────────────
            if not msg.tool_calls:
                state.finished = True
                if verbose:
                    print(f"[Agent] Done in {state.iterations} iterations, "
                          f"{state.total_tokens} tokens")
                return msg.content

            # ── Execute tool calls ────────────────────────────
            state.messages.append(msg)   # assistant message with tool_calls

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool = self._tool_map.get(tool_name)

                if tool is None:
                    result = f"Unknown tool: {tool_name}"
                else:
                    result = tool.execute(tool_call.function.arguments)
                    if verbose:
                        print(f"[Tool] {tool_name}({tool_call.function.arguments[:60]}) → {result[:80]}")

                state.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result,
                })

    def _abort(self, state: AgentState, reason: str) -> str:
        state.error = reason
        try:
            r = client.chat.completions.create(
                model=self.model,
                messages=state.messages + [{
                    "role":    "user",
                    "content": f"Stopping early: {reason}. Summarise what you found so far.",
                }],
            )
            return f"[{reason}] {r.choices[0].message.content}"
        except Exception:
            return f"[{reason}] No partial result available."
```

## Using the agent {#using-the-agent}

Define tools as `Tool` objects, instantiate the agent once, and call `run()` for each task:

```python
import datetime

# ── Define tools ────────────────────────────────────────────────

tools = [
    Tool(
        name="get_current_date",
        description="Return today's date in ISO format (YYYY-MM-DD).",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=lambda: datetime.date.today().isoformat(),
    ),
    Tool(
        name="calculate",
        description="Evaluate a safe arithmetic expression. Example: '2 + 2 * 10'.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The arithmetic expression"}
            },
            "required": ["expression"],
        },
        fn=lambda expression: str(eval(expression, {"__builtins__": {}})),
    ),
]

# ── Create and run ───────────────────────────────────────────────

agent = Agent(
    model="gpt-4o-mini",
    system_prompt=(
        "You are a research assistant. Use your tools to gather information. "
        "Always cite sources and acknowledge uncertainty."
    ),
    tools=tools,
    max_iterations=8,
    timeout_s=60.0,
)

answer = agent.run(
    "What is today's date? Also, if a $1000 investment grows 7% per year, "
    "how much is it worth after 10 years?",
    verbose=True,
)
print(answer)
```

## Adding persistent memory {#adding-memory}

The `Agent` above is stateless between `run()` calls — each call starts fresh. For conversational agents, you want to carry history across runs. The minimal approach:

```python
class ConversationalAgent(Agent):
    """Extends Agent with cross-run memory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history: list = []

    def chat(self, user_message: str, verbose: bool = False) -> str:
        answer = self.run(user_message, verbose=verbose)

        # Update history
        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant",  "content": answer})

        # Trim to last 20 turns to stay within context window
        if len(self._history) > 40:
            self._history = self._history[-40:]

        return answer
```

This is the same pattern that every chat SDK and framework uses — they just wrap it in more abstraction. Seeing the raw version makes it trivial to debug context window issues, trim strategies, and memory bugs.

## When to roll your own vs use a framework {#when-to-roll-your-own}

| Scenario | Recommendation |
|---|---|
| Simple agent, 2–5 tools, single step | Raw API — fewer moving parts |
| Agent inside a larger application | Raw API — cleaner integration |
| Strict type guarantees on result | PydanticAI — typed result enforcement |
| Complex branching, persistence, human-in-the-loop | LangGraph — built for this |
| Debugging a framework agent | Reproduce the failing case with raw API first |
| Teaching or onboarding a team | Raw API first — then introduce a framework once the team understands the fundamentals |

<div class="callout info">
<strong>Industry note:</strong> Many production AI systems at companies like Stripe, Linear, and Notion are built on raw API calls with a thin custom agent class — not on LangChain or LangGraph. Frameworks are valuable tools, not requirements.
</div>
