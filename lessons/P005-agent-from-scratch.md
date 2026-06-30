---
layout: project
lesson_id: "P005"
chapter: 2
chapter_title: "AI System Design"
project_type: "Project"
title: "Build a research agent from scratch — no framework"
description: "Estimated time: 2–3 hours"
prev: "0015-agents-from-scratch.html"
prev_title: "Building agents from scratch — no framework needed"
next: "0016-pydantic-ai.html"
next_title: "PydanticAI — type-safe AI workflows"
prereqs:
  - "[Lesson 15](0015-agents-from-scratch.html): the Agent class — use it as your starting point"
  - "[Lesson 14](0014-agentic-loops.html): agentic loops — multi-hop reasoning is the primary use case"
  - "[Lesson 12](0012-tool-calling.html): tool calling — each tool must define a proper JSON Schema"
---

## Overview

You have built the `Agent` class in Lesson 15. Now you will build a real research agent on top of it that can answer multi-hop questions requiring several chained searches and calculations. The agent must use the raw OpenAI SDK only — no LangChain, LangGraph, or PydanticAI.

A **multi-hop question** is one that requires chaining lookups — you need the answer to step 1 to know what to search for in step 2. Example: "Who is the CEO of the company that makes the Vision Pro, and how long have they held that role?"

{% include prereqs.html %}

## Specification

### Tools to implement

Your agent must have at least these five tools:

1. **`search_web(query: str) → str`** — Use either [Tavily](https://tavily.com) (free tier available) or a DuckDuckGo scraper. If using a mock, return plausible static results but note it clearly in the output.
2. **`get_current_date() → str`** — Returns today's date in ISO format.
3. **`calculate(expression: str) → str`** — Evaluates safe arithmetic.
4. **`summarise_content(content: str, focus: str) → str`** — An LLM call that distils a long search result down to the key facts relevant to `focus`. This is a "sub-call" tool — the agent calls a second LLM to process content before continuing.
5. **`check_fact(claim: str, context: str) → str`** — An LLM call that checks whether a specific claim is supported by the provided context. Returns "supported", "unsupported", or "uncertain" with a one-sentence explanation.

### Test questions

Your agent must correctly answer all four of these (or similar questions of equal complexity):

1. "What is today's date, and what day of the week is it?"
2. "If a $5,000 investment grows at 8% per year, how much will it be worth in 15 years? Show the calculation."
3. "Who founded OpenAI, and in what year? What is the current CEO's name?"
4. "What is the current version of Python, and when was it released?"

For questions 3 and 4: if using a mock search tool, write mock results that contain the correct information. The point is to verify the agent reasons through multi-step lookups correctly — not to have a live internet connection.

### Iteration logging

For each run, print a log in this format:

```text
--- Question 3 ---
[Iter 1] Tool: search_web(query="OpenAI founders") → 412 chars
[Iter 2] Tool: summarise_content(content=..., focus="founders") → 89 chars
[Iter 3] Tool: search_web(query="OpenAI current CEO 2026") → 387 chars
[Iter 4] Final answer (no tool call)
Answer: OpenAI was founded in 2015 by Sam Altman, Greg Brockman, Ilya Sutskever...
Tokens: 2,847 | Iterations: 4
```

## Requirements

- Use only the `Agent` / `Tool` classes from Lesson 15 (with any additions you need)
- No LangChain, LangGraph, or PydanticAI
- All five tools must be present and actually callable (not commented out)
- `summarise_content` and `check_fact` must make a real second LLM call (not just return the input unchanged)
- Verbose logging must show tool name, truncated arguments, and result length for each iteration
- Iteration limit: 10; timeout: 60s; token budget: 30,000 per run
- If the iteration limit is hit, print the partial answer with a clear warning

## Suggested build order

1. Copy the `Agent` and `Tool` classes from Lesson 15.
2. Implement and test each tool individually (call them directly as Python functions first).
3. Register all five tools, wire up the agent, and run test question 1 (date + day of week — simplest).
4. Run test question 2 (calculation — tests multi-step arithmetic reasoning).
5. Run test question 3 (multi-hop search + summarise — tests the sub-call tool).
6. Run test question 4 (multi-hop — tests chained lookup).
7. Add iteration logging in the format above.
8. Verify the iteration limit triggers correctly by temporarily reducing `max_iterations=2` and running question 3.

## Completion checklist

- [ ] All five tools are implemented and reachable by the agent
- [ ] `summarise_content` and `check_fact` make a second LLM call
- [ ] All four test questions produce a correct answer
- [ ] Iteration log prints for every run
- [ ] Setting `max_iterations=2` triggers the abort path correctly
- [ ] Token count prints at the end of each run

## Extension challenges

- **Real web search:** Integrate the Tavily API or DuckDuckGo so your agent can answer questions about real current events.
- **Confidence score:** After the agent produces its final answer, run a separate `check_fact` call that verifies each factual claim and produces a confidence summary.
- **Cost tracking:** Log the estimated cost of each run (based on input/output token counts and model pricing) to a JSONL file.
- **Conversational mode:** Extend to `ConversationalAgent` from Lesson 15 and maintain history across five sequential questions.
