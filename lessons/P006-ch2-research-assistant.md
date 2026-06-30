---
layout: project
lesson_id: "P006"
chapter: 2
chapter_title: "AI System Design"
project_type: "Chapter Project"
title: "Automated Research Assistant"
description: "Estimated time: 5–8 hours · Portfolio-grade · Deployable"
prev: "0048-n8n-zapier.html"
prev_title: "Visual AI workflows — n8n and Zapier"
next: "0018-fastapi-basics.html"
next_title: "Serving AI with FastAPI"
prereqs:
  - "All of Chapter 2 (Lessons 8–17, 50–51) — this project integrates all of it"
  - "P005 completed — the raw agent loop you built there is a sub-component of this system"
  - "A working search API — Tavily free tier recommended (tavily.com), or use a mock for offline testing"
---

## Overview

This is the Chapter 2 capstone. It integrates every pattern from the chapter into a single, deployable system that is demonstrably more powerful than a single LLM call:

- **LLM router** (Lesson 13) — classifies incoming research queries
- **Parallel information gathering** (Lesson 11) — runs multiple searches concurrently
- **Agentic loop** (Lesson 14) — iterates until enough evidence is gathered
- **Evaluator-optimizer** (Lesson 13) — scores and improves the final report
- **Structured output** (Lesson 4) — produces a typed, well-formed report
- **Orchestration** (Lesson 50 or 16) — the outer shell uses either LangGraph or PydanticAI (your choice)

The finished system accepts a research question via CLI, runs a multi-step pipeline to gather and synthesise information, and produces a structured research report in both Markdown and JSON.

<div class="callout info">
<strong>Portfolio note:</strong> This is a project worth showing in interviews. The combination of router + parallel search + evaluator demonstrates applied AI system design at a level most candidates cannot show.
</div>

{% include prereqs.html %}

## System design {#system-design}

The system has four stages:

```
User query
    │
    ▼
[1. Router]  — classifies query as "factual", "analytical", or "opinion"
    │
    ▼
[2. Research Agent]  — parallel searches, summarisation, fact-checking loop
    │                  (runs until evidence quality threshold is met)
    ▼
[3. Synthesis]  — drafts a structured report from gathered evidence
    │
    ▼
[4. Evaluator]  — scores the report; requests improvement if score < 8/10
    │
    ▼
Structured ResearchReport (JSON + Markdown)
```

### Stage 1 — Router

Classify the incoming query into one of three types. Each type receives a different system prompt in Stage 2:

- **factual** — a question with a definite correct answer ("When did X happen?"). Prioritise finding primary sources.
- **analytical** — a comparison or evaluation ("How does X compare to Y?"). Prioritise gathering multiple perspectives and data points.
- **opinion** — a "should I" or "what do you think" query. Gather expert opinions, cite sources, present a balanced view.

### Stage 2 — Research Agent

Run an agentic loop with these tools (same as P005, with one addition):

- `search_web(query)`
- `summarise_content(content, focus)`
- `check_fact(claim, context)`
- `get_current_date()`
- `store_evidence(key, content)` — **new:** stores a piece of evidence in a shared dictionary that the synthesis stage reads. This is the agent's "notepad".

The system prompt for the research agent instructs it to:

1. Break the research question into sub-questions
2. Search for each sub-question and store relevant evidence
3. Fact-check any claims that seem uncertain before storing them
4. Indicate when it has gathered sufficient evidence by producing a final message (no tool call) that says "RESEARCH COMPLETE" followed by a brief summary

### Stage 3 — Synthesis

After the agent loop exits, run a dedicated synthesis prompt that reads all stored evidence and produces a `ResearchReport`:

```python
from pydantic import BaseModel

class Source(BaseModel):
    title: str
    url: str | None
    relevance: str       # one sentence

class ResearchReport(BaseModel):
    query: str
    query_type: str      # factual | analytical | opinion
    summary: str         # 2–3 paragraph executive summary
    key_findings: list[str]   # bullet points, 5–10 items
    sources: list[Source]
    confidence: str      # "high" | "medium" | "low"
    limitations: str     # what this report does not cover
    generated_at: str    # ISO datetime
```

### Stage 4 — Evaluator

Score the report on four criteria (each 1–10), producing an overall average:

- **Completeness** — does it answer the question fully?
- **Accuracy** — are the claims well-supported by the evidence?
- **Clarity** — is it well-structured and readable?
- **Sourcing** — does it cite specific sources for key claims?

If the average score is below 8, send the report back to Stage 3 (synthesis) with the evaluator's feedback. Maximum 3 synthesis-evaluation iterations.

## Usage {#usage}

```bash
# Single research question
uv run research.py "What are the main differences between PydanticAI and LangGraph?"

# Save report to files
uv run research.py "How does vector search work?" --output reports/

# Verbose mode (shows all agent iterations)
uv run research.py "Who are the main players in the AI chip market?" --verbose

# Use a specific orchestration backend
uv run research.py "..." --backend langgraph   # or --backend pydantic-ai
```

**Sample output:**

```text
[Router] Query type: analytical (confidence: high)
[Research] Starting information gathering...
  Iter 1: search_web("PydanticAI vs LangGraph comparison") → 2,341 chars
  Iter 2: summarise_content(...) → 312 chars → stored as "pydanticai_overview"
  Iter 3: search_web("LangGraph features 2026") → 1,847 chars
  Iter 4: check_fact("LangGraph supports human-in-the-loop", ...) → supported
  Iter 5: store_evidence("langgraph_hitl", ...) → stored
  Iter 6: RESEARCH COMPLETE — gathered 5 evidence items
[Synthesis] Drafting report...
[Evaluator] Scores: completeness=7, accuracy=9, clarity=8, sourcing=6 → avg 7.5
[Synthesis] Retry with feedback: "Add more specific source citations"
[Evaluator] Scores: completeness=8, accuracy=9, clarity=8, sourcing=8 → avg 8.25 ✓
[Report] Saved to reports/2026-06-24-pydanticai-vs-langgraph.{md,json}

══════════════════════════════════════════════
RESEARCH REPORT
══════════════════════════════════════════════
Query: What are the main differences between PydanticAI and LangGraph?
Type:  analytical | Confidence: medium

SUMMARY
PydanticAI and LangGraph represent different philosophies in AI workflow
orchestration. PydanticAI prioritises type safety and testability through
Python's native type system...

KEY FINDINGS
• PydanticAI enforces typed result models — agents cannot return
  unvalidated output
• LangGraph models workflows as explicit state graphs with visible
  nodes and edges
• PydanticAI offers superior dependency injection; LangGraph has
  native human-in-the-loop via interrupt()
...

SOURCES
1. PydanticAI Agents documentation (ai.pydantic.dev/agents/)
2. LangGraph: Multi-Agent Workflows (blog.langchain.dev/langgraph/)
...

LIMITATIONS
This report covers the libraries as of 2026-06-24. Both are actively
developed and APIs may change.
══════════════════════════════════════════════
```

## Orchestration choice: LangGraph vs PydanticAI {#orchestration-choice}

The outer pipeline (router → research → synthesis → evaluator loop) can be built with either LangGraph or PydanticAI. You must implement one of them — not raw API — for the outer shell. The research agent loop inside Stage 2 must remain raw API (no framework).

### If you choose LangGraph

Model the four stages as a state graph. Use `add_conditional_edges` for the evaluator-optimizer loop. State should include the research evidence dict, the current report, and the iteration count.

### If you choose PydanticAI

Build the outer system as a PydanticAI agent whose result type is `ResearchReport`. The router, research, and evaluator are tools registered on this outer agent. Dependencies include a shared evidence store (dataclass).

<div class="callout info">
<strong>Recommendation:</strong> If your goal is to demonstrate LangGraph for job applications, choose LangGraph. If you prefer clean Python types and dependency injection, choose PydanticAI. Either is equally valid — just be ready to explain your choice.
</div>

## Requirements {#requirements}

- All four pipeline stages (router, research, synthesis, evaluator) must be present and functional
- The outer orchestration uses LangGraph or PydanticAI (not raw API)
- The Stage 2 research agent loop uses raw OpenAI API only (no framework inside the loop)
- Output includes both a `.md` (Markdown) and a `.json` (ResearchReport JSON) file
- Evaluator-optimizer loop runs at most 3 synthesis cycles
- The router correctly classifies at least "factual" and "analytical" queries
- Verbose mode logs all agent iterations including tool names, truncated arguments, and result lengths
- The system handles search API errors gracefully — if search fails, the agent logs the error and uses a "search unavailable" tool result rather than crashing
- Total cost per report is printed at the end

## Suggested build order {#build-order}

1. Define all Pydantic models first (`ResearchReport`, `RouterDecision`, `EvaluationResult`).
2. Build and test Stage 2 (the raw research agent loop) on its own — run a question and verify it stores evidence correctly.
3. Build Stage 3 (synthesis) as a standalone function — pass mock evidence, verify it produces a valid `ResearchReport`.
4. Build Stage 4 (evaluator) as a standalone function — pass a mock report, verify scores and feedback.
5. Build Stage 1 (router) — test on three query types.
6. Wire all stages together with your chosen orchestration (LangGraph or PydanticAI).
7. Add CLI argument parsing, file output, and verbose logging.
8. Run on three real research questions and verify end-to-end output.

## Completion checklist

- [ ] Router correctly identifies query type for at least 3 test questions
- [ ] Research agent runs a multi-iteration loop and stores evidence
- [ ] Synthesis produces a valid `ResearchReport` object
- [ ] Evaluator retries synthesis at least once for a deliberately weak report
- [ ] Output saved to both `.md` and `.json` files
- [ ] `--verbose` shows all research agent iterations
- [ ] A search API failure is handled gracefully (no crash)
- [ ] Total cost prints at the end of each run
- [ ] The system works end-to-end on at least three different research questions

## Extension challenges

- **Caching:** store evidence in a local SQLite database keyed by search query. Skip searches for queries you have run in the last 24 hours — cuts API costs dramatically for repeated questions.
- **Report comparison:** run the same question with `--backend langgraph` and `--backend pydantic-ai`, then add a `compare` command that diffs the two reports side by side.
- **Multi-query synthesis:** accept a YAML file of research questions, process them all, and produce a combined report that synthesises findings across all questions.
- **Citation verification:** for each source URL in the final report, send a second HTTP request to verify the URL is alive, and flag any broken citations in the output.
