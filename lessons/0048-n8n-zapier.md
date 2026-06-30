---
layout: lesson
lesson_id: "0048"
chapter: 2
chapter_title: "AI System Design"
title: "Visual AI workflows — n8n and Zapier"
description: "30–40 min read · Hands-on setup"
prev: "0051-crewai.html"
prev_title: "CrewAI — role-based multi-agent coordination"
next: "P006-ch2-research-assistant.html"
next_title: "Chapter Project: Automated research assistant"
prereqs:
  - "[Lesson 14](0014-agentic-loops.html): agentic loops — you understand what a workflow is and why you'd chain LLM calls"
  - "[Lessons 17, 50, and 51](0017-langchain.html): LangChain, LangGraph, and CrewAI — you know where visual tools fit on the orchestration spectrum"
  - "Docker installed — used to run n8n locally (see [Lesson 1](0001-ai-dev-environment.html))"
  - "An OpenAI API key in your environment"
assignment:
  article:
    title: "n8n vs Zapier: Which workflow automation tool should you choose?"
    url: "https://blog.n8n.io/n8n-vs-zapier/"
    author: "n8n team"
    time: "12 minutes"
    why: "Written by the n8n team but genuinely useful as a practitioner comparison — it covers the architectural differences (self-hosted vs SaaS, code nodes vs no-code) and the scenarios where each tool wins. Reading it will tell you more about what kind of automation work each tool is actually suited to than any abstract explanation."
  task:
    description: "Run n8n locally and build a two-node AI classifier workflow."
    steps:
      - "Start n8n with Docker: `docker run -it --rm -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n`"
      - "Open http://localhost:5678 and create a new workflow"
      - "Add a **Webhook** node — set it to POST, path `classify`, response mode `Last Node`"
      - "Add an **OpenAI** node — Model: `gpt-4o-mini`, Operation: `Message a Model`, System prompt: `Classify this customer message as exactly one of: complaint, question, praise. Reply with only the word.`, User: `{{ $json.body.message }}`"
      - "Connect Webhook → OpenAI. Activate the workflow."
      - "Test: `curl -X POST http://localhost:5678/webhook/classify -H 'Content-Type: application/json' -d '{\"message\": \"Your shipping is always late and the support team never replies\"}'`"
    expected: "The HTTP response contains the single word `complaint` (or similar classification)."
    why: "You have now run an LLM inside a visual automation, no Python required. This is the pattern ops teams reach for when they need AI-powered routing or classification without waiting for engineering bandwidth."
knowledge_check:
  - q: "What is the key architectural difference between n8n and Zapier?"
    a: "n8n is open-source and self-hosted — you run it on your own infrastructure (Docker, a VM, etc.) so your data never leaves your control. Zapier is a cloud SaaS — all data goes through Zapier's servers. n8n also has a Code node for custom logic; Zapier is no-code only."
    section: "#n8n"
    section_title: "n8n — developer-friendly, self-hosted"
  - q: "What does n8n's Code node let you do that pure visual tools cannot?"
    a: "Write custom JavaScript or Python logic directly inside the workflow — loops, conditional transformations, custom parsing, arithmetic — anything that cannot be expressed by clicking together pre-built nodes. This is what makes n8n developer-friendly: you are never stuck at the edge of what the GUI supports."
    section: "#code-node"
    section_title: "The Code node"
  - q: "Name two scenarios where a visual tool like n8n or Zapier is a better choice than writing a Python script."
    a: "Any two of: (1) the automation will be owned and maintained by a non-developer (marketer, ops person); (2) you're prototyping quickly to validate an idea before investing engineering time; (3) you need many SaaS integrations (Slack, HubSpot, Google Sheets) that already exist as pre-built nodes; (4) the business wants to change the workflow frequently without deploying code."
    section: "#when-visual"
    section_title: "When to use visual tools vs Python"
  - q: "Where do n8n and Zapier sit on the orchestration spectrum compared to LangGraph?"
    a: "At the low-control end — more accessible, less customisable. LangGraph gives you full control over every state transition in Python; n8n and Zapier give you a drag-and-drop canvas that trades flexibility for speed. The full spectrum runs: Raw API → LangGraph → PydanticAI → CrewAI → n8n → Zapier."
    section: "#spectrum"
    section_title: "Where visual tools fit in the orchestration spectrum"
additional_resources:
  - title: "n8n documentation — AI nodes"
    url: "https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain/"
    desc: "Reference for n8n's built-in AI and LangChain nodes, including the OpenAI, Anthropic, and memory nodes"
  - title: "n8n self-hosting guide"
    url: "https://docs.n8n.io/hosting/"
    desc: "Docker, Docker Compose, and cloud VM deployment options for n8n"
  - title: "Zapier AI documentation"
    url: "https://help.zapier.com/hc/en-us/categories/22022023-AI"
    desc: "Zapier's official docs for AI steps — ChatGPT, Claude, and the Formatter AI step"
---

## Motivation

Not every AI workflow needs Python. Marketing ops teams run Slack-to-CRM automations, support engineers route tickets automatically, and finance teams generate weekly summary reports — all without writing a line of code. Visual workflow tools like n8n and Zapier sit at the low-code end of the orchestration spectrum you explored in Lesson 15. Understanding when to reach for them — and when to write Python instead — is a practical AI engineering skill. You will encounter workflows built in these tools in the wild, and you may be asked to extend or debug them.

{% include prereqs.html %}

## What visual workflow tools are {#visual-tools}

A visual workflow tool lets you build automations by connecting nodes on a canvas. Each node does one thing: call an HTTP endpoint, transform data, send a Slack message, query a database, or wait for a webhook trigger. You connect nodes by drawing arrows between them.

The AI-relevant nodes are the key addition of the last two years: every major visual tool now ships direct integrations with OpenAI, Anthropic, Google Gemini, and others. You can build a pipeline like:

```text
New email arrives → Extract action items (GPT-4o) → Create Notion tasks → Reply to sender
```

without writing Python — and then hand that automation to a non-technical teammate who can read and adjust it in the GUI.

## n8n — developer-friendly, self-hosted {#n8n}

**n8n** (pronounced "n-eight-n") is open-source workflow automation software you run yourself — on your laptop, a Docker container, or a cloud VM. The self-hosted version is free; they charge for their cloud-hosted version.

### Why developers choose n8n

- **Self-hosted:** your data never leaves your infrastructure. This matters for AI systems handling sensitive documents, customer data, or anything under a compliance policy.
- **Code node:** when the visual nodes are not enough, drop into a JavaScript or Python function inline. This is what separates n8n from purely no-code tools — you are never blocked by the limits of the GUI.
- **400+ integrations:** Slack, HubSpot, Google Sheets, Postgres, HTTP webhooks, plus native OpenAI and Anthropic nodes.

### Running n8n with Docker

```bash
docker run -it --rm \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

Open `http://localhost:5678`. You will see the n8n editor — a blank canvas ready for nodes. The volume mount (`~/.n8n`) persists your workflows between runs.

### Building an AI classifier workflow

This example wires a webhook trigger into an OpenAI call that classifies customer messages:

**Step 1 — Webhook node**

Add a **Webhook** node. Set:
- HTTP method: `POST`
- Path: `classify`
- Response mode: `Last Node` (return whatever the last node produces)

This creates an endpoint at `http://localhost:5678/webhook/classify`.

**Step 2 — OpenAI node**

Add an **OpenAI** node. Set:
- Model: `gpt-4o-mini`
- Operation: `Message a Model`
- System prompt: `Classify this customer message as exactly one of: complaint, question, praise. Reply with only the single word.`
- User message: `{% raw %}{{ $json.body.message }}{% endraw %}` (n8n's expression syntax pulls `message` from the POST body)

Connect **Webhook → OpenAI**.

**Step 3 — Activate and test**

Activate the workflow (toggle in the top-right). Then test it:

```bash
curl -X POST http://localhost:5678/webhook/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "Your shipping is always late and support never replies"}'
```

Response:
```
complaint
```

An LLM-powered classifier running as an HTTP endpoint, with no Python. This is the pattern support engineering teams use to route tickets automatically.

### The Code node {#code-node}

When you need logic that goes beyond what the visual nodes can express — a loop, a custom transformation, a calculation — the **Code node** runs JavaScript (or Python in newer n8n versions) inline:

```javascript
// Code node: filter LLM results below a quality threshold
const items = $input.all();

return items.filter(item => {
  const score = parseInt(item.json.score, 10);
  return score >= 7;
});
```

The Code node reads from `$input` (the previous node's output) and returns an array of items for the next node. This is the escape hatch that makes n8n powerful for developers: you can drop into code whenever the visual layer is not expressive enough, then pop back out into visual nodes.

## Zapier — SaaS, no-code {#zapier}

**Zapier** is a cloud-hosted, no-code workflow tool. It is older, more polished, and designed for non-technical users — marketers, operations people, customer success teams. Zapier's pricing is per task (each node execution costs a task); your data goes through Zapier's servers.

### When Zapier wins

- **Your teammates are non-developers.** A Zap is easy to read and edit in the Zapier UI without any programming knowledge.
- **You need SaaS breadth quickly.** Zapier has 5,000+ pre-built integrations: Salesforce, HubSpot, Mailchimp, Airtable, DocuSign, and hundreds more.
- **You do not want to manage infrastructure.** Zapier handles hosting, scaling, and uptime.

### What a Zap looks like

A "Zap" is Zapier's name for a workflow. A simple AI-powered Zap might be:

1. **Trigger:** New row added to a Google Sheet (a list of customer feedback)
2. **Action — ChatGPT:** Feed the row's feedback text to GPT-4o with the prompt: `Summarise this feedback and identify the main issue in one sentence.`
3. **Action — Gmail:** Create a draft email to the customer success team with the AI summary

Every step is configured in Zapier's web UI — no code, no deployment. A non-technical person on your team can read this Zap, understand what it does, and update the prompt or the email template themselves.

<div class="callout info">
<strong>n8n vs Zapier on AI:</strong> Both support OpenAI and Anthropic out of the box. n8n gives you more control (model parameters, custom headers, the Code node). Zapier is simpler to configure and faster to share with non-technical teammates.
</div>

## When to use visual tools vs Python {#when-visual}

The decision comes down to one question: **who owns this automation after it is built?**

| Scenario | Recommendation |
|---|---|
| A non-developer will maintain the workflow | Visual tool — they can read and edit the canvas |
| Rapid prototyping to validate an idea | Visual tool — build in 20 min vs 2 hours of Python |
| Sensitive data (documents, PII) | n8n self-hosted — data stays on your infra |
| Many SaaS integrations (Slack, HubSpot, Sheets) | Zapier or n8n — hundreds of pre-built connectors |
| Complex logic: loops, custom parsing, error handling | Python — code is easier to reason about than visual spaghetti |
| High-volume or latency-critical processing | Python — visual tools add HTTP round-trip overhead |
| Part of a larger production codebase | Python — version control, tests, and CI work better |
| Team has no Python developers | Zapier — fastest path to production for ops teams |

<div class="callout warn">
<strong>Visual workflows are hard to version-control.</strong> n8n exports workflows as JSON (which you can commit to git), but the natural editing environment is the GUI. Large n8n workflows become hard to diff, review, and test. When a workflow grows beyond 15–20 nodes or starts encoding meaningful business logic, migrating it to Python is usually the right call.
</div>

## Where visual tools fit in the orchestration spectrum {#spectrum}

In Lesson 15, you saw LangGraph, LangChain, and CrewAI — Python frameworks that give you fine-grained control over multi-agent workflows. n8n and Zapier sit at the opposite end of the same spectrum:

```text
Most control ←─────────────────────────────────────────────→ Least control
Raw API  ·  LangGraph  ·  PydanticAI  ·  CrewAI  ·  n8n  ·  Zapier
```

Each step to the right: less Python, more visual, easier for non-developers, harder to customise. There is no single right answer — the right tool depends on who is building, who will maintain it, and how much custom logic the automation needs.

An AI engineer is expected to understand this full spectrum and make the right call for each situation. Knowing n8n and Zapier exist — and when to recommend them over a Python service — is as important as knowing when to use LangGraph.
