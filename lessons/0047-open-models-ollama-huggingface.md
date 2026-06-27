---
layout: lesson
lesson_id: "0047"
chapter: 1
chapter_title: "Foundations of AI Engineering"
title: "Open models — running LLMs without an API key"
description: "30–40 min read · Hands-on coding"
prev: "0007-anthropic-api.html"
prev_title: "The Anthropic API — Claude and how it differs"
next: "P003-ch1-multi-model-assistant.html"
next_title: "Chapter Project: Multi-model assistant"
prereqs:
  - "[Lesson 1](0001-ai-dev-environment.html): Python dev environment with uv and .env files"
  - "[Lesson 2](0002-openai-api-basics.html): The LLM API call pattern — you will use the same pattern here, pointed at a different server"
  - "[Lesson 7](0007-anthropic-api.html): The provider-swap abstraction — this lesson adds a third provider (local/open)"
  - "**Ollama installed:** download the free installer from [ollama.com](https://ollama.com) (macOS, Linux, and Windows supported)"
assignment:
  article:
    title: "Open LLM Leaderboard v2"
    url: "https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-2"
    author: "Hugging Face team"
    time: "about 12 minutes"
    why: "Before you can choose a local model, you need to know how open models are compared. Explore the leaderboard to understand what benchmarks measure, what the numbers mean in practice, and how to read rankings when selecting a model for a real project."
  task:
    description: "Add Ollama as a third provider to your multi-provider assistant from Lesson 7."
    steps:
      - "Confirm Ollama is installed and running: `ollama list`"
      - "Pull a model: `ollama pull llama3.2:3b`"
      - "Implement `OllamaClient` following the pattern from this lesson"
      - "Add `--provider ollama` as a valid option in your CLI assistant"
      - "Run the same question against all three providers and compare the responses"
    expected: "`uv run assistant.py --provider ollama \"Explain what an API is in one sentence.\"` returns a response from a local model — no internet API call, no API key used."
    why: "The three-provider setup (OpenAI + Anthropic + Ollama) mirrors what you find in production systems that route tasks by cost and privacy constraints — paid APIs for quality-critical paths, local models for high-volume or sensitive data."
knowledge_check:
  - q: "What makes a model \"open\" as opposed to \"closed\"?"
    a: "An open model's weights are publicly available to download and run yourself. A closed model's weights are kept on the company's servers — you can only access it through their API. Open examples: Llama, Mistral, Qwen. Closed examples: GPT-4o, Claude."
    section: "#what-are-open-models"
    section_title: "What are open models?"
  - q: "What is the difference between a \"base\" model and an \"Instruct\" model?"
    a: "A base model predicts the next token — it continues text but does not reliably follow instructions. An instruction-tuned model (marked \"Instruct\" or \"Chat\") has been further trained to follow instructions and hold a conversation. Always use instruction-tuned models for AI engineering tasks."
    section: "#hugging-face-hub"
    section_title: "The Hugging Face Hub"
  - q: "How do you call an Ollama model from existing OpenAI Python code?"
    a: "Set `base_url=\"http://localhost:11434/v1\"` and `api_key=\"ollama\"` when creating the `OpenAI` client. Change `model=` to the Ollama model name. The response parsing is identical — `response.choices[0].message.content` — because Ollama matches OpenAI's response format exactly."
    section: "#calling-ollama"
    section_title: "Calling Ollama from Python"
  - q: "On the Hugging Face Hub, what is a \"feature extraction\" model used for?"
    a: "A feature extraction model converts text into a vector — a list of numbers that represents the meaning of the text. These are called embeddings models. They are used for semantic search and RAG retrieval: finding text that is *similar in meaning* to a query. You will build RAG systems using embeddings models in Chapter 4."
    section: "#model-taxonomy"
    section_title: "Model categories on the Hub"
  - q: "Name two scenarios where you would choose an open model over the OpenAI API."
    a: "Any two of: (1) data privacy requirements — data cannot leave your servers; (2) high volume where per-token API costs add up significantly; (3) offline or air-gapped environments with no internet access; (4) a specific open model outperforms the general API on a narrow task."
    section: "#when-to-use"
    section_title: "When to use open models"
additional_resources:
  - title: "Ollama model library"
    url: "https://ollama.com/library"
    desc: "Browse all supported models with sizes and descriptions"
  - title: "Ollama on GitHub"
    url: "https://github.com/ollama/ollama"
    desc: "Source code and full REST API reference"
  - title: "Hugging Face Models"
    url: "https://huggingface.co/models"
    desc: "Search 800k+ models by task, language, and licence"
  - title: "Transformers quick tour"
    url: "https://huggingface.co/docs/transformers/quicktour"
    desc: "Official introduction to the `pipeline()` API"
  - title: "Open LLM Leaderboard v2"
    url: "https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-2"
    desc: "Benchmark rankings for open models — the canonical source for comparing open-weight models"
---

## Motivation

Most AI engineering job descriptions list experience with "open-source models," "local LLMs," or "Hugging Face." The reason is practical: many companies cannot send data to OpenAI for compliance reasons — healthcare records, legal documents, and financial data cannot leave their servers. Others run high-volume tasks where per-token costs compound into real money. Open models — Llama, Mistral, Qwen, Gemma — run on your own hardware with no API key, no per-token bill, and no data sent to a third party.

This lesson teaches you where open models live, how to run one on your machine in minutes, and how to make the paid-API-vs-local-model decision with confidence. You will also meet the Hugging Face ecosystem — the central hub for open AI models — and learn enough about model categories to navigate it without getting lost.

{% include prereqs.html %}

## What are "open models"? {#what-are-open-models}

Every LLM you have called so far — GPT-4o, Claude — is a **closed model**. The weights (the billions of numbers that encode the model's learned knowledge) are owned by OpenAI or Anthropic and kept on their servers. When you call the API, your text travels to their data centre, the model runs there, and the response comes back. You never see the weights. You can never run the model yourself.

An **open model** is one whose weights are published publicly. Anyone can download them and run them on their own machine. No request leaves your network. Meta's Llama 3, Mistral AI's Mistral 7B, Google's Gemma, Alibaba's Qwen, and Microsoft's Phi are all open models.

One term distinction worth knowing: "open weights" and "open source" are not the same thing. Llama 3 publishes its weights for free but has a commercial licence with some restrictions. True open-source models publish the weights, the training code, and the data. For AI engineering, the practical question is simply: *can I download and run this myself?* If yes, it is an open model.

Model sizes are measured in parameters. 3B means 3 billion parameters; 7B means 7 billion; 70B means 70 billion. More parameters generally means more capable, but also requires more memory to run. A 3B model fits comfortably on any modern laptop. A 70B model requires a high-end GPU.

## The Hugging Face Hub {#hugging-face-hub}

[Hugging Face](https://huggingface.co) is the central repository for open AI models. Think of it as npm for models, or GitHub for weights. It hosts over 800,000 models, plus datasets and interactive demos. Whenever you need an open model — for text generation, embeddings, classification, or image tasks — the Hub is where you look first.

Every model on the Hub has a **model card**: a page describing what the model does, how it was trained, its licence, known limitations, and how to use it. Before using any model in a project, read its model card. The licence tells you whether commercial use is permitted. The limitations section tells you what the model is bad at.

### Reading a model name

Models on the Hub are identified as `organisation/model-name`. For example: `meta-llama/Llama-3.2-3B-Instruct`. The name encodes useful information:

- **meta-llama** — the organisation that published it
- **Llama-3.2** — the model family and version
- **3B** — 3 billion parameters (the size)
- **Instruct** — *instruction-tuned*: further trained to follow instructions and have a conversation. A model without "Instruct" or "Chat" in its name is a *base model* — it predicts the next token but does not reliably follow instructions.

Always use instruction-tuned models for AI engineering work. Base models are for researchers who are fine-tuning or evaluating raw capabilities.

## Model categories on the Hub {#model-taxonomy}

The Hub organises its 800,000+ models by *task*. As an AI engineer, you will encounter these categories regularly:

| Category | What it does | Example use |
|---|---|---|
| **Text generation** | Takes text in, returns text — the familiar LLM pattern | Chat, summarisation, code generation |
| **Feature extraction** | Converts text into a list of numbers — a *vector* — that represents its meaning. Also called an *embeddings model*. | Semantic search, RAG retrieval (Chapter 4) |
| **Text classification** | Assigns a label to text from a fixed set of categories | Sentiment analysis, spam detection, moderation |
| **Text2text generation** | Encoder-decoder models that transform one text into another | Translation, summarisation (T5-style models) |
| **Image generation** | Takes a text prompt, returns an image | Content creation (Stable Diffusion, Flux) |
| **Multimodal / any-to-any** | Handles multiple input and output types: text, images, and sometimes audio | Vision-language models that describe images or answer questions about photos |

For most AI engineering work you will use **text generation** models (running LLMs locally) and **feature extraction** models (generating embeddings for search and RAG, covered in depth in Chapter 4). The other categories are important to know exist — and the Hub is where you go when you need them.

## Ollama — running models locally {#ollama}

Running a raw Hugging Face model requires handling quantisation formats, GPU memory management, batching, and model serving — non-trivial engineering. Ollama packages all of that into a single tool. It downloads a model, optimises it for your hardware (GPU if available, CPU otherwise), and runs it as a local server.

Two terminal commands are all you need:

```bash
# Download a 3B parameter model (~2 GB — runs on any laptop)
ollama pull llama3.2:3b

# Send a quick prompt to verify it works
ollama run llama3.2:3b "What is the capital of Japan?"
```

Ollama runs as a background service on port 11434. On macOS it starts automatically as a menu bar app. On Linux, start it with `ollama serve` if it is not running. To see which models you have downloaded:

```bash
ollama list
```

### Choosing a model

Ollama has its own curated library at [ollama.com/library](https://ollama.com/library) — pre-packaged, ready-to-run versions of popular open models. A quick guide for development:

| Model | Size | RAM needed | Best for |
|---|---|---|---|
| `llama3.2:3b` | 3B params | ~2 GB | Fast testing on any laptop; verifying integrations |
| `llama3.1:8b` | 8B params | ~8 GB | Better quality; default choice for most dev work |
| `mistral:7b` | 7B params | ~5 GB | Fast; popular for coding tasks |
| `qwen2.5:7b` | 7B params | ~5 GB | Strong multilingual support; good for non-English content |

Start with `llama3.2:3b`. It downloads in a few minutes, runs on any laptop without a GPU, and is fast enough for development. Once your integration works, try a larger model to evaluate quality.

## Calling Ollama from Python {#calling-ollama}

Ollama exposes its API at `http://localhost:11434/v1` using the exact same format as OpenAI's API. This means you can call a local Ollama model using the `openai` Python library — just change two values.

```python
from openai import OpenAI

# Point the OpenAI client at your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the library, but ignored by Ollama
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is the capital of Japan?"}
    ]
)

print(response.choices[0].message.content)  # Tokyo
```

The response parsing — `response.choices[0].message.content` — is identical because Ollama deliberately matches OpenAI's response format. Your existing OpenAI code works with local models with minimal changes.

Here is how to add Ollama to the `LLMClient` abstraction from [Lesson 7](0007-anthropic-api.html), making it a proper third provider:

```python
from openai import OpenAI

class OllamaClient:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content

# Usage — identical to OpenAIClient and AnthropicClient from Lesson 7
def get_client(provider: str):
    if provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    elif provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown provider: {provider}")

client = get_client("ollama")
result = client.complete(
    system="You are a helpful assistant.",
    user="What is the capital of Japan?",
)
print(result)  # Tokyo — answered locally, no internet call
```

<div class="callout info">
<strong>Make sure Ollama is running</strong> before calling this code. If you get a connection error (<code>ConnectionRefusedError</code>), the server is not started. On macOS: look for the Ollama icon in your menu bar. On Linux: run <code>ollama serve</code> in a terminal.
</div>

## The Hugging Face Transformers library {#transformers-library}

`transformers` is the Python library for loading and running Hugging Face models directly — no Ollama in between. It is the lower-level option: more control, more complexity. For text generation, Ollama is almost always simpler and faster to set up. Transformers becomes the right tool when:

- You need a specific **embeddings model** that Ollama does not serve (very common in RAG pipelines)
- You need a specialised **classifier** (sentiment, toxicity, named entity recognition)
- You are doing fine-tuning or model evaluation research

The easiest entry point is the `pipeline()` function, which wraps a model with sensible defaults for a given task:

```python
from transformers import pipeline

# Feature extraction — converts text to an embedding vector
# (You will use this pattern heavily in Chapter 4 — RAG)
embedder = pipeline(
    "feature-extraction",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

text = "What is the capital of Japan?"
output = embedder(text, return_tensors=False)
# output is a nested list: [batch][token][dimension]
vector = output[0][0]  # the CLS token vector
print(f"Embedding has {len(vector)} dimensions")  # 384
```

```python
# Text classification — sentiment analysis with a pre-trained model
classifier = pipeline("sentiment-analysis")
result = classifier("This product is absolutely fantastic!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

The first run of either snippet downloads the model automatically from the Hub (~90 MB for MiniLM). Subsequent runs load from a local cache at `~/.cache/huggingface/`.

Install with: `uv add transformers torch`. PyTorch is required as the computation backend and is a large download (~800 MB). If you are only generating embeddings and not running full text generation, you can often use the lighter-weight [Sentence Transformers](https://www.sbert.net) library (`uv add sentence-transformers`), which this curriculum uses in Chapter 4.

## The Hugging Face Inference API {#inference-api}

Downloading a 7B model takes gigabytes of space and minutes of time. If you want to test a model on the Hub before committing to running it locally, Hugging Face hosts many models behind a free cloud API — the **Inference API**.

```python
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN"),
)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is the capital of Japan?"}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Add it to your `.env` file as `HF_TOKEN`. Install with: `uv add huggingface-hub`.

The free tier has rate limits and is not suitable for production use. In production, companies typically run one of:

- **Ollama** — for a single machine or a small team
- **vLLM** — a production-grade inference server for serving open models at scale; covered in Chapter 6
- **Cloud AI providers** like AWS Bedrock or Azure OpenAI, which host open models behind a managed API; also Chapter 6

Use the Inference API for exploration — to answer "does this model handle my task well?" before building infrastructure around it.

## When to use open models vs paid APIs {#when-to-use}

You will face this decision on every AI engineering project. Neither option is universally better. The right choice depends on the specific constraints:

| Use a paid API (OpenAI / Anthropic) | Use an open model |
|---|---|
| You need the highest reasoning quality for complex tasks | Data cannot leave your servers (healthcare, legal, finance) |
| You want to ship quickly with no infrastructure to manage | High volume — per-token costs add up at scale |
| Your data is not sensitive (public content, general knowledge) | Offline or air-gapped environments where internet access is restricted |
| You need a very long context window or strong multimodal capability | A specific open model outperforms the general API on your narrow task |

The practical default for most projects: **start with a paid API.** It is faster to get going, the models are stronger, and there is no infrastructure to manage. Switch to open models when a concrete constraint — privacy, cost, or offline access — pushes you there.

<div class="callout info">
<strong>The Staircase applies here too.</strong> Open models are not a starting point — they are an optimisation you reach after validating your system on a paid API. Debug with GPT-4o or Claude, then migrate to a local model once you know exactly what the task requires.
</div>
