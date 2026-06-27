---
layout: lesson
lesson_id: "0027"
chapter: 4
chapter_title: "RAG & Vector Search"
title: "Embeddings — turning text into numbers"
description: "30–40 min read · Hands-on coding"
prev: "0026-what-is-rag.html"
prev_title: "What is RAG and why LLMs need it"
next: "0028-chunking.html"
next_title: "Chunking strategies — splitting documents for retrieval"
prereqs:
  - "[Lesson 26](0026-what-is-rag.html): What is RAG — you need to know why we need to measure document similarity"
  - "[Lesson 2](0002-openai-api-basics.html): OpenAI API basics — the embedding API follows the same pattern as the completions API"
assignment:
  article:
    title: "The Beginner's Guide to Text Embeddings & Techniques"
    url: "https://www.deepset.ai/blog/the-beginners-guide-to-text-embeddings"
    author: "deepset"
    time: "15 min"
    why: "deepset builds Haystack, one of the most widely used open-source RAG frameworks. This guide goes deeper than most — it explains embedding techniques (sparse vs dense, bi-encoders, cross-encoders) in plain English, which is exactly the vocabulary you need to understand the advanced retrieval lesson coming in Lesson 32."
  task:
    description: "Compute the embedding of three sentences and rank them by similarity."
    steps:
      - "Create `embeddings_demo.py`"
      - "Embed these three sentences using `text-embedding-3-small`: (1) 'The cat sat on the mat', (2) 'A kitten rested on the rug', (3) 'The stock market fell sharply today'"
      - "Implement `cosine_similarity(a, b)` using `numpy` (formula in this lesson)"
      - "Print the similarity score between sentences 1 and 2, and between sentences 1 and 3"
      - "Confirm that 1 and 2 score higher than 1 and 3"
    expected: "Similarity(1,2) > 0.8, Similarity(1,3) < 0.8 — the semantically related pair scores higher"
    why: "Cosine similarity is the fundamental operation that powers every RAG retrieval step. Once you have computed it by hand, the vector database abstractions in Lesson 29 will feel obvious rather than magical."
knowledge_check:
  - q: "What is a text embedding, in plain English?"
    a: "A text embedding is a list of numbers that represents the meaning of a piece of text. The numbers are chosen so that texts with similar meanings produce similar lists of numbers. You can think of an embedding as coordinates in a semantic space — texts about the same topic end up near each other."
    section: "#what-an-embedding-is"
    section_title: "What an embedding is"
  - q: "What does the cosine similarity score tell you, and what is its range?"
    a: "Cosine similarity measures how closely two embedding vectors point in the same direction. A score of 1.0 means the texts are semantically identical (or very close). A score of 0 means they are unrelated. Negative scores are rare for text embeddings but would indicate opposite meanings. In practice, relevant documents in RAG typically score 0.7–0.95."
    section: "#cosine-similarity"
    section_title: "Cosine similarity — how closeness is measured"
  - q: "What is the OpenAI model name for the standard text embedding model, and what does its output look like?"
    a: "`text-embedding-3-small` (or the larger `text-embedding-3-large`). The output is a Python list of floating-point numbers — 1,536 numbers for `text-embedding-3-small`. Each number is a coordinate in the model's semantic space. The list as a whole is the vector."
    section: "#the-embedding-api"
    section_title: "The OpenAI embedding API"
  - q: "Why do we embed the user's *question* when doing retrieval, not just the documents?"
    a: "To find documents relevant to a query you need to compare the query to the documents in the same semantic space. Embedding both the query and the documents lets you compute cosine similarity between them and rank documents by how closely they match the question's meaning — not just its exact words."
    section: "#how-embeddings-power-retrieval"
    section_title: "How embeddings power retrieval"
additional_resources:
  - title: "OpenAI embeddings guide"
    url: "https://platform.openai.com/docs/guides/embeddings"
    desc: "Official reference for models, dimensions, pricing, and the full API surface"
  - title: "Sentence Transformers"
    url: "https://www.sbert.net/"
    desc: "Open-source Python library for generating embeddings locally without an API key — useful when you cannot send data to a third-party API"
  - title: "MTEB leaderboard"
    url: "https://huggingface.co/spaces/mteb/leaderboard"
    desc: "Benchmark comparing embedding models across retrieval, classification, and clustering tasks — use this when choosing a model for production"
---

## Motivation

In the previous lesson you saw RAG at a high level: retrieve relevant documents, add them to the prompt, let the LLM answer. But how does retrieval actually work? The user asks "What is the refund window for enterprise customers?" — how does the system find the right paragraph in a thousand-page policy document?

You cannot search by exact keyword match. The document might say "money-back period" instead of "refund window". The user might ask in a different language. The document might use different terminology entirely.

What you need is a way to search by *meaning*, not by words. Embeddings make that possible. They are the foundational mechanism behind all vector search, and therefore behind all of RAG.

{% include prereqs.html %}

## What an embedding is

An **embedding** is a list of numbers that represents the meaning of a piece of text.

The key property is this: texts with similar meanings produce lists of numbers that are mathematically similar. Texts about unrelated topics produce lists that are mathematically far apart.

Here is a simple analogy. Imagine plotting cities on a map. London and Paris are close because they share geography, culture, and history. London and Tokyo are far apart. The map coordinates are not the cities themselves — they are a *representation* that encodes how similar the cities are to each other.

An embedding works the same way, except the "map" has 1,536 dimensions instead of 2, and the "cities" are pieces of text. A sentence about dogs will be close to a sentence about puppies. A sentence about dogs will be far from a sentence about interest rates.

This is exactly what you need for retrieval: given a user's question, find the documents whose embeddings are closest to the question's embedding. Those are the documents most likely to contain the answer.

## The OpenAI embedding API

Generating an embedding is a single API call:

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What is the refund policy for enterprise customers?"
)

embedding = response.data[0].embedding
print(type(embedding))   # <class 'list'>
print(len(embedding))    # 1536
print(embedding[:5])     # [-0.021, 0.008, 0.034, -0.007, 0.015] — five of 1536 numbers
```

The call takes a piece of text and returns a list of 1,536 floating-point numbers. That list is the embedding.

You can embed multiple texts in one call to reduce API latency:

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "What is the refund policy for enterprise customers?",
        "Enterprise accounts are eligible for refunds within 180 days.",
        "Our standard return window is 30 days for all customers.",
        "The stock market fell sharply on Thursday."
    ]
)

embeddings = [item.embedding for item in response.data]
print(len(embeddings))      # 4 — one embedding per input
print(len(embeddings[0]))   # 1536
```

All four embeddings are now in the same semantic space. The first two (question and enterprise refund policy) will be close to each other. The last one (stock market) will be far from all the others.

<div class="callout info">
<strong>What model should I use?</strong> For most applications, <code>text-embedding-3-small</code> is the right default — it is fast, cheap, and accurate. <code>text-embedding-3-large</code> produces 3,072-dimensional embeddings and performs better on hard retrieval tasks, at roughly 3× the cost. Start with small; upgrade only if retrieval quality is measurably poor.
</div>

## Cosine similarity — how closeness is measured

Once you have two embeddings you need a way to measure how similar they are. The standard method is **cosine similarity**.

Cosine similarity measures the angle between two vectors. If two vectors point in exactly the same direction, the angle between them is 0° and the similarity is 1.0. If they point in perpendicular directions (completely unrelated), the similarity is 0. Negative values (opposite directions) rarely occur in practice for text embeddings.

You do not need to understand the trigonometry to use this. The formula is:

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

Where `A · B` is the dot product (multiply each pair of corresponding numbers and sum the results), and `|A|` and `|B|` are the lengths of the vectors.

In Python with NumPy:

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

Now put it all together:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Three texts: two about cats, one about finance
emb_cat1 = embed("The cat sat on the mat")
emb_cat2 = embed("A kitten was resting on the rug")
emb_stocks = embed("The stock market fell sharply on Thursday")

print(f"Cat sentences:  {cosine_similarity(emb_cat1, emb_cat2):.4f}")
# → 0.8912 — high similarity
print(f"Cat vs stocks:  {cosine_similarity(emb_cat1, emb_stocks):.4f}")
# → 0.0891 — low similarity
```

The numbers will vary slightly each run (floating-point precision), but the cat sentences will always score much higher against each other than against the stock market sentence. That gap is what retrieval exploits.

## How embeddings power retrieval

With embeddings and cosine similarity you can implement a simple retrieval system from scratch:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# A tiny "document store" — in real systems this lives in a vector database
documents = [
    "Enterprise accounts are eligible for refunds within 180 days of purchase.",
    "Standard accounts have a 30-day return window.",
    "All refunds are processed within 5–7 business days.",
    "Products must be returned in original packaging.",
    "Shipping costs for returns are covered for orders over $100.",
]

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Embed all documents at ingestion time
doc_embeddings = [embed(doc) for doc in documents]

# At query time: embed the question and find the closest documents
question = "How long do enterprise customers have to request a refund?"
question_embedding = embed(question)

# Score each document against the question
scores = [
    (cosine_similarity(question_embedding, doc_emb), doc)
    for doc_emb, doc in zip(doc_embeddings, documents)
]

# Sort by score, highest first
scores.sort(key=lambda x: x[0], reverse=True)

print("Top results:")
for score, doc in scores[:3]:
    print(f"  [{score:.4f}] {doc}")
```

Output:
```text
Top results:
  [0.8734] Enterprise accounts are eligible for refunds within 180 days of purchase.
  [0.7102] Standard accounts have a 30-day return window.
  [0.6431] All refunds are processed within 5–7 business days.
```

The enterprise refund policy ranks first — even though the question uses "request a refund" and the document uses "eligible for refunds". Keyword search would have struggled with this; embedding search handles it naturally because the underlying meaning is the same.

This is the complete retrieval mechanism. In practice, the "document store" is replaced by a vector database (Lesson 29) that can handle millions of documents and run similarity search in milliseconds, but the mathematical operation is identical to what you just implemented.

<div class="callout good">
<strong>What you just built</strong> is a working retrieval system. It searches by meaning rather than by keywords, handles paraphrasing, and ranks results by relevance. The only thing missing is scale — which is what vector databases provide.
</div>

## Embedding costs and performance

The OpenAI embedding API charges per token of input text. Costs are low but accumulate when indexing large document collections:

| Model | Dimensions | Cost (per 1M tokens) | Best for |
|-------|-----------|---------------------|----------|
| `text-embedding-3-small` | 1,536 | ~$0.02 | Most applications |
| `text-embedding-3-large` | 3,072 | ~$0.13 | High-accuracy requirements |
| `ada-002` (legacy) | 1,536 | ~$0.10 | Only for backward compatibility |

A "document" of average length (300 words ≈ 400 tokens) costs roughly $0.000008 to embed with `text-embedding-3-small`. Indexing 100,000 documents costs about $0.80. Ingestion is done once; queries are cheap.

<div class="callout info">
<strong>One-time cost:</strong> you embed documents once at ingestion time and store the vectors. You only re-embed when documents change. Query embeddings are cheap — a typical user question is 10–20 tokens.
</div>

## What embeddings cannot do

Understanding the limits of embeddings is as important as understanding what they can do.

**Embeddings capture general semantic similarity, not domain-specific meaning.** The word "python" in "Python is a programming language" and "The python snake is cold-blooded" may produce embeddings that are closer together than you expect, because the model has seen both uses in training. In specialised domains (legal, medical, scientific), generic embedding models sometimes underperform domain-specific ones.

**Embeddings do not capture exact matches well.** If a user searches for "invoice #INV-20260312", embedding similarity will not reliably surface that specific invoice. For exact identifiers, product codes, or serial numbers, keyword search (covered in Lesson 31) is more appropriate. This is why production RAG systems often combine embedding search with keyword search — a technique called hybrid search.

**Embedding quality depends on the model and the language.** `text-embedding-3-small` performs well on English text. For other languages or for highly specialised domains, you may need a different model. The MTEB leaderboard (linked in Additional Resources) benchmarks models across tasks and languages.

**Embeddings are frozen at generation time.** Once you embed a document, the vector captures the document's meaning as the model understood it when you called the API. If the document changes, you must re-embed it.

## What comes next

You now understand what embeddings are, how to generate them, and how cosine similarity turns them into a retrieval mechanism. The next question is: what exactly do you embed?

A 50-page PDF cannot be embedded as a single piece of text — it would be too long for the embedding model and the resulting vector would be too coarse to retrieve specific passages. You need to split the document into meaningful fragments first. That is **chunking**, the topic of the next lesson.
