---
layout: project
lesson_id: "P012"
chapter: 4
chapter_title: "RAG & Vector Search"
project_type: "Chapter Project"
title: "Enterprise knowledge base"
description: "Estimated time: 6–8 hours"
prev: "0033-retrieval-evaluation.html"
prev_title: "Evaluating retrieval quality"
next: "0034-observability.html"
next_title: "Observability for AI systems"
prereqs:
  - "[Lesson 30](0030-ingestion-pipeline.html): Building an ingestion pipeline"
  - "[Lesson 31](0031-similarity-hybrid-search.html): Similarity search and hybrid search"
  - "[Lesson 32](0032-advanced-retrieval.html): Advanced retrieval — contextual retrieval, re-ranking"
  - "[Lesson 33](0033-retrieval-evaluation.html): Evaluating retrieval quality"
  - "[Project P009](P009-ch3-ai-backend.html): Full AI backend service — the Chapter 3 FastAPI stack"
---

## Overview

This is your Chapter 4 portfolio project. You will build a production-shaped enterprise knowledge base: a FastAPI service that can ingest documents from multiple sources, retrieve relevant passages using hybrid search and re-ranking, and answer questions with full source attribution — all backed by a PostgreSQL store for metadata and Chroma for vectors.

You will also build a retrieval evaluation harness and use it to measure your pipeline's quality, apply improvements, and show measurable gains. This is the kind of before/after analysis that AI engineering interviews ask for.

The finished system is something you can demo to a hiring manager, show to a client, and extend into a real product.

## What you are building

An enterprise knowledge base service with three components:

**1. Ingestion API** — `POST /ingest` accepts a URL or file upload; the service downloads, chunks, and indexes it automatically. Results are tracked in PostgreSQL.

**2. Query API** — `POST /query` accepts a natural-language question; the service runs hybrid search with re-ranking and returns a structured answer with cited sources.

**3. Evaluation harness** — `python eval.py` runs a test dataset against the live API and reports Precision@3 and MRR@10. You run this before and after each pipeline improvement.

```
Browser / curl
    │
    ▼
FastAPI (query + ingest endpoints)
    │
    ├── Chroma (vector store)
    │     └── Hybrid search: BM25 + embedding
    │
    ├── PostgreSQL (document registry, ingest logs)
    │
    └── OpenAI API (text-embedding-3-small + gpt-4o)
```

## Project structure

```
knowledge_base/
├── api/
│   ├── main.py               # FastAPI app
│   ├── routes/
│   │   ├── ingest.py         # POST /ingest
│   │   └── query.py          # POST /query
│   └── models.py             # Pydantic request/response schemas
├── pipeline/
│   ├── loaders.py            # PDF, web, Markdown loaders
│   ├── chunk.py              # RecursiveCharacterTextSplitter
│   ├── contextualise.py      # Contextual retrieval (LLM context injection)
│   ├── embed.py              # Batched OpenAI embeddings
│   └── store.py              # Chroma + BM25 index
├── retrieval/
│   ├── hybrid.py             # BM25 + semantic + RRF
│   ├── rerank.py             # Cross-encoder re-ranking
│   └── answer.py             # Build prompt, call GPT-4o
├── db/
│   ├── models.py             # SQLAlchemy ORM models
│   ├── migrations/           # Alembic migrations
│   └── session.py            # Async SQLAlchemy session
├── eval/
│   ├── eval.py               # Run test dataset, compute P@3 and MRR
│   └── test_questions.json   # Your 20-question test dataset
├── docker-compose.yml        # PostgreSQL + app
└── .env
```

## Spec

### PostgreSQL schema

Two tables — track what has been ingested and log each query:

```sql
-- Ingested documents
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    source_url  TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,          -- 'pdf', 'web', 'markdown'
    title       TEXT,
    chunk_count INT,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    checksum    TEXT                    -- MD5 of content for change detection
);

-- Query log
CREATE TABLE query_log (
    id            SERIAL PRIMARY KEY,
    question      TEXT NOT NULL,
    answer        TEXT,
    retrieved_ids TEXT[],               -- Chroma IDs of retrieved chunks
    latency_ms    INT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### Ingestion endpoint

`POST /ingest` accepts either a URL or a file upload:

```python
# Request
class IngestRequest(BaseModel):
    url: str | None = None  # web page or public PDF URL

# Response
class IngestResponse(BaseModel):
    source: str
    chunk_count: int
    status: str   # "created", "updated", "unchanged"
    message: str
```

The endpoint:
1. Checks if the source is already in `documents` and whether it has changed (compare checksum).
2. If unchanged, returns `{"status": "unchanged"}`.
3. Otherwise: load → chunk → contextualise → embed → write to Chroma. Delete old chunks first.
4. Store or update the `documents` record.

### Query endpoint

`POST /query`:

```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5    # number of sources to include in context

class Source(BaseModel):
    text: str
    source: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
```

The endpoint:
1. Run hybrid retrieval (BM25 + embedding) to get top 20 candidates.
2. Re-rank the top 20 with a cross-encoder to get top `top_k`.
3. Build the prompt and call GPT-4o.
4. Log the query to `query_log`.
5. Return the answer and sources.

### Contextual retrieval

Apply contextual retrieval at ingestion time. For each chunk, call Claude Haiku to generate a context sentence (from Lesson 32). Prepend it to the chunk before embedding. This is optional but required to earn the "contextual retrieval" checkbox.

### Evaluation harness

Build a 20-question test dataset for your knowledge base. Include:
- 14 answerable questions with known expected sources
- 3 paraphrase questions (same question in different words than the document)
- 3 unanswerable questions (to test "I don't know" behaviour)

`eval.py` queries the API directly (no internal imports):

```python
import json
import httpx

API_URL = "http://localhost:8000"

def evaluate(test_file: str, k: int = 3) -> None:
    with open(test_file) as f:
        dataset = json.load(f)
    
    passed = 0
    mrr_sum = 0.0
    failures = []
    
    for item in dataset:
        resp = httpx.post(f"{API_URL}/query", json={"question": item["question"], "top_k": 10})
        result = resp.json()
        
        sources = [s["text"].lower() for s in result["sources"]]
        expected = item["expected_contains"].lower()
        
        rank = next(
            (i + 1 for i, src in enumerate(sources) if expected in src),
            None
        )
        
        if rank and rank <= k:
            passed += 1
        
        mrr_sum += (1.0 / rank) if rank else 0.0
        
        if not rank or rank > k:
            failures.append({
                "q": item["question"],
                "expected": expected[:60],
                "got": [s[:60] for s in sources[:3]]
            })
    
    total = len(dataset)
    print(f"Precision@{k}: {passed}/{total} = {passed/total:.2%}")
    print(f"MRR@10:        {mrr_sum/total:.4f}")
    print(f"\nFailures ({len(failures)}):")
    for f in failures:
        print(f"  Q: {f['q'][:60]}")
        print(f"  Expected: {f['expected']}")
        print(f"  Got: {f['got'][0] if f['got'] else 'nothing'}")
        print()

if __name__ == "__main__":
    evaluate("eval/test_questions.json", k=3)
```

## The before/after improvement cycle

This is the core deliverable that demonstrates you understand retrieval engineering:

1. **Baseline**: run `eval.py` with standard embedding search (no BM25, no re-ranking, no contextual retrieval). Record Precision@3 and MRR@10.

2. **Add hybrid search**: enable BM25 + RRF. Re-run `eval.py`. Record the improvement.

3. **Add re-ranking**: enable the cross-encoder re-ranker. Re-run `eval.py`. Record the improvement.

4. **Add contextual retrieval**: re-ingest all documents with context sentences prepended. Re-run `eval.py`. Record the improvement.

Report your results in a `RESULTS.md` file:

```markdown
## Retrieval quality improvements

| Configuration              | Precision@3 | MRR@10 |
|---------------------------|-------------|--------|
| Baseline (embedding only)  | 60%         | 0.52   |
| + Hybrid search (BM25+RRF) | 72%         | 0.67   |
| + Re-ranking               | 80%         | 0.75   |
| + Contextual retrieval     | 88%         | 0.84   |
```

This table, in an interview or on your GitHub, shows that you understand the retrieval stack deeply — not just that you used a vector database.

## Requirements

- [ ] `docker-compose up` starts the app and PostgreSQL
- [ ] `POST /ingest` with a URL ingests the document and returns chunk count
- [ ] `POST /query` returns an answer with cited sources and latency
- [ ] Re-ingesting an unchanged document returns `{"status": "unchanged"}`
- [ ] Re-ingesting a changed document removes old chunks and re-ingests
- [ ] Hybrid search is active (BM25 + embedding + RRF)
- [ ] Cross-encoder re-ranking is applied to the top 20 candidates
- [ ] `eval.py` runs the 20-question test dataset and prints Precision@3 and MRR@10
- [ ] `RESULTS.md` shows at least two configurations compared (e.g., baseline vs hybrid + rerank)
- [ ] All queries are logged to `query_log` in PostgreSQL

## What to put in your portfolio

1. **The README** — describe what the system does, how to run it (`docker-compose up`, ingest a URL, query it), and what the eval results show.

2. **RESULTS.md** — the before/after table. This is the technical proof that you can improve a RAG system systematically.

3. **A short video or GIF** — demo the CLI ingest and query flow. Show the sources cited in the output.

When a hiring manager asks "have you built a RAG system?" you can show them this project, point to the results table, and explain each improvement. That is a much stronger answer than "yes, I've used LangChain."

## Extension challenges

**1. Streaming responses.** Stream the GPT-4o output via Server-Sent Events so the query endpoint returns tokens as they are generated, rather than waiting for the full answer.

**2. RAGAS integration.** Install the `ragas` library and compute additional metrics: faithfulness (does the answer contradict the context?), answer relevancy (does the answer actually address the question?).

**3. Query caching.** Cache query results in Redis for 24 hours. Use semantic similarity (not exact match) to detect near-duplicate queries — if a cached query's embedding is within 0.05 cosine distance of the new query, return the cached result.

**4. Metadata extraction.** During ingestion, use GPT-4o to extract structured metadata from each document: title, author, date, category, summary. Store this in PostgreSQL. Add a `GET /documents` endpoint that lists all ingested documents with their metadata.

**5. Multi-tenant support.** Add a `namespace` parameter to ingest and query endpoints. Each namespace has its own isolated Chroma collection and document registry. Different teams can maintain separate knowledge bases within the same service.
