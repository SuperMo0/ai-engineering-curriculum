---
layout: project
lesson_id: "P011"
chapter: 4
chapter_title: "RAG & Vector Search"
project_type: "Project"
title: "Multi-source RAG system"
description: "Estimated time: 3–4 hours"
prev: "0032-advanced-retrieval.html"
prev_title: "Advanced retrieval — contextual retrieval, query expansion, re-ranking"
next: "0033-retrieval-evaluation.html"
next_title: "Evaluating retrieval quality"
prereqs:
  - "[Lesson 30](0030-ingestion-pipeline.html): Building an ingestion pipeline — you need the loader and chunker patterns"
  - "[Lesson 31](0031-similarity-hybrid-search.html): Similarity search and hybrid search — this project uses hybrid search"
  - "[Lesson 32](0032-advanced-retrieval.html): Advanced retrieval — source attribution is a direct application of this lesson"
---

## Overview

Real-world knowledge bases are not a pile of PDFs. They combine product documentation (PDFs), web pages (HTML), structured records (CSV), and internal wikis (Markdown). A production RAG system must ingest all of them and retrieve across them uniformly.

In this project you will build a multi-source RAG system that:
- Ingests PDFs, web pages, and CSV files into a single vector store
- Uses hybrid search (BM25 + embedding) for retrieval
- Displays which specific source each retrieved passage came from

This is the closest project so far to what you would build for a real client.

## What you are building

```bash
# Ingest multiple source types
python rag.py ingest docs/manual.pdf
python rag.py ingest https://docs.example.com/api-reference
python rag.py ingest data/products.csv

# Query across all sources
python rag.py ask "What are the API rate limits?"

# Output:
# Answer: The API is rate-limited to 100 requests per minute per API key. ...
#
# Sources used:
#   [0.94] docs/manual.pdf (chunk 12, page 4) — "The API enforces a rate limit of..."
#   [0.87] https://docs.example.com/api-reference (section: Rate Limits) — "100 requests/min..."
```

## Spec

### Project structure

```
multi_rag/
├── rag.py              # CLI: ingest + ask commands
├── loaders/
│   ├── __init__.py
│   ├── pdf.py          # PyMuPDF PDF loader
│   ├── web.py          # httpx + BeautifulSoup web page loader
│   └── csv_loader.py   # CSV row-to-text loader
├── pipeline/
│   ├── __init__.py
│   ├── chunk.py        # RecursiveCharacterTextSplitter wrapper
│   ├── embed.py        # Batched OpenAI embeddings
│   └── store.py        # Chroma read/write
├── retrieval/
│   ├── __init__.py
│   ├── bm25.py         # BM25 index (in-memory, rebuilt each query)
│   ├── hybrid.py       # Hybrid search + RRF
│   └── answer.py       # Build prompt, call GPT-4o, format output
└── .env
```

### Dependencies

```bash
uv add openai chromadb pymupdf langchain-text-splitters \
        beautifulsoup4 httpx rank-bm25 python-dotenv
```

### Loaders

**PDF loader** (`loaders/pdf.py`): same as Lesson 30. Extract text page by page. Store page number in metadata.

**Web loader** (`loaders/web.py`): fetch the page with `httpx`, parse with `BeautifulSoup`. Extract only the main content — strip `<nav>`, `<footer>`, `<script>`, `<style>` tags. Store the URL and (if present) the `<h1>` heading in metadata.

```python
import httpx
from bs4 import BeautifulSoup

def load_web(url: str) -> tuple[str, dict]:
    """Fetch a web page and return (text, metadata)."""
    resp = httpx.get(url, follow_redirects=True, timeout=10)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Remove noise elements
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    
    title = soup.find("h1")
    text = soup.get_text(separator="\n", strip=True)
    
    return text, {
        "source": url,
        "source_type": "web",
        "title": title.get_text() if title else url,
    }
```

**CSV loader** (`loaders/csv_loader.py`): read each row and convert it to a prose sentence. This lets the embedding model understand structured data as natural language.

```python
import csv
from pathlib import Path

def load_csv(file_path: str) -> list[tuple[str, dict]]:
    """Convert each CSV row to a text passage with metadata."""
    results = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Convert row to prose: "Product: Widget A. Price: $29.99. Stock: 142 units."
            prose = ". ".join(f"{k}: {v}" for k, v in row.items() if v.strip())
            results.append((
                prose,
                {
                    "source": str(file_path),
                    "source_type": "csv",
                    "row_index": i,
                    "filename": Path(file_path).name,
                }
            ))
    return results
```

### Chunking

For PDFs and web pages: `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)`.

For CSV rows: do not chunk further — each row is already a single fact. Store rows directly as documents.

### Hybrid search

Implement hybrid search as described in Lesson 31:
- At query time, load all stored document texts from Chroma and build a BM25 index in memory.
- Run BM25 search and embedding search in parallel.
- Merge with RRF, return top 5 results.

For this project size, rebuilding the BM25 index per query is acceptable. For large collections (>10,000 chunks), persist the BM25 index to disk or use a dedicated keyword search backend.

### Source attribution

Every retrieved chunk must carry enough metadata to cite its source. Format the output clearly:

```python
def format_answer_with_sources(answer: str, retrieved_chunks: list[dict]) -> str:
    lines = [f"Answer: {answer}", "", "Sources used:"]
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        source = meta.get("source", "unknown")
        source_type = meta.get("source_type", "")
        
        if source_type == "web":
            location = f'section: {meta.get("title", "")}'
        elif source_type == "pdf":
            location = f'page {meta.get("page", "?")}'
        else:
            location = f'row {meta.get("row_index", "?")}'
        
        preview = chunk["text"][:80].replace("\n", " ")
        lines.append(f'  [{chunk["score"]:.2f}] {source} ({location}) — "{preview}..."')
    
    return "\n".join(lines)
```

## Requirements

- [ ] `python rag.py ingest path/to/file.pdf` ingests a PDF (with chunk count printed)
- [ ] `python rag.py ingest https://example.com/page` ingests a web page
- [ ] `python rag.py ingest data/products.csv` ingests a CSV with each row as a document
- [ ] `python rag.py ask "question"` returns an answer using hybrid search
- [ ] Every answer shows which sources were used, with source type (PDF/web/CSV), location (page/URL/row), and a text preview
- [ ] Re-ingesting a source updates rather than duplicates its chunks
- [ ] Questions about all three source types are answered correctly

## Test plan

1. **Ingest a PDF** — use any technical PDF (a public spec, a white paper, a manual). Ask 3 questions with answers in the PDF.

2. **Ingest a web page** — use a public documentation page (e.g., `https://platform.openai.com/docs/api-reference/introduction`). Ask questions about its content.

3. **Ingest a CSV** — create a small CSV with 10 rows of product data (name, price, stock, category). Ask "Which products cost less than $30?" and "How many units of Widget B are in stock?"

4. **Cross-source query** — ask a question whose answer spans two source types (e.g., a general concept from the web page plus a specific data point from the CSV).

5. **Source attribution check** — verify that for each answer, the cited source is actually the one that contains the answer.

## Completion checklist

- [ ] All three source types ingest successfully
- [ ] Hybrid search returns better results than pure embedding search on at least one test query (verify by comparing the ranked lists)
- [ ] Source attribution is correct for 4 out of 5 test questions
- [ ] Re-ingesting a source replaces its chunks without creating duplicates

## Extension challenges

**1. Add re-ranking.** After the hybrid search returns the top 10 results, apply a cross-encoder re-ranker (from Lesson 32) to get the top 5. Does answer quality improve? Does latency increase noticeably?

**2. Add contextual retrieval.** Use the contextual retrieval approach from Lesson 32 to prepend document-level context to each chunk at ingestion time. Compare retrieval quality on chunks that lose context when separated from their source document.

**3. Support Markdown.** Add a Markdown loader that respects heading structure — split on `##` and `###` headings, keeping the heading as part of the chunk text. Ingest a Markdown knowledge base and test retrieval.

**4. Ingestion CLI with progress bar.** When ingesting large sources (a PDF with 100 pages, a CSV with 1,000 rows), add a progress indicator using `tqdm`.

**5. REST API.** Wrap the `ask` command in a FastAPI endpoint: `POST /query` with `{"question": "..."}` returns `{"answer": "...", "sources": [...]}`. Add a second endpoint `POST /ingest` that accepts a URL or file path.
