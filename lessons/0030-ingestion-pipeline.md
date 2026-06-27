---
layout: lesson
lesson_id: "0030"
chapter: 4
chapter_title: "RAG & Vector Search"
title: "Building an ingestion pipeline"
description: "30–40 min read · Hands-on coding"
prev: "P010-pdf-qa.html"
prev_title: "Build a PDF Q&A system"
next: "0049-llamaindex.html"
next_title: "LlamaIndex — RAG without the boilerplate"
prereqs:
  - "[Project P010](P010-pdf-qa.html): Build a PDF Q&A system — the ingestion logic you built there is what this lesson productionises"
  - "[Lesson 28](0028-chunking.html): Chunking strategies — you will extend the chunking approach to multiple document types"
  - "[Lesson 20](0020-async-python.html): Async Python — the pipeline uses async for concurrent embedding calls"
assignment:
  article:
    title: "How to Build a RAG Pipeline from Scratch in 2026"
    url: "https://www.kapa.ai/blog/how-to-build-a-rag-pipeline-from-scratch-in-2026"
    author: "kapa.ai"
    time: "15 min"
    why: "kapa.ai builds AI assistants for developer documentation at scale. This article reflects hard-won lessons from running production ingestion pipelines over millions of documents — it covers the failure modes and operational concerns you don't encounter until your pipeline runs in the real world."
  task:
    description: "Add incremental update support to your PDF Q&A ingestion pipeline."
    steps:
      - "Create `ingested.json` — a JSON file that records the path and modification time of every ingested file"
      - "At the start of ingest, check if the file is in `ingested.json` and whether its modification time has changed"
      - "Skip ingestion if the file hasn't changed; re-ingest (delete old chunks, re-embed) if it has"
      - "Update `ingested.json` after successful ingestion"
      - "Test: ingest a PDF, check the JSON, modify the PDF (or `touch` it to update its mtime), run ingest again, and confirm the chunks were replaced"
    expected: "Ingesting the same file twice with no changes prints 'Already up to date, skipping.' Ingesting after a modification re-ingests and updates the record."
    why: "Idempotency is what separates a script you run once from a pipeline you run on a schedule. Every production ingestion system needs to handle re-runs gracefully."
knowledge_check:
  - q: "What are the five stages of a document ingestion pipeline, in order?"
    a: "Load → Extract → Chunk → Embed → Write. Load brings the raw file into memory. Extract pulls plain text from it (stripping formatting, PDF metadata, HTML tags). Chunk splits the text into retrievable units. Embed converts each chunk to a vector. Write stores the vectors and metadata in the vector database."
    section: "#the-five-stages"
    section_title: "The five stages"
  - q: "What is idempotency in the context of an ingestion pipeline, and why does it matter?"
    a: "An idempotent ingestion pipeline can be run multiple times on the same document without creating duplicate entries. It matters because pipelines are run on schedules — daily, hourly, on file change — and every run must produce the same result as the first run. Without idempotency, each run adds more duplicate chunks, which pollutes the vector store and degrades retrieval quality."
    section: "#idempotency"
    section_title: "Idempotency — safe re-runs"
  - q: "Why should embedding API calls be batched rather than called one chunk at a time?"
    a: "Each individual API call has network round-trip latency (~100–300ms). A document split into 100 chunks would take 10–30 seconds if embedded one at a time. Batching all 100 chunks into one API call reduces that to a single round-trip (~300ms). The OpenAI embeddings endpoint accepts a list of strings and returns all embeddings in one response."
    section: "#embedding-in-batches"
    section_title: "Embedding in batches"
  - q: "What is an incremental update and when would you use it over a full re-ingest?"
    a: "An incremental update only re-ingests documents that have changed since the last run, and leaves unchanged documents alone. You use it when your document corpus is large — re-embedding thousands of unchanged documents on every run is expensive and slow. Full re-ingest is simpler and acceptable for small collections or when major restructuring has occurred."
    section: "#incremental-updates"
    section_title: "Incremental updates"
additional_resources:
  - title: "Unstructured.io"
    url: "https://unstructured.io/"
    desc: "Open-source library for extracting text from PDFs, Word documents, HTML, Excel, PowerPoint, and more — useful when your pipeline must handle mixed document types"
  - title: "PyMuPDF documentation"
    url: "https://pymupdf.readthedocs.io/"
    desc: "Fast PDF processing library used in this lesson — reference for page extraction, text blocks, and bounding boxes"
  - title: "LlamaIndex ingestion pipeline"
    url: "https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/"
    desc: "LlamaIndex's built-in ingestion pipeline, which implements the same stages covered in this lesson — useful to compare once you have built your own"
---

## Motivation

In Project P010 you wrote a working ingestion pipeline for a single PDF. It ran once. In production, ingestion runs on a schedule — every hour, every time a document is uploaded, every time a file changes. The script from P010 is not ready for that: it would re-embed the same document every run, accumulating duplicate chunks in the vector store. It has no error handling, no logging, and no way to skip files that haven't changed.

This lesson productionises the ingestion side of RAG. You will build a pipeline that handles multiple document types, batches embedding calls for efficiency, runs idempotently (safe to re-run), and supports incremental updates that only process what has changed.

{% include prereqs.html %}

## The five stages

Every RAG ingestion pipeline has the same five stages, regardless of the framework or tools used:

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  Load   │──▶│ Extract │──▶│  Chunk  │──▶│  Embed  │──▶│  Write  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
  Raw file      Plain text    List of       List of        Vectors in
  from disk     from file     text chunks   vectors        vector DB
```

**Load** — read the raw file from disk, an API, a URL, or a database.

**Extract** — pull plain text out of whatever format the file is in. A PDF has structure, images, and metadata that need to be stripped. An HTML page has tags and navigation elements. A Word document has formatting markup. The extractor produces clean, readable text.

**Chunk** — split the text into retrievable units using the strategies from Lesson 28.

**Embed** — convert each chunk to a vector using the embedding model. This is the most expensive step (API cost and latency).

**Write** — store each (vector, chunk_text, metadata) tuple in the vector database.

## Loading multiple document types

Different file formats need different loaders. A production pipeline handles at least PDFs, plain text, and Markdown:

```python
import fitz  # PyMuPDF
from pathlib import Path

def load_document(file_path: str) -> str:
    """Load a file and return its text content."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def _load_pdf(file_path: str) -> str:
    """Extract text from a PDF, preserving page boundaries."""
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[Page {page_num}]\n{text}")
    doc.close()
    return "\n\n".join(pages)
```

The `[Page N]` markers preserved in the extracted text give you page references that you can store in chunk metadata.

## Chunking with metadata

When chunking, attach everything you know about the source:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import re

def chunk_document(text: str, file_path: str) -> list[dict]:
    """Split text into chunks with source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )
    
    raw_chunks = splitter.split_text(text)
    
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        # Try to extract page number from the [Page N] markers
        page_match = re.search(r'\[Page (\d+)\]', chunk_text)
        page_num = int(page_match.group(1)) if page_match else None
        
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": Path(file_path).name,
                "file_path": str(file_path),
                "chunk_index": i,
                "page": page_num,
            }
        })
    
    return chunks
```

## Embedding in batches

Embedding one chunk at a time is slow. The OpenAI API accepts a list of strings and returns all embeddings in a single round-trip:

```python
from openai import OpenAI

client = OpenAI()

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add embeddings to a list of chunk dicts. Batches all API calls."""
    texts = [chunk["text"] for chunk in chunks]
    
    # One API call for all chunks
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    embeddings = [item.embedding for item in response.data]
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return chunks
```

For very large documents (thousands of chunks), the API has a batch size limit (typically 2,048 inputs per call). Split into sub-batches:

```python
def embed_chunks_batched(chunks: list[dict], batch_size: int = 500) -> list[dict]:
    result = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        result.extend(embed_chunks(batch))
    return result
```

## Writing to the vector store

Store each chunk with its embedding and metadata:

```python
import chromadb
from chromadb.utils import embedding_functions
import uuid

def write_to_vector_store(chunks: list[dict], collection_name: str = "docs") -> None:
    """Write embedded chunks to Chroma."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [chunk["text"] for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"Wrote {len(chunks)} chunks to collection '{collection_name}'")
```

## Idempotency

A pipeline that can be safely re-run is called **idempotent**. The simplest approach: delete the chunks from a source file before re-ingesting it.

In Chroma you can filter by metadata when deleting:

```python
def delete_file_chunks(collection, file_path: str) -> None:
    """Delete all chunks from a specific source file."""
    results = collection.get(where={"file_path": str(file_path)})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"Deleted {len(results['ids'])} existing chunks for {file_path}")
```

The full ingest function with idempotency:

```python
def ingest(file_path: str) -> None:
    """Full pipeline: load → chunk → embed → write. Idempotent."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="docs",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Delete any existing chunks for this file before re-ingesting
    delete_file_chunks(collection, file_path)
    
    # Run the pipeline
    text = load_document(file_path)
    chunks = chunk_document(text, file_path)
    chunks = embed_chunks_batched(chunks)
    write_to_vector_store(chunks)
    
    print(f"Ingested {len(chunks)} chunks from {file_path}")
```

## Incremental updates

Re-embedding every document on every run is wasteful. An incremental update only processes files that have changed:

```python
import json
import os
from pathlib import Path

INGEST_RECORD = "ingested.json"

def load_ingest_record() -> dict:
    if Path(INGEST_RECORD).exists():
        return json.loads(Path(INGEST_RECORD).read_text())
    return {}

def save_ingest_record(record: dict) -> None:
    Path(INGEST_RECORD).write_text(json.dumps(record, indent=2))

def file_has_changed(file_path: str, record: dict) -> bool:
    """Return True if the file is new or has been modified since last ingest."""
    mtime = os.path.getmtime(file_path)
    return record.get(file_path) != mtime

def ingest_if_changed(file_path: str) -> None:
    record = load_ingest_record()
    
    if not file_has_changed(file_path, record):
        print(f"Already up to date, skipping: {file_path}")
        return
    
    ingest(file_path)
    
    # Update the record after successful ingestion
    record[file_path] = os.path.getmtime(file_path)
    save_ingest_record(record)
```

Now run this over a directory of documents:

```python
def ingest_directory(directory: str) -> None:
    """Incrementally ingest all supported documents in a directory."""
    supported = {".pdf", ".txt", ".md"}
    files = [
        f for f in Path(directory).rglob("*")
        if f.suffix.lower() in supported
    ]
    
    print(f"Found {len(files)} documents")
    for file_path in files:
        ingest_if_changed(str(file_path))
```

## Error handling

Production pipelines fail — files are corrupted, the API is temporarily down, a chunk is too long. Handle errors without stopping the whole run:

```python
import time

def ingest_with_retry(file_path: str, max_retries: int = 3) -> None:
    for attempt in range(max_retries):
        try:
            ingest_if_changed(file_path)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"Error ingesting {file_path}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed to ingest {file_path} after {max_retries} attempts: {e}")
```

## Putting it all together

```python
# run_ingestion.py
from pathlib import Path
from loader import load_document
from chunker import chunk_document
from embedder import embed_chunks_batched
from writer import write_to_vector_store, delete_file_chunks
from tracker import load_ingest_record, save_ingest_record, file_has_changed
import chromadb, os

def run(directory: str) -> None:
    """Incrementally ingest a directory of documents."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="docs", metadata={"hnsw:space": "cosine"}
    )
    record = load_ingest_record()
    
    files = list(Path(directory).rglob("*.pdf")) + \
            list(Path(directory).rglob("*.txt")) + \
            list(Path(directory).rglob("*.md"))
    
    skipped, ingested = 0, 0
    for file_path in files:
        path_str = str(file_path)
        if not file_has_changed(path_str, record):
            skipped += 1
            continue
        try:
            delete_file_chunks(collection, path_str)
            text = load_document(path_str)
            chunks = chunk_document(text, path_str)
            chunks = embed_chunks_batched(chunks)
            collection.add(
                ids=[f"{file_path.stem}_{i}" for i in range(len(chunks))],
                documents=[c["text"] for c in chunks],
                embeddings=[c["embedding"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
            record[path_str] = os.path.getmtime(path_str)
            ingested += 1
            print(f"Ingested {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    save_ingest_record(record)
    print(f"\nDone. Ingested: {ingested}, Skipped (no change): {skipped}")

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
```

## What comes next

You now have a production-grade ingestion pipeline. The next lesson introduces **LlamaIndex**, a framework that wraps everything you just built — loaders, chunkers, embedders, and writers — into a concise abstraction. Once you understand what LlamaIndex is doing under the hood (which you now do), you can use it to save time without losing visibility into what is happening.
