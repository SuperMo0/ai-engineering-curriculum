---
layout: project
lesson_id: "P010"
chapter: 4
chapter_title: "RAG & Vector Search"
project_type: "Project"
title: "Build a PDF Q&A system"
description: "Estimated time: 2–3 hours"
prev: "0029-vector-databases.html"
prev_title: "Vector databases — storing and searching embeddings"
next: "0030-ingestion-pipeline.html"
next_title: "Building an ingestion pipeline"
prereqs:
  - "[Lesson 26](0026-what-is-rag.html): What is RAG — the concept you are implementing"
  - "[Lesson 27](0027-embeddings.html): Embeddings — you will use `text-embedding-3-small` directly"
  - "[Lesson 28](0028-chunking.html): Chunking — you will chunk the PDF before embedding"
  - "[Lesson 29](0029-vector-databases.html): Vector databases — Chroma is your storage layer"
---

## Overview

You have learned the three foundational pieces of RAG — embeddings, chunking, and vector databases. Now you combine them into a working system.

You will build a command-line tool that ingests any PDF, stores its content as searchable embeddings, and answers questions about it. The tool will also show which specific passages it used to answer — a transparency feature that is essential in production RAG systems.

By the end of this project you will have a complete, functional RAG pipeline that you built without a framework. You will understand every layer. That is the foundation for using higher-level tools (LlamaIndex, LangChain) confidently — you will know exactly what they are doing for you.

## What you are building

A two-mode CLI tool:

```bash
# Mode 1: Ingest a PDF into the vector store
python qa.py ingest path/to/document.pdf

# Mode 2: Ask a question and get an answer
python qa.py ask "What is the main argument of chapter 3?"
```

The ingestion mode loads a PDF, splits it into chunks, embeds each chunk, and stores the embeddings in Chroma. The Q&A mode embeds the user's question, retrieves the top 3 most relevant chunks, and asks GPT-4o to answer using only those chunks.

## Spec

### Project structure

```
pdf_qa/
├── qa.py              # CLI entry point (ingest + ask commands)
├── ingest.py          # PDF loading, chunking, embedding, and storage
├── retrieve.py        # Query the vector store, return top-K chunks
├── answer.py          # Build the prompt and call GPT-4o
└── .env               # OPENAI_API_KEY
```

### Dependencies

```bash
uv add openai chromadb pymupdf langchain-text-splitters python-dotenv
```

- `pymupdf` — fast, reliable PDF text extraction
- `chromadb` — local vector store
- `langchain-text-splitters` — `RecursiveCharacterTextSplitter`
- `openai` — embedding API and chat completions

### ingestion pipeline (`ingest.py`)

1. Load the PDF with PyMuPDF (`fitz`). Extract text from each page.
2. Combine pages into a single string, preserving page breaks.
3. Split with `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)`.
4. Embed each chunk with `text-embedding-3-small`. Batch your calls — pass a list of strings to the embedding API rather than one call per chunk.
5. Store in a Chroma collection named `pdf_qa`. Each document must include its chunk index and page number (if available) as metadata.
6. Print a summary: "Ingested N chunks from path/to/document.pdf".

### Retrieval (`retrieve.py`)

1. Embed the user's question.
2. Query the `pdf_qa` Chroma collection for the top 3 results.
3. Return a list of dicts: `[{"text": "...", "source": "filename.pdf", "chunk_index": 4}]`.

### Answer generation (`answer.py`)

1. Format retrieved chunks into a numbered context block:

```
[1] <chunk text>
[2] <chunk text>
[3] <chunk text>
```

2. System prompt: "Answer the question using only the context below. After your answer, list the context numbers you used (e.g. 'Sources: [1], [3]'). If the answer is not in the context, say so."
3. Return both the answer text and the retrieved chunks so the CLI can display the sources.

### CLI (`qa.py`)

```python
import sys
from ingest import ingest_pdf
from retrieve import retrieve_chunks
from answer import answer_question

command = sys.argv[1]

if command == "ingest":
    pdf_path = sys.argv[2]
    ingest_pdf(pdf_path)

elif command == "ask":
    question = sys.argv[2]
    chunks = retrieve_chunks(question, n_results=3)
    answer, cited_chunks = answer_question(question, chunks)
    print(f"\nAnswer:\n{answer}\n")
    print("Retrieved passages:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk['text'][:120]}... (from {chunk['source']})")
```

## Requirements

Your finished system must satisfy all of the following:

- [ ] `python qa.py ingest path/to/file.pdf` runs without error and prints a chunk count
- [ ] Re-running ingest on the same file does not duplicate entries in the vector store (use `get_or_create_collection` and clear before re-ingesting, or use document IDs that deduplicates naturally)
- [ ] `python qa.py ask "your question"` returns a grounded answer that cites context numbers
- [ ] The CLI always prints the retrieved passages so the user can verify the sources
- [ ] Questions clearly answered by the document return correct answers
- [ ] Questions not covered by the document return "I don't have information about that in this document" (or similar), not a hallucinated answer
- [ ] Chunks include source filename and chunk index in metadata
- [ ] Embeddings are batched (single API call for all chunks), not one call per chunk

## Testing your system

Use a real document for testing. Good options:

- **A public annual report**: most companies publish PDFs on their investor relations page. Ask questions like "What was the revenue in fiscal year X?" or "What are the main risk factors?"
- **A research paper**: download any PDF from [arxiv.org](https://arxiv.org). Ask questions about the abstract, methods, and conclusions.
- **A product manual**: any user guide with technical specs, troubleshooting steps, and installation instructions.

For each document, write 5 test questions:
- 2 questions with clear, direct answers in the document
- 2 questions requiring combining information from different sections
- 1 question with no answer in the document (to verify the "not found" behaviour)

## Completion checklist

Run through this checklist before moving to the next lesson:

- [ ] Ingested a real PDF and confirmed the chunk count makes sense for the document size
- [ ] Asked 5 test questions and verified the answers
- [ ] Confirmed that at least one "not found" question returns a graceful non-answer
- [ ] Ran ingest twice on the same file and confirmed no duplicate chunks
- [ ] Inspected the raw retrieved chunks in the CLI output to verify they are relevant

## Extension challenges

Once the core system works, try these to deepen your understanding:

**1. Multi-PDF support.** Allow ingesting multiple PDFs into the same collection. Add the source filename to the collection name or to metadata. When answering, show which PDF each retrieved chunk came from.

**2. Re-ingestion detection.** Instead of clearing the collection on re-ingest, track which files have been ingested (e.g., in a JSON file). Skip re-ingesting files that haven't changed since last ingest.

**3. Top-K tuning.** Try retrieving 5 or 10 chunks instead of 3. Does answer quality improve? Does the answer get longer and noisier? Find the sweet spot for your test document.

**4. Show chunk scores.** Print the cosine similarity score next to each retrieved chunk. Which score threshold reliably separates "actually relevant" from "retrieved by accident"?

**5. Interactive mode.** Add a third command `python qa.py chat` that drops into a REPL loop. The user can ask multiple questions without re-running the script. Keep a history of what was asked.
