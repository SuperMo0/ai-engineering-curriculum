---
layout: lesson
lesson_id: "0049"
chapter: 4
chapter_title: "RAG & Vector Search"
title: "LlamaIndex — RAG without the boilerplate"
description: "30–40 min read · Hands-on coding"
prev: "0030-ingestion-pipeline.html"
prev_title: "Building an ingestion pipeline"
next: "0031-similarity-hybrid-search.html"
next_title: "Similarity search and hybrid search"
prereqs:
  - "[Lesson 30](0030-ingestion-pipeline.html): Building an ingestion pipeline — LlamaIndex wraps what you built there; you need to have built it manually first"
  - "[Lesson 29](0029-vector-databases.html): Vector databases — LlamaIndex sits on top of a vector store"
  - "[Project P010](P010-pdf-qa.html): Build a PDF Q&A system — LlamaIndex replaces most of that code"
assignment:
  article:
    title: "LlamaIndex in Python: A RAG Guide With Examples"
    url: "https://realpython.com/llamaindex-examples/"
    author: "Real Python"
    time: "20 min"
    why: "Real Python is one of the most reliable technical writing sources in the Python ecosystem. This guide builds a complete RAG application step by step, explaining what LlamaIndex is doing at each stage — exactly the bridge between your manual pipeline and the framework."
  task:
    description: "Rebuild your PDF Q&A system using LlamaIndex."
    steps:
      - "Install: `uv add llama-index llama-index-embeddings-openai llama-index-llms-openai`"
      - "Load the same PDF you used in P010 using `SimpleDirectoryReader`"
      - "Build a `VectorStoreIndex` from the documents"
      - "Create a `query_engine` and ask it the same 5 questions you tested in P010"
      - "Compare: how many lines of code is the LlamaIndex version vs your manual version?"
    expected: "Correct answers to the same questions as P010, in roughly 15–20 lines of code. The LlamaIndex version should produce the same quality results as your manual pipeline."
    why: "The goal is not to see LlamaIndex work — it's to feel the trade-off. Once you have written both versions, you can make an informed decision about when the abstraction is worth the dependency."
knowledge_check:
  - q: "What is LlamaIndex, in one sentence?"
    a: "LlamaIndex is a Python framework that wraps the five stages of a RAG pipeline (load, extract, chunk, embed, store/query) into a concise API, so you can build a working RAG system in a few lines of code instead of wiring each stage manually."
    section: "#what-llamaindex-is"
    section_title: "What LlamaIndex is"
  - q: "What does a `VectorStoreIndex` do?"
    a: "It takes a list of documents, chunks them, embeds each chunk, and stores the embeddings in an in-memory (or configured) vector store. It also provides a query interface. It is LlamaIndex's main abstraction for the index step of a RAG pipeline."
    section: "#the-core-abstractions"
    section_title: "The core abstractions"
  - q: "What is a query engine in LlamaIndex?"
    a: "A query engine is an object that, given a question, performs retrieval against the index and calls an LLM to generate an answer from the retrieved context. Calling `index.as_query_engine()` creates one. It combines the retrieve and answer steps into a single `.query()` call."
    section: "#query-engines"
    section_title: "Query engines"
  - q: "When should you use LlamaIndex versus building a pipeline manually?"
    a: "Use LlamaIndex when you need a working prototype quickly, when your pipeline follows standard patterns (load files, embed, query), or when you want to swap components (embedding model, LLM, vector store) without rewriting code. Roll your own when you need tight control over chunking logic, custom document formats, specific metadata handling, or when the abstraction's defaults are wrong for your use case. Always understand the manual pipeline first."
    section: "#when-to-use-llamaindex"
    section_title: "When to use LlamaIndex"
additional_resources:
  - title: "LlamaIndex documentation"
    url: "https://docs.llamaindex.ai/"
    desc: "Full reference including custom retrievers, agents, structured extraction, and multi-modal RAG"
  - title: "LlamaIndex with Chroma"
    url: "https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/"
    desc: "How to plug Chroma into LlamaIndex as a persistent vector store instead of the in-memory default"
  - title: "LlamaIndex vs LangChain (2026 comparison)"
    url: "https://www.llamaindex.ai/blog"
    desc: "When LlamaIndex is the better fit vs LangChain for RAG-specific workloads"
---

## Motivation

In Lesson 30 you built an ingestion pipeline with about 150 lines of Python. You wrote the loader, the chunker, the embedder, and the writer. You understand every part. Now a colleague asks you to add RAG to their project over the weekend. Writing 150 lines from scratch for every new project is not the right answer.

LlamaIndex is a framework that packages the RAG pipeline you built manually into reusable abstractions. A basic RAG system goes from 150 lines to about 15. But abstractions have costs — they hide what is happening, they make unusual cases harder to handle, and they add a dependency with its own update cycle. This lesson teaches you both sides: what LlamaIndex does for you and when you should skip it.

{% include prereqs.html %}

## What LlamaIndex is

LlamaIndex (formerly GPT Index) is an open-source Python framework for building RAG applications. It wraps the five stages of the RAG ingestion pipeline:

| Stage | Manual (what you built) | LlamaIndex abstraction |
|-------|-------------------------|----------------------|
| Load | `fitz.open()`, `Path.read_text()` | `SimpleDirectoryReader` |
| Extract | `page.get_text()`, string handling | Built into the readers |
| Chunk | `RecursiveCharacterTextSplitter` | `SentenceSplitter` (built-in) |
| Embed | `client.embeddings.create()` | `OpenAIEmbedding` |
| Write | `collection.add()` | `VectorStoreIndex` |
| Query | `collection.query()` + `completions.create()` | `query_engine.query()` |

Each abstraction has a standard interface and can be swapped for a different implementation without rewriting the surrounding code. You can replace the embedding model without changing the chunker. You can replace Chroma with Pinecone without changing the query engine.

## Installation

```bash
uv add llama-index llama-index-embeddings-openai llama-index-llms-openai
```

LlamaIndex is modular — you install only the integrations you need. The base package plus the OpenAI integration is enough to get started.

## The five-line RAG system

This is the complete RAG pipeline in LlamaIndex:

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load all documents in a directory
documents = SimpleDirectoryReader("./docs").load_data()

# Build the index (chunks, embeds, and stores in memory)
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Ask a question
response = query_engine.query("What is the main argument of chapter 3?")
print(response)
```

That is it. LlamaIndex handles the loading, chunking, embedding, and vector storage. The `query` call handles retrieval and generation.

This uses OpenAI's `gpt-4o` and `text-embedding-3-small` by default. You can configure these, as shown below.

## The core abstractions

### SimpleDirectoryReader

Loads all files in a directory. Supports PDF, .txt, .md, Word, PowerPoint, HTML, and more. Returns a list of `Document` objects, each containing the text and metadata:

```python
from llama_index.core import SimpleDirectoryReader

# Load a single file
documents = SimpleDirectoryReader(input_files=["report.pdf"]).load_data()

# Load all PDFs in a directory
documents = SimpleDirectoryReader(
    input_dir="./docs",
    required_exts=[".pdf"]
).load_data()

print(len(documents))               # Number of pages/documents loaded
print(documents[0].metadata)        # Source, page, file type, etc.
print(documents[0].text[:200])      # Raw extracted text
```

### VectorStoreIndex

Takes documents, chunks them, embeds each chunk, and stores the embeddings:

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Configure the models explicitly
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 400
Settings.chunk_overlap = 80

# Build the index
index = VectorStoreIndex.from_documents(documents)
```

The index is in-memory by default. The next section shows how to persist it.

### Persistent storage with Chroma

The in-memory index is lost when your script exits. To persist:

```python
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Connect to a persistent Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Build the index with the persistent store
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Next time: load the existing index without re-ingesting
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)
```

### Query engines

A query engine wraps retrieval and generation into a single `.query()` call:

```python
# Default: retrieves top 2 chunks, uses gpt-4o
query_engine = index.as_query_engine()
response = query_engine.query("What is the return policy?")
print(response.response)          # The answer text
print(response.source_nodes)      # The retrieved chunks that were used
```

You can configure retrieval:

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,       # Retrieve 5 chunks instead of 2
    response_mode="compact",  # Summarise multiple chunks into one answer
)
```

To show sources alongside the answer:

```python
response = query_engine.query("What are the key risks?")
print(f"Answer: {response.response}\n")
print("Sources:")
for node in response.source_nodes:
    print(f"  [{node.score:.4f}] {node.text[:100]}...")
```

## Side-by-side comparison

Here is the same task — load a PDF, build an index, answer a question — in the manual pipeline vs LlamaIndex:

**Manual pipeline (~40 lines):**
```python
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb, numpy as np

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

doc = fitz.open("report.pdf")
text = "\n".join(page.get_text() for page in doc)

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_text(text)

embeddings = client.embeddings.create(
    model="text-embedding-3-small", input=chunks
).data
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=[e.embedding for e in embeddings]
)

question = "What is the main finding?"
q_emb = client.embeddings.create(
    model="text-embedding-3-small", input=question
).data[0].embedding

results = collection.query(query_embeddings=[q_emb], n_results=3)
context = "\n".join(results["documents"][0])

answer = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"Answer from this context:\n{context}"},
        {"role": "user", "content": question}
    ]
).choices[0].message.content
print(answer)
```

**LlamaIndex (~10 lines):**
```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader(input_files=["report.pdf"]).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What is the main finding?")
print(response.response)
for node in response.source_nodes:
    print(f"  [{node.score:.3f}] {node.text[:80]}...")
```

The manual version gives you full control. The LlamaIndex version gets you to a working prototype in minutes.

## When to use LlamaIndex

Use LlamaIndex when:
- You need a working RAG system quickly and standard patterns fit your problem
- You want to swap components without rewriting surrounding code (e.g., change embedding model, switch from Chroma to Weaviate)
- You are building a prototype or exploring what RAG can do for your data
- Your documents are in standard formats that LlamaIndex readers support

Roll your own when:
- You need custom chunking logic that respects your document's specific structure (tables, code blocks, custom sections)
- Your document format is not supported by LlamaIndex readers and the custom reader interface is more complex than writing a loader from scratch
- You need precise control over how metadata is attached to chunks
- Your retrieval logic is non-standard (e.g., filtering by date before embedding search)
- You want to minimise dependencies in a production system

<div class="callout info">
<strong>The right mental model:</strong> LlamaIndex is a productivity tool for standard RAG patterns. It is not a replacement for understanding the pipeline. If you had not built the manual pipeline in Lessons 27–30, using LlamaIndex would be opaque — you would not know what to tune when retrieval quality is poor. Now that you have built it manually, LlamaIndex saves you time on straightforward cases.
</div>

## What comes next

You have now built a full RAG pipeline both from scratch and with a framework. The next lesson addresses a quality problem that affects all RAG systems: pure embedding search fails when users search with specific keywords, product codes, or exact phrases. The fix is **hybrid search** — combining embedding search with keyword search.
