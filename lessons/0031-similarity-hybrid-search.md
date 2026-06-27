---
layout: lesson
lesson_id: "0031"
chapter: 4
chapter_title: "RAG & Vector Search"
title: "Similarity search and hybrid search"
description: "30–40 min read · Hands-on coding"
prev: "0049-llamaindex.html"
prev_title: "LlamaIndex — RAG without the boilerplate"
next: "0032-advanced-retrieval.html"
next_title: "Advanced retrieval — contextual retrieval, query expansion, re-ranking"
prereqs:
  - "[Lesson 27](0027-embeddings.html): Embeddings and cosine similarity — this lesson extends those concepts"
  - "[Lesson 29](0029-vector-databases.html): Vector databases — you need to know how similarity search is executed"
  - "[Project P010](P010-pdf-qa.html): PDF Q&A system — you have seen pure similarity search in action"
assignment:
  article:
    title: "Hybrid Search and Re-ranking in Production RAG 2026"
    url: "https://appscale.blog/en/blog/hybrid-search-and-reranking-production-rag-bm25-dense-cross-encoder-2026"
    author: "AppScale"
    time: "15 min"
    why: "AppScale runs AI systems at enterprise scale. This 2026 guide covers the full hybrid search + reranking pipeline with benchmarks, code examples, and the specific failure modes that each technique addresses. Reading it before the next lesson means the advanced retrieval concepts will feel like natural extensions rather than new ideas."
  task:
    description: "Implement BM25 search and compare it to embedding search on the same query."
    steps:
      - "Install: `uv add rank-bm25`"
      - "Create `hybrid_demo.py`. Load 10 short text passages (use the same PDF chunks from P010 or write your own)"
      - "Build a BM25 index with `BM25Okapi`. Score the passages against the query: 'return policy enterprise customers'"
      - "Build an embedding index and score the same passages against the same query"
      - "Print both ranked lists side by side"
      - "Find a query where BM25 ranks better and one where embedding search ranks better"
    expected: "Two side-by-side ranked lists. BM25 should score higher for passages containing the exact words from the query; embedding search should score higher for semantic paraphrases."
    why: "Seeing the two ranked lists side by side is the clearest demonstration of why hybrid search outperforms either method alone. You will understand the trade-off intuitively, not just abstractly."
knowledge_check:
  - q: "What is BM25 and what problem does it solve that pure embedding search cannot?"
    a: "BM25 (Best Match 25) is a keyword-based ranking algorithm. It scores documents by how often query terms appear in them, normalised for document length. It excels at exact keyword matching — product codes, names, identifiers, and rare technical terms that may have imprecise embeddings. Embedding search finds semantically similar text but can miss documents containing an exact technical term the user searched for."
    section: "#bm25-keyword-search"
    section_title: "BM25 keyword search"
  - q: "What is Reciprocal Rank Fusion (RRF) and why is it used to combine search results?"
    a: "RRF is a simple algorithm for merging two ranked lists without relying on their score scales. For each document it computes: score = 1/(rank_in_list1 + k) + 1/(rank_in_list2 + k), where k is a constant (usually 60). Documents that rank high in both lists get the highest combined score. RRF is used because BM25 and embedding search use incomparable score scales — you cannot simply add them together. RRF avoids that problem by working with ranks, not scores."
    section: "#reciprocal-rank-fusion"
    section_title: "Reciprocal Rank Fusion (RRF)"
  - q: "Give a concrete example of a query where keyword search outperforms embedding search."
    a: "Any query containing a specific product code, serial number, legal citation, medical term, or proper noun. For example, 'invoice INV-20260312' or 'clause 4(b)(ii) of the agreement' or 'error code E_OAUTH_TOKEN_EXPIRED'. These strings have a precise, non-semantic meaning that embedding models may not represent sharply. BM25 will find them immediately; embedding search may return conceptually related but wrong results."
    section: "#when-each-works-better"
    section_title: "When each search type works better"
  - q: "What is the standard hybrid search pipeline in a production RAG system?"
    a: "Query → run BM25 search (top 20 results) in parallel with embedding search (top 20 results) → merge both lists using RRF → optionally re-rank the top 20–30 results with a cross-encoder → pass the top 5–10 to the LLM as context. The parallel searches maximise recall; RRF combines them fairly; the re-ranker improves precision."
    section: "#the-hybrid-pipeline"
    section_title: "The hybrid search pipeline"
additional_resources:
  - title: "Hybrid Search for RAG: Combining BM25 and Dense Vector Search (2026)"
    url: "https://denser.ai/blog/hybrid-search-for-rag/"
    desc: "Benchmarks showing a 7.4% NDCG lift from hybrid over either method alone, with implementation code"
  - title: "rank-bm25 on PyPI"
    url: "https://pypi.org/project/rank-bm25/"
    desc: "The BM25 implementation used in this lesson"
  - title: "Weaviate hybrid search"
    url: "https://weaviate.io/developers/weaviate/search/hybrid"
    desc: "How Weaviate implements hybrid search natively — useful if you use a managed vector database with built-in BM25"
  - title: "Qdrant sparse vectors"
    url: "https://qdrant.tech/articles/sparse-vectors/"
    desc: "Qdrant's approach to hybrid search using sparse vector representations of BM25 scores"
---

## Motivation

Pure embedding search has a blind spot: it is bad at exact matches.

Imagine a user searching "error code E_AUTH_EXPIRED". Your knowledge base has a document that says exactly "Error code E_AUTH_EXPIRED occurs when the OAuth token has expired." Embedding search might return documents about authentication in general, because it is searching by meaning — and "authentication problems" are semantically close to many things. The exact string match it needs to find the right document is something keyword search does better.

This lesson covers both search modes, why they fail in complementary ways, and how combining them — hybrid search — is the single biggest quality improvement you can make to a naive RAG pipeline.

{% include prereqs.html %}

## Two kinds of search

Every retrieval system makes a choice between two fundamentally different matching strategies:

**Semantic search** (what you have been doing): represent text as vectors; find documents whose vectors are close to the query vector. Matches by meaning. Handles paraphrasing, synonyms, and conceptual similarity. Fails on specific identifiers and rare terms.

**Keyword search** (what BM25 does): find documents containing the words in the query. Matches by exact word overlap. Reliable for names, codes, and technical terminology. Fails on paraphrasing and intent understanding.

Neither is strictly better. They fail in opposite situations, which is why combining them works.

## When each works better

| Scenario | Better approach |
|----------|----------------|
| "What is the return policy?" | Embedding search — handles paraphrases like "money-back guarantee" |
| "Error code E_AUTH_EXPIRED" | Keyword search — exact code match |
| "How do I calculate compound interest?" | Embedding search — finds financial math even if the exact phrase differs |
| "Invoice INV-20260312 payment status" | Keyword search — specific identifier |
| "Best practices for securing API keys" | Embedding search — conceptual |
| "clause 4(b)(ii) of the master service agreement" | Keyword search — legal citation |

The pattern: use keyword search for things with a precise, non-substitutable form; use embedding search for questions with intent and context.

## BM25 keyword search

**BM25** (Best Match 25) is the standard keyword ranking algorithm. It is what powers Elasticsearch, Solr, and most search engines at their core.

BM25 scores a document against a query by:
1. Counting how often each query term appears in the document
2. Normalising for document length (long documents with many matches are not unfairly favoured)
3. Applying diminishing returns for very high term frequencies (the 25th occurrence matters less than the 1st)

You do not need to understand the exact formula — the key point is that BM25 rewards documents that contain the exact words you searched for.

Using `rank-bm25`:

```python
from rank_bm25 import BM25Okapi

# Your document corpus (chunk texts)
documents = [
    "Error code E_AUTH_EXPIRED occurs when the OAuth token has expired.",
    "Authentication errors are common in distributed systems.",
    "Compound interest is calculated as P*(1+r/n)^(nt).",
    "The return policy covers refunds within 90 days of purchase.",
    "API keys should be stored in environment variables, never in source code.",
]

# Tokenize — BM25 works on word lists
tokenized_docs = [doc.lower().split() for doc in documents]

# Build the BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Query
query = "error code E_AUTH_EXPIRED"
tokenized_query = query.lower().split()

scores = bm25.get_scores(tokenized_query)
for score, doc in sorted(zip(scores, documents), reverse=True):
    print(f"[{score:.4f}] {doc[:70]}")
```

Output:
```text
[5.2341] Error code E_AUTH_EXPIRED occurs when the OAuth token has expired.
[1.1203] Authentication errors are common in distributed systems.
[0.0000] Compound interest is calculated as P*(1+r/n)^(nt).
[0.0000] The return policy covers refunds within 90 days of purchase.
[0.0000] API keys should be stored in environment variables...
```

The exact-match document scores dramatically higher. The generic "authentication" document scores slightly (it shares the word). Unrelated documents score 0.

## The hybrid search approach

Hybrid search runs both BM25 and embedding search on the same query, then merges the two ranked lists. Each search uses its strengths:

- BM25 ensures specific keywords and identifiers are not missed
- Embedding search ensures semantically similar but differently-worded documents are found

**Step 1: Run both searches in parallel and collect results**

```python
from rank_bm25 import BM25Okapi
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Corpus
documents = [...]  # your list of chunk texts

# Build BM25 index
tokenized = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized)

# Build embedding index
doc_embeddings = [embed(doc) for doc in documents]

def keyword_search(query: str, top_k: int = 10) -> list[tuple[int, float]]:
    """Return (doc_index, score) pairs sorted by BM25 score."""
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def semantic_search(query: str, top_k: int = 10) -> list[tuple[int, float]]:
    """Return (doc_index, similarity) pairs sorted by cosine similarity."""
    q_emb = embed(query)
    scores = [(i, cosine_similarity(q_emb, e)) for i, e in enumerate(doc_embeddings)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
```

## Reciprocal Rank Fusion (RRF)

The two search methods return scores on completely different scales — BM25 might return scores like 5.2, 1.1, 0.3; embedding similarity returns values between -1 and 1. You cannot add them together.

**Reciprocal Rank Fusion (RRF)** solves this by ignoring scores entirely and working only with ranks. For each document in each list, it computes:

```
RRF score = 1 / (rank + k)
```

Where `rank` is the document's position in the list (0-indexed) and `k` is a constant (typically 60) that prevents the highest-ranked document from dominating.

A document that appears at rank 0 in the keyword list and rank 2 in the embedding list gets an RRF score of `1/60 + 1/62 ≈ 0.033`. A document that appears at rank 5 in both lists gets `1/65 + 1/65 ≈ 0.031`. Documents that rank highly in both searches win.

```python
def reciprocal_rank_fusion(
    keyword_results: list[tuple[int, float]],
    semantic_results: list[tuple[int, float]],
    k: int = 60
) -> list[tuple[int, float]]:
    """Merge two ranked lists using RRF. Returns (doc_index, rrf_score) pairs."""
    scores = {}
    
    for rank, (doc_idx, _) in enumerate(keyword_results):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (rank + k)
    
    for rank, (doc_idx, _) in enumerate(semantic_results):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (rank + k)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Run hybrid search
query = "error code E_AUTH_EXPIRED when does it happen"

kw_results = keyword_search(query, top_k=10)
sem_results = semantic_search(query, top_k=10)
hybrid_results = reciprocal_rank_fusion(kw_results, sem_results)

print("Hybrid search results:")
for doc_idx, score in hybrid_results[:5]:
    print(f"  [{score:.4f}] {documents[doc_idx][:80]}")
```

## The hybrid pipeline

A production hybrid search pipeline looks like this:

```
User query
    │
    ├──▶ BM25 search (top 20)  ──┐
    │                             ├──▶ RRF merge (top 30) ──▶ LLM context
    └──▶ Embedding search (top 20)─┘
```

Both searches run in parallel (or with `asyncio.gather`). RRF merges them. The merged list is passed to the LLM.

In practice, you often add a re-ranking step between RRF and the LLM — a more expensive model that scores the top 20–30 results precisely and returns the best 5–10. Re-ranking is covered in Lesson 32.

```python
async def hybrid_retrieve(
    query: str,
    documents: list[str],
    bm25_index: BM25Okapi,
    doc_embeddings: list[list[float]],
    top_k: int = 5
) -> list[str]:
    """Full hybrid retrieval: BM25 + semantic + RRF."""
    import asyncio
    
    # Run both searches (could be parallelised with asyncio if both are async)
    kw_results = keyword_search(query, top_k=20)
    sem_results = semantic_search(query, top_k=20)
    
    hybrid = reciprocal_rank_fusion(kw_results, sem_results)
    
    # Return the top-K document texts
    return [documents[idx] for idx, _ in hybrid[:top_k]]
```

## Hybrid search in vector databases

Several vector databases implement hybrid search natively — you do not need to implement BM25 yourself:

**Weaviate** supports hybrid search with a single API call that runs BM25 and vector search internally:

```python
results = client.query.get("Document", ["text"]).with_hybrid(
    query="error code E_AUTH_EXPIRED",
    alpha=0.5  # 0 = pure BM25, 1 = pure embedding, 0.5 = equal weight
).with_limit(10).do()
```

**Qdrant** supports sparse vectors (a dense representation of BM25 scores) alongside dense embedding vectors.

**Chroma** (the database used in this chapter) does not yet support hybrid search natively — you implement BM25 separately and merge the results as shown above.

## What comes next

Hybrid search significantly improves recall — you are less likely to miss relevant documents. But retrieving the right documents is only half the battle. The next lesson covers how to further improve the quality of what you retrieve: **contextual retrieval** (adding document context to each chunk at ingestion time), **query expansion** (generating multiple search queries from one user question), and **re-ranking** (using a more precise model to rank the retrieved results before passing them to the LLM).
