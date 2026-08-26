"""Ties chunking + embeddings + a Chroma vector store + the LLM together,
and times every stage so the UI can show a latency breakdown."""

import time
from pathlib import Path

import chromadb

from rag.chunking import STRATEGIES
from rag.embeddings import embed
from rag.llm import generate

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

_client = chromadb.PersistentClient(path=str(DB_DIR))


def _collection_name(strategy: str) -> str:
    return f"chunks_{strategy}"


def build_index(strategy: str) -> int:
    """Chunk every doc with the given strategy, embed each chunk (via the
    embedding cache), and (re)populate that strategy's Chroma collection.
    Returns the number of chunks indexed."""
    chunk_fn = STRATEGIES[strategy]

    name = _collection_name(strategy)
    if name in [c.name for c in _client.list_collections()]:
        _client.delete_collection(name)
    collection = _client.create_collection(name)

    ids, docs, embeddings, metadatas = [], [], [], []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_fn(text)):
            vector, _ = embed(chunk)
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            embeddings.append(vector)
            metadatas.append({"source": path.stem, "chunk_index": i})

    if docs:
        collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    return len(docs)


def _get_collection(strategy: str):
    return _client.get_or_create_collection(_collection_name(strategy))


def query(strategy: str, question: str, top_k: int = 3) -> dict:
    """Run the full retrieve-then-generate pipeline for one question against
    one chunking strategy's index, returning timings and cache-hit flags for
    every stage."""
    timings = {}

    t0 = time.perf_counter()
    query_vector, embed_cached = embed(question)
    timings["embed_query_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    collection = _get_collection(strategy)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]
    retrieved_dist = results["distances"][0]

    t0 = time.perf_counter()
    answer, llm_cached = generate(question, retrieved_docs)
    timings["generation_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = sum(timings.values())

    return {
        "strategy": strategy,
        "question": question,
        "answer": answer,
        "chunks": [
            {"text": d, "source": m["source"], "distance": dist}
            for d, m, dist in zip(retrieved_docs, retrieved_meta, retrieved_dist)
        ],
        "timings": timings,
        "embed_cache_hit": embed_cached,
        "llm_cache_hit": llm_cached,
    }
