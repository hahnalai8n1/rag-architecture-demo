"""Ties chunking + embeddings + a Chroma vector store + a BM25 keyword index +
the LLM together, and times every stage so the UI can show a latency
breakdown. Supports three retrieval methods: vector (semantic), bm25
(keyword), and hybrid (reciprocal rank fusion of the two)."""

import re
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from rag.chunking import STRATEGIES
from rag.embeddings import embed
from rag.llm import generate

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

_client = chromadb.PersistentClient(path=str(DB_DIR))
_bm25_index: dict[str, dict] = {}

RRF_K = 60


def _collection_name(strategy: str) -> str:
    return f"chunks_{strategy}"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(strategy: str) -> int:
    """Chunk every doc with the given strategy, embed each chunk (via the
    embedding cache), and (re)populate that strategy's Chroma collection plus
    a matching BM25 keyword index. Returns the number of chunks indexed."""
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

    _bm25_index[strategy] = {
        "bm25": BM25Okapi([_tokenize(d) for d in docs]) if docs else None,
        "ids": ids,
        "docs": docs,
        "metas": metadatas,
    }
    return len(docs)


def _get_collection(strategy: str):
    return _client.get_or_create_collection(_collection_name(strategy))


def _vector_rank(strategy: str, query_vector: list[float], top_n: int) -> list[tuple]:
    collection = _get_collection(strategy)
    results = collection.query(query_embeddings=[query_vector], n_results=top_n)
    return list(
        zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
    )


def _bm25_rank(strategy: str, question: str, top_n: int) -> list[tuple]:
    idx = _bm25_index[strategy]
    if idx["bm25"] is None:
        return []
    scores = idx["bm25"].get_scores(_tokenize(question))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [(idx["ids"][i], idx["docs"][i], idx["metas"][i], float(scores[i])) for i in order]


def query(strategy: str, question: str, method: str = "vector", top_k: int = 3) -> dict:
    """Run the full retrieve-then-generate pipeline for one question, against
    one chunking strategy's index, using one of three retrieval methods:

    - "vector": semantic search over embeddings (the original behavior)
    - "bm25":   lexical keyword search, no embedding call needed
    - "hybrid": reciprocal rank fusion of the two rankings (RAG-Fusion style)

    Returns timings and cache-hit flags for every stage.
    """
    timings: dict[str, float] = {}
    embed_cached = None

    if method == "bm25":
        t0 = time.perf_counter()
        ranked = _bm25_rank(strategy, question, top_k)
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000
        chunks = [{"text": d, "source": m["source"], "score": s} for _, d, m, s in ranked]

    elif method == "vector":
        t0 = time.perf_counter()
        query_vector, embed_cached = embed(question)
        timings["embed_query_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ranked = _vector_rank(strategy, query_vector, top_k)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        chunks = [{"text": d, "source": m["source"], "distance": dist} for _, d, m, dist in ranked]

    elif method == "hybrid":
        t0 = time.perf_counter()
        query_vector, embed_cached = embed(question)
        timings["embed_query_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        vector_ranked = _vector_rank(strategy, query_vector, top_n=10)
        timings["vector_retrieval_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        bm25_ranked = _bm25_rank(strategy, question, top_n=10)
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, tuple] = {}
        for rank, (id_, d, m, _) in enumerate(vector_ranked):
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1 / (RRF_K + rank + 1)
            by_id[id_] = (d, m)
        for rank, (id_, d, m, _) in enumerate(bm25_ranked):
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1 / (RRF_K + rank + 1)
            by_id[id_] = (d, m)
        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000
        chunks = [
            {"text": by_id[id_][0], "source": by_id[id_][1]["source"], "rrf_score": score}
            for id_, score in fused
        ]

    else:
        raise ValueError(f"unknown retrieval method: {method}")

    t0 = time.perf_counter()
    answer, llm_cached = generate(question, [c["text"] for c in chunks])
    timings["generation_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "strategy": strategy,
        "method": method,
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "timings": timings,
        "embed_cache_hit": embed_cached,
        "llm_cache_hit": llm_cached,
    }
