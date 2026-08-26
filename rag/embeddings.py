"""Embeddings via a local Ollama model (embeddinggemma), with a disk cache
so re-embedding the same text is instant on the second call."""

import requests

from rag.cache import DiskCache

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "embeddinggemma:latest"

_cache = DiskCache("embeddings")


def embed(text: str) -> tuple[list[float], bool]:
    """Return (embedding, was_cache_hit)."""
    cached = _cache.get(text)
    if cached is not None:
        return cached, True

    response = requests.post(
        OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60
    )
    response.raise_for_status()
    vector = response.json()["embedding"]
    _cache.set(text, vector)
    return vector, False


def cache_stats() -> dict:
    return {"hits": _cache.hits, "misses": _cache.misses}


def clear_cache() -> None:
    _cache.clear()
