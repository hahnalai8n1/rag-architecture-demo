"""Answer generation via a local Ollama model (gemma3:1b), with an LLM
response cache keyed on the exact prompt (question + retrieved context)."""

import requests

from rag.cache import DiskCache

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "gemma3:1b"

_cache = DiskCache("llm_responses")

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about \
Retrieval-Augmented Generation (RAG), using only the context below. \
Keep the answer to 2-4 sentences.

Context:
{context}

Question: {question}

Answer:"""


def generate(question: str, context_chunks: list[str]) -> tuple[str, bool]:
    """Return (answer, was_cache_hit)."""
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    cached = _cache.get(prompt)
    if cached is not None:
        return cached, True

    response = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    answer = response.json()["response"].strip()
    _cache.set(prompt, answer)
    return answer, False


def cache_stats() -> dict:
    return {"hits": _cache.hits, "misses": _cache.misses}


def clear_cache() -> None:
    _cache.clear()
