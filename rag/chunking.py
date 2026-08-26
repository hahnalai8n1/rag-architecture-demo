"""Three chunking strategies, kept intentionally simple so the effect on
retrieval is easy to see and explain live."""

import re


def fixed_size_chunks(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def sentence_chunks(text: str, max_chars: int = 400) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def paragraph_chunks(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


STRATEGIES = {
    "fixed": fixed_size_chunks,
    "sentence": sentence_chunks,
    "paragraph": paragraph_chunks,
}
