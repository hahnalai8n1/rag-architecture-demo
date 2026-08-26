"""Tiny disk-backed cache used to demonstrate embedding cache and LLM
response cache. Deliberately simple (a pickled dict) -- the point is to make
cache hits vs misses visible in the UI, not to build a production cache."""

import hashlib
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


class DiskCache:
    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.pkl"
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            with open(self.path, "rb") as f:
                self._store: dict = pickle.load(f)
        else:
            self._store = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str):
        key = self._key(text)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, text: str, value) -> None:
        self._store[self._key(text)] = value
        with open(self.path, "wb") as f:
            pickle.dump(self._store, f)

    def clear(self) -> None:
        self._store = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            self.path.unlink()
