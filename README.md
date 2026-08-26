# RAG Observability Demo

A small Retrieval-Augmented Generation (RAG) app built to *show*, not just tell, the concepts behind a RAG pipeline: chunking strategy, multi-layer caching, and latency metrics.

Instead of another "chat with your PDF" wrapper, this app indexes the same 5 short documents (about RAG itself — vector search, chunking, caching, evaluation metrics, and RAG vs. fine-tuning vs. prompt engineering) three different ways, and lets you compare them side by side, live, for any question you ask.

## What it demonstrates

- **Chunking strategies** — fixed-size, sentence-based, and paragraph-based chunking, run against the same source docs and the same question, so you can see how chunk boundaries change what gets retrieved and how the answer reads.
- **Caching layers** — an on-disk embedding cache and LLM response cache. Ask the same question twice and watch `generation_ms` collapse from ~1-3 seconds to ~1 millisecond on the second call.
- **Metrics** — every query reports a latency breakdown (embed query / retrieval / generation) plus a running history table and chart, so the cost of each pipeline stage is visible instead of hidden behind a single "total time."

## Stack

100% local, no API keys required:

- **Embeddings & generation**: [Ollama](https://ollama.com) running `embeddinggemma` (embeddings) and `gemma3:1b` (generation)
- **Vector store**: [Chroma](https://www.trychroma.com/) (persistent, local)
- **UI**: [Streamlit](https://streamlit.io/)

## Setup

```bash
# 1. Make sure Ollama is running with the two models pulled
ollama pull embeddinggemma
ollama pull gemma3:1b

# 2. Install Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open http://localhost:8501.

## How it's organized

```
app.py                 Streamlit UI
rag/
  chunking.py           fixed / sentence / paragraph chunking functions
  embeddings.py          Ollama embedding calls + disk cache
  llm.py                 Ollama generation calls + disk cache
  cache.py               generic disk-backed cache (hit/miss tracking)
  pipeline.py             ties it together: build_index() + query()
data/docs/              the 5 source documents that get chunked & indexed
```

`chroma_db/` and `cache/` are created at runtime and gitignored — delete them to reset the demo from scratch.

## Background

Built while researching RAG architecture: chunking strategies, the four caching layers in a RAG pipeline (query / embedding / vector-search / LLM-response cache), and the metrics used to monitor one in production (latency, throughput, error rate, retrieval quality, embedding performance, chunking efficiency).
