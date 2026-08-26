import pandas as pd
import streamlit as st

from rag import embeddings, llm
from rag.pipeline import build_index, query

st.set_page_config(page_title="RAG Observability Demo", page_icon="🔎", layout="wide")

STRATEGIES = ["fixed", "sentence", "paragraph"]
SAMPLE_QUESTIONS = [
    "What is the difference between vector search and keyword search?",
    "What are the four caching layers in a RAG pipeline?",
    "When should I use fine-tuning instead of RAG?",
    "Why does chunking strategy matter for retrieval quality?",
]

if "history" not in st.session_state:
    st.session_state.history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = {}


def ensure_index(strategy: str):
    if strategy not in st.session_state.indexed:
        with st.spinner(f"Indexing docs with '{strategy}' chunking..."):
            n = build_index(strategy)
            st.session_state.indexed[strategy] = n
    return st.session_state.indexed[strategy]


st.title("🔎 RAG Observability Demo")
st.caption(
    "Same 5 docs about RAG, indexed 3 ways. Ask a question and watch chunking, "
    "caching, and latency play out live."
)

with st.sidebar:
    st.header("Settings")
    compare_mode = st.checkbox("Compare all 3 chunking strategies", value=True)
    strategy = None
    if not compare_mode:
        strategy = st.radio("Chunking strategy", STRATEGIES, index=0)

    st.divider()
    st.subheader("Cache stats")
    if st.button("Clear all caches"):
        embeddings.clear_cache()
        llm.clear_cache()
        st.session_state.history = []
        st.success("Caches cleared.")
    cache_stats_placeholder = st.container()

st.subheader("Ask a question")
cols = st.columns(len(SAMPLE_QUESTIONS))
for col, q in zip(cols, SAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        st.session_state["question_input"] = q

question = st.text_input("Or type your own question", key="question_input")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    strategies_to_run = STRATEGIES if compare_mode else [strategy]
    results = []
    for s in strategies_to_run:
        ensure_index(s)
        results.append(query(s, question))

    for r in results:
        st.session_state.history.append(
            {
                "question": r["question"],
                "strategy": r["strategy"],
                "embed_hit": r["embed_cache_hit"],
                "llm_hit": r["llm_cache_hit"],
                **r["timings"],
            }
        )

    st.divider()
    result_cols = st.columns(len(results))
    for col, r in zip(result_cols, results):
        with col:
            st.markdown(f"### Strategy: `{r['strategy']}`")
            st.info(r["answer"])
            t = r["timings"]
            badge = lambda hit: "🟢 cached" if hit else "🔴 computed"
            st.write(
                f"embed query: **{t['embed_query_ms']:.0f} ms** ({badge(r['embed_cache_hit'])})  \n"
                f"retrieval: **{t['retrieval_ms']:.0f} ms**  \n"
                f"generation: **{t['generation_ms']:.0f} ms** ({badge(r['llm_cache_hit'])})  \n"
                f"**total: {t['total_ms']:.0f} ms**"
            )
            with st.expander(f"Retrieved chunks ({len(r['chunks'])})"):
                for c in r["chunks"]:
                    st.caption(f"source: {c['source']}  ·  distance: {c['distance']:.3f}")
                    st.write(c["text"])
                    st.markdown("---")

if st.session_state.history:
    st.divider()
    st.subheader("Query history & latency")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    st.caption(
        "Tip: ask the same question twice. Watch generation_ms collapse once the "
        "LLM response cache kicks in."
    )
    st.bar_chart(df[["embed_query_ms", "retrieval_ms", "generation_ms"]])

with cache_stats_placeholder:
    e_stats = embeddings.cache_stats()
    l_stats = llm.cache_stats()
    st.metric("Embedding cache hits", e_stats["hits"])
    st.metric("Embedding cache misses", e_stats["misses"])
    st.metric("LLM response cache hits", l_stats["hits"])
    st.metric("LLM response cache misses", l_stats["misses"])
