import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from rag import embeddings, llm
from rag.pipeline import build_index, query

st.set_page_config(page_title="RAG Architecture Demo", page_icon="🔎", layout="wide")

STRATEGIES = ["fixed", "sentence", "paragraph"]
SAMPLE_QUESTIONS = [
    "What is the difference between vector search and keyword search?",
    "What are the four caching layers in a RAG pipeline?",
    "When should I use fine-tuning instead of RAG?",
    "Why does chunking strategy matter for retrieval quality?",
]

ARCHITECTURE_SVG = """
<svg viewBox="0 0 1040 300" xmlns="http://www.w3.org/2000/svg"
     style="width:100%; height:auto; background:#ffffff; border-radius:12px; padding:10px;">
  <defs>
    <marker id="arrowGray" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#64748b"/>
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#a855f7"/>
    </marker>
  </defs>

  <text x="40" y="30" font-size="16" font-weight="600" fill="#1e3a8a">📥 Indexing (offline) — runs once, re-runs when docs change</text>
  <g font-family="sans-serif">
    <rect x="40"  y="45" width="180" height="60" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="130" y="80" font-size="15" fill="#1e3a8a" text-anchor="middle">Documents</text>
    <rect x="260" y="45" width="180" height="60" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="350" y="80" font-size="15" fill="#1e3a8a" text-anchor="middle">Chunk</text>
    <rect x="480" y="45" width="180" height="60" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="570" y="80" font-size="15" fill="#1e3a8a" text-anchor="middle">Embed</text>
    <rect x="700" y="45" width="180" height="60" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
    <text x="790" y="80" font-size="15" fill="#1e3a8a" text-anchor="middle">Vector DB</text>

    <line x1="220" y1="75" x2="255" y2="75" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)"/>
    <line x1="440" y1="75" x2="475" y2="75" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)"/>
    <line x1="660" y1="75" x2="695" y2="75" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)"/>
  </g>

  <text x="40" y="185" font-size="16" font-weight="600" fill="#14532d">🔄 Query (online) — runs on every single user question</text>
  <g font-family="sans-serif">
    <rect x="40"  y="200" width="150" height="60" rx="10" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
    <text x="115" y="235" font-size="14" fill="#14532d" text-anchor="middle">Question</text>
    <rect x="220" y="200" width="150" height="60" rx="10" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
    <text x="295" y="235" font-size="14" fill="#14532d" text-anchor="middle">Embed</text>
    <rect x="400" y="200" width="150" height="60" rx="10" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
    <text x="475" y="235" font-size="14" fill="#14532d" text-anchor="middle">Retrieve</text>
    <rect x="580" y="200" width="150" height="60" rx="10" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
    <text x="655" y="228" font-size="13" fill="#14532d" text-anchor="middle">Augment</text>
    <text x="655" y="244" font-size="13" fill="#14532d" text-anchor="middle">prompt</text>
    <rect x="760" y="200" width="150" height="60" rx="10" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
    <text x="835" y="235" font-size="14" fill="#14532d" text-anchor="middle">Generate</text>

    <line x1="190" y1="230" x2="215" y2="230" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)"/>
    <line x1="370" y1="230" x2="395" y2="230" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)"/>
    <line x1="550" y1="230" x2="575" y2="230" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)"/>
    <line x1="730" y1="230" x2="755" y2="230" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)"/>
  </g>

  <path d="M 790,105 L 790,150 L 475,150 L 475,195" fill="none"
        stroke="#a855f7" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#arrowPurple)"/>
  <text x="500" y="145" font-size="12.5" font-style="italic" fill="#7e22ce">indexed vectors get searched at query time</text>
</svg>
"""

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


st.title("🔎 RAG Architecture Demo")
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

with st.expander("🏗️ Architecture — two pipelines, one system", expanded=True):
    components.html(ARCHITECTURE_SVG, height=320, scrolling=False)

with st.expander("📊 RAG vs. Fine-Tuning vs. Prompt Engineering", expanded=True):
    comparison = pd.DataFrame(
        {
            "": ["Setup speed", "Knowledge freshness", "Cost", "Best for"],
            "Prompt Engineering": ["Fastest", "Whatever the model already knows", "Lowest", "Quick prototyping"],
            "Fine-tuning": ["Slowest (needs labeled data)", "Frozen at training time", "Highest (compute + data)", "Changing behavior/tone/format"],
            "RAG": ["Medium (needs a pipeline)", "Updated instantly by editing docs", "Medium (infra to maintain)", "Fresh or private knowledge"],
        }
    ).set_index("")
    st.table(comparison)
    st.caption(
        "These aren't mutually exclusive — the most capable production systems combine "
        "all three: RAG for facts, fine-tuning for behavior, prompt engineering for interaction style."
    )

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
