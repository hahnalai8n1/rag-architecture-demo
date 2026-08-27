import pandas as pd
import streamlit as st

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

tabs = st.tabs(
    [
        "1️⃣ Why RAG",
        "2️⃣ Architecture",
        "3️⃣ Chunking",
        "4️⃣ Caching",
        "5️⃣ Metrics",
        "6️⃣ RAG vs FT vs PE",
        "🔎 Live Demo",
    ]
)

with tabs[0]:
    st.header("Why RAG?")
    st.markdown(
        """
LLMs have two structural problems:

- **They hallucinate** — when they don't know something, they still generate a confident-sounding answer.
- **Their knowledge has a cutoff date** — and they know nothing about your private, internal, or newly-created data.

**Retrieval-Augmented Generation (RAG)** fixes both by looking things up in an external
knowledge source *before* generating an answer, instead of relying only on what the
model memorized during training.
"""
    )

with tabs[1]:
    st.header("Two Pipelines, One System")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📥 Indexing (offline)")
        st.markdown("Runs once, re-runs when documents change.")
        steps = ["Documents", "Chunk", "Embed", "Vector DB"]
        cols = st.columns(len(steps))
        for col, step in zip(cols, steps):
            col.info(step)
    with c2:
        st.subheader("🔄 Query (online)")
        st.markdown("Runs on every single user question.")
        steps = ["Question", "Embed", "Retrieve", "Augment prompt", "Generate"]
        cols = st.columns(len(steps))
        for col, step in zip(cols, steps):
            col.success(step)
    st.markdown(
        "The retriever finds relevant chunks; the generator writes the answer "
        "*grounded* in those chunks instead of from memory alone. → **Try both live in the Demo tab.**"
    )

with tabs[2]:
    st.header("Chunking Strategies")
    st.markdown(
        "Before anything gets embedded, documents are split into chunks. "
        "*How* you split them has a bigger effect on retrieval quality than "
        "which embedding model you pick."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Fixed-size")
        st.caption("Every N characters, with a small overlap.")
        st.markdown("✅ Simple, predictable size\n\n⚠️ Can cut a sentence in half")
    with c2:
        st.markdown("#### Sentence-based")
        st.caption("Groups whole sentences up to a size limit.")
        st.markdown(
            "✅ Never cuts mid-sentence\n\n⚠️ Ignores paragraph/section "
            "breaks — unrelated sentences can end up in the same chunk"
        )
    with c3:
        st.markdown("#### Paragraph-based")
        st.caption("Follows the document's own structure.")
        st.markdown("✅ Most semantically coherent\n\n⚠️ Chunk sizes get uneven")
    st.info(
        "→ In the **Live Demo** tab, ask the same question with all 3 strategies and "
        "open **Retrieved chunks** — you'll see sentence-based chunking sometimes bundle "
        "a section heading and an unrelated intro sentence together with the answer, "
        "exactly because it doesn't know about paragraph boundaries.",
        icon="💡",
    )

with tabs[3]:
    st.header("Caching Layers")
    st.markdown(
        "Every question triggers at least one embedding call and one LLM call. "
        "A mature RAG pipeline caches four things:"
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, title, desc, implemented in [
        (c1, "Query cache", "Final answer for an exact/near-duplicate question — skips the whole pipeline.", False),
        (c2, "Embedding cache", "Vector for a piece of text, keyed by content hash.", True),
        (c3, "Vector search cache", "Nearest-neighbor result for a repeated query.", False),
        (c4, "LLM response cache", "Generated answer for a given question + context.", True),
    ]:
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.markdown("🟢 **implemented in this demo**" if implemented else "⚪ conceptual only here")
    st.info(
        "→ In the **Live Demo** tab: ask the same question twice. The second time, "
        "`generation_ms` collapses from ~1–3 seconds to ~1 millisecond because the "
        "embedding cache and LLM response cache both hit.",
        icon="💡",
    )

with tabs[4]:
    st.header("Metrics to Monitor")
    metrics = [
        ("Response time", "Total latency, broken down by stage: embed / retrieve / generate."),
        ("Throughput", "Questions served per second — usually bottlenecked by the LLM call."),
        ("Error rate", "How often a stage fails outright (timeout, empty result, API error)."),
        ("Retrieval quality", "Do the retrieved chunks actually contain the answer? (precision@k, recall@k, MRR)"),
        ("Embedding performance", "Latency of the embedding model, and how well it separates relevant from irrelevant text."),
        ("Chunking efficiency", "Whether the chunking strategy is helping or hurting retrieval quality."),
    ]
    for name, desc in metrics:
        st.markdown(f"**{name}** — {desc}")
    st.info(
        "→ The **Live Demo** tab's latency table and chart give you the first three "
        "(response time breakdown, and cache hit/miss as a proxy for error-free operation) "
        "for real, measured with a wall clock — not simulated.",
        icon="💡",
    )

with tabs[5]:
    st.header("RAG vs. Fine-Tuning vs. Prompt Engineering")
    comparison = pd.DataFrame(
        {
            "": ["Setup speed", "Knowledge freshness", "Cost", "Best for"],
            "Prompt Engineering": ["Fastest", "Whatever the model already knows", "Lowest", "Quick prototyping"],
            "Fine-tuning": ["Slowest (needs labeled data)", "Frozen at training time", "Highest (compute + data)", "Changing behavior/tone/format"],
            "RAG": ["Medium (needs a pipeline)", "Updated instantly by editing docs", "Medium (infra to maintain)", "Fresh or private knowledge"],
        }
    ).set_index("")
    st.table(comparison)
    st.markdown(
        "**Takeaway:** these aren't mutually exclusive. The most capable production "
        "systems combine all three — RAG for facts, fine-tuning for behavior, "
        "prompt engineering for interaction style."
    )

with tabs[6]:
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
        st.caption(
            "👉 Watch the **generation_ms** and **total_ms** columns (and the chart below): "
            "ask the same question twice and the second row collapses to almost 0 once "
            "`llm_hit` / `embed_hit` turn True."
        )
        df = pd.DataFrame(st.session_state.history)

        def highlight_cache_hits(row):
            color = "background-color: rgba(46, 204, 113, 0.25)" if row["llm_hit"] else ""
            return [color] * len(row)

        st.dataframe(
            df.style.apply(highlight_cache_hits, axis=1).format(
                {c: "{:.0f}" for c in ["embed_query_ms", "retrieval_ms", "generation_ms", "total_ms"]}
            ),
            use_container_width=True,
        )
        st.bar_chart(df[["embed_query_ms", "retrieval_ms", "generation_ms"]])

with cache_stats_placeholder:
    e_stats = embeddings.cache_stats()
    l_stats = llm.cache_stats()
    st.metric("Embedding cache hits", e_stats["hits"])
    st.metric("Embedding cache misses", e_stats["misses"])
    st.metric("LLM response cache hits", l_stats["hits"])
    st.metric("LLM response cache misses", l_stats["misses"])
