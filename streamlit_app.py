import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os

st.set_page_config(
    page_title="Chunking Strategy Comparison",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_PATHS = {
    "eval": os.path.join(BASE_DIR, "logs", "eval_results.json"),
    "retrieval": os.path.join(BASE_DIR, "logs", "retrieval_results.json"),
    "chunks": os.path.join(BASE_DIR, "logs", "chunks_summary.json"),
}

STRATEGY_COLORS = {
    "fixed": "#378ADD",
    "recursive": "#1D9E75",
    "semantic": "#D85A30",
    "parent_child": "#7F77DD",
}

STRATEGY_LABELS = {
    "fixed": "Fixed",
    "recursive": "Recursive",
    "semantic": "Semantic",
    "parent_child": "Parent-Child",
}


def load_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Sidebar
st.sidebar.title("📚 Chunking Strategies")
st.sidebar.markdown("**Dataset:** Wikipedia articles")
st.sidebar.markdown("**Articles:** AI · WW2 · Climate · Python · Renaissance")
st.sidebar.divider()
st.sidebar.markdown("**Strategies**")
st.sidebar.markdown("🔵 Fixed — split by character count")
st.sidebar.markdown("🟢 Recursive — split by separators")
st.sidebar.markdown("🟠 Semantic — split by topic shift")
st.sidebar.markdown("🟣 Parent-Child — two granularities")
st.sidebar.divider()
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# Header
st.title("📚 Chunking Strategy Comparison Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Load data
eval_results = load_json(LOG_PATHS["eval"])
retrieval_results = load_json(LOG_PATHS["retrieval"])
chunks_summary = load_json(LOG_PATHS["chunks"])

if not eval_results:
    st.warning(
        "No evaluation results found. Run `python run_comparison.py` first."
    )
    st.stop()

strategies = [k for k in eval_results if not k.startswith("_")]
winners = eval_results.get("_winners", {})

# ── KPI Row ────────────────────────────────────────────────────────────
st.subheader("📊 Overview")
cols = st.columns(4)
for i, strategy in enumerate(strategies):
    r = eval_results[strategy]
    cols[i].metric(
        label=STRATEGY_LABELS.get(strategy, strategy),
        value=f"{r['chunk_count']:,} chunks",
        delta=f"avg {r['size_stats']['mean']:.0f} chars",
    )

st.divider()

# ── Coherence + Boundary Bar Charts ───────────────────────────────────
st.subheader("🎯 Chunk Quality Metrics")
col_left, col_right = st.columns(2)

with col_left:
    coherence_data = {
        "Strategy": [STRATEGY_LABELS.get(s, s) for s in strategies],
        "Coherence": [
            eval_results[s]["semantic_coherence"]["mean_coherence"]
            for s in strategies
        ],
    }
    fig1 = px.bar(
    coherence_data,
    x="Strategy",
    y="Coherence",
    title="Semantic coherence (higher = better)",
    )
    fig1.update_layout(showlegend=False, yaxis_range=[0, 1])
    fig1.add_hline(y=0.7, line_dash="dot", line_color="gray",
                annotation_text="0.7 threshold")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    boundary_data = {
        "Strategy": [STRATEGY_LABELS.get(s, s) for s in strategies],
        "Boundary Score": [
            eval_results[s]["boundary_quality"]["boundary_score"]
            for s in strategies
        ],
    }
    fig2 = px.bar(
    boundary_data,
    x="Strategy",
    y="Boundary Score",
    title="Boundary quality (higher = more natural cuts)",
    )
    fig2.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Chunk Size Distribution ────────────────────────────────────────────
st.subheader("📏 Chunk Size Distribution")

size_records = []
for strategy in strategies:
    r = eval_results[strategy]
    s = r["size_stats"]
    size_records.append({
        "Strategy": STRATEGY_LABELS.get(strategy, strategy),
        "Mean": s["mean"],
        "Std": s["std"],
        "Min": s["min"],
        "Max": s["max"],
        "Total Chunks": r["chunk_count"],
    })

df_sizes = pd.DataFrame(size_records)

fig3 = go.Figure()
for _, row in df_sizes.iterrows():
    strategy_key = next(
        (k for k, v in STRATEGY_LABELS.items() if v == row["Strategy"]),
        row["Strategy"]
    )
    fig3.add_trace(go.Bar(
        name=row["Strategy"],
        x=[row["Strategy"]],
        y=[row["Mean"]],
        error_y=dict(type="data", array=[row["Std"]], visible=True),
        marker_color=STRATEGY_COLORS.get(strategy_key, "#888"),
    ))

fig3.update_layout(
    title="Mean chunk size ± std (chars)",
    showlegend=False,
    yaxis_title="Characters",
)
st.plotly_chart(fig3, use_container_width=True)

st.dataframe(df_sizes, use_container_width=True, hide_index=True)

st.divider()

# ── Winners Table ──────────────────────────────────────────────────────
st.subheader("🏆 Winners by Metric")

winner_rows = [
    {"Metric": "Best semantic coherence",
     "Winner": STRATEGY_LABELS.get(winners.get("semantic_coherence", ""), "—")},
    {"Metric": "Best boundary quality",
     "Winner": STRATEGY_LABELS.get(winners.get("boundary_quality", ""), "—")},
    {"Metric": "Lowest overlap ratio",
     "Winner": STRATEGY_LABELS.get(winners.get("lowest_overlap", ""), "—")},
    {"Metric": "Most consistent size",
     "Winner": STRATEGY_LABELS.get(winners.get("most_consistent_size", ""), "—")},
]
st.dataframe(
    pd.DataFrame(winner_rows),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Retrieval Results ──────────────────────────────────────────────────
st.subheader("🔍 Retrieval Comparison")

if retrieval_results:
    query = st.selectbox(
        "Select a query to compare retrieval results:",
        options=list(retrieval_results.keys()),
    )

    if query:
        cols = st.columns(len(strategies))
        for i, strategy in enumerate(strategies):
            with cols[i]:
                st.markdown(
                    f"**{STRATEGY_LABELS.get(strategy, strategy)}**"
                )
                hits = retrieval_results.get(query, {}).get(strategy, [])
                for j, hit in enumerate(hits[:3]):
                    score = hit.get("score", 0)
                    text = hit.get("text", "")[:200]
                    st.markdown(
                        f"**#{j+1}** Score: `{score:.4f}`\n\n{text}..."
                    )
                    st.divider()
else:
    st.info("No retrieval results found.")

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────
st.caption(
    "Chunking Strategies RAG | Wikipedia dataset | "
    "github.com/Sonal12061/chunking-strategies-rag | "
    "huggingface.co/spaces/Sonal1288/chunking"
)