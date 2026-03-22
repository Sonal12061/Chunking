---
title: Chunking Strategies RAG
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: streamlit_app.py
pinned: true
---

# Chunking Strategies for RAG Pipelines

🚀 **[Live Dashboard](https://huggingface.co/spaces/Sonal1288/chunking)** — explore results without running any code.

A systematic comparison of four chunking strategies for Retrieval-Augmented Generation (RAG) pipelines, evaluated on 5 Wikipedia articles (339,712 chars · 52,212 words) across semantic coherence, boundary quality, chunk size consistency, and overlap ratio.

---

## Results

| Strategy | Chunks | Avg Size | Coherence | Boundary Score |
|---|---|---|---|---|
| Fixed | 738 | 510 chars | 0.3861 | 0.0400 |
| Recursive | 1,313 | 308 chars | 0.3658 | 0.2456 |
| Semantic | 2,238 | 151 chars | **0.9822** | **0.9484** |
| Parent-Child | 3,454 | 118 chars | 0.5234 | 0.0392 |

### Winners by metric

| Metric | Winner | Why |
|---|---|---|
| Best semantic coherence | **Semantic** | Splits at topic shifts — sentences within chunks are topically related |
| Best boundary quality | **Semantic** | 94.8% of chunks end at natural sentence/paragraph boundaries |
| Lowest overlap | **Semantic** | No fixed overlap window — cuts only where meaning shifts |
| Most consistent size | **Parent-Child** | Children are fixed-size by design |

### Key insight

Fixed chunking boundary score is only 0.04 — meaning 96% of chunks cut mid-sentence. Semantic chunking boundary score is 0.9484 — meaning 94.8% of chunks end at natural topic boundaries. For RAG pipelines where retrieval precision matters, semantic chunking produces dramatically more coherent context windows despite being 3x more expensive to compute.

---

## Strategies

### Fixed chunking
Splits every N characters with overlap. Fast and deterministic but completely ignores sentence and paragraph boundaries.
```python
chunker = FixedChunker(config)  # chunk_size=512, overlap=50
chunks = chunker.chunk_articles(articles)
```

### Recursive chunking
Tries separators in priority order: `\n\n` → `\n` → `. ` → ` `. Always prefers the most natural boundary available.
```python
chunker = RecursiveChunker(config)
chunks = chunker.chunk_articles(articles)
```

### Semantic chunking
Embeds every sentence using `all-MiniLM-L6-v2`. Cuts a new chunk when cosine similarity between consecutive sentences drops below threshold (0.7) — meaning the topic has shifted.
```python
chunker = SemanticChunker(config)  # breakpoint_threshold=0.7
chunks = chunker.chunk_articles(articles)
```

### Parent-Child chunking
Stores two granularities — large parent chunks (512 chars) for LLM context, small child chunks (128 chars) for retrieval. Child chunks match queries precisely; parent chunks give the LLM enough surrounding context to answer well.
```python
chunker = ParentChildChunker(config)
all_chunks = chunker.chunk_articles(articles)
children = chunker.get_children(all_chunks)   # used for retrieval
parents  = chunker.get_parents(all_chunks)    # returned to LLM
```

---

## Architecture
```
5 Wikipedia articles (339k chars)
        │
        ├──► Fixed chunker       (512 chars, 50 overlap)
        ├──► Recursive chunker   (\n\n → \n → . → space)
        ├──► Semantic chunker    (sentence embeddings + cosine threshold)
        └──► Parent-Child        (512 parent / 128 child)
                │
                ▼
        Chunk quality evaluator
        (coherence · boundary · size · overlap)
                │
                ▼
        ChromaDB (separate collection per strategy)
                │
                ▼
        Streamlit comparison dashboard
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Semantic coherence | Mean cosine similarity between consecutive sentences within a chunk. Higher = more topically consistent. |
| Boundary quality | Fraction of chunks starting with capital letter and ending with punctuation. Higher = more natural cuts. |
| Size consistency | Standard deviation of chunk sizes. Lower = more predictable retrieval units. |
| Overlap ratio | Fraction of duplicated content across chunks. Lower = less redundancy in vector DB. |

---

## Quick Start
```bash
# 1. Clone and setup
git clone https://github.com/Sonal12061/chunking-strategies-rag
cd chunking-strategies-rag
python -m venv venv
source venv/Scripts/activate    # Windows
source venv/bin/activate        # Mac/Linux
pip install -r requirements.txt

# 2. Fetch Wikipedia articles
python data/fetch_articles.py

# 3. Run full comparison pipeline
python run_comparison.py

# 4. Launch dashboard locally
streamlit run streamlit_app.py
```

---

## Project Structure
```
chunking-strategies-rag/
├── config.yaml              # all thresholds and hyperparameters
├── data/
│   └── fetch_articles.py    # Wikipedia article fetcher
├── chunking/
│   ├── fixed.py             # fixed size chunking
│   ├── recursive.py         # recursive separator chunking
│   ├── semantic.py          # embedding-based semantic chunking
│   └── parent_child.py      # parent-child retrieval pattern
├── evaluation/
│   └── evaluator.py         # coherence, boundary, size, overlap metrics
├── retrieval/
│   └── pipeline.py          # ChromaDB indexing + retrieval
├── logs/                    # precomputed evaluation results
├── streamlit_app.py         # Streamlit comparison dashboard
└── run_comparison.py        # main orchestration script
```

---

## Links

- 🚀 Live dashboard: [huggingface.co/spaces/Sonal1288/chunking](https://huggingface.co/spaces/Sonal1288/chunking)
- 💻 GitHub: [github.com/Sonal12061/chunking-strategies-rag](https://github.com/Sonal12061/chunking-strategies-rag)

---

## Tech Stack

`Python` · `HuggingFace` · `sentence-transformers` · `ChromaDB` · `Streamlit` · `Plotly` · `Wikipedia API`