#!/usr/bin/env python3
"""
Chunking Strategy Comparison Pipeline
======================================
Runs all four chunking strategies on Wikipedia articles,
evaluates chunk quality, indexes into ChromaDB, and saves
results for the Streamlit dashboard.

Usage:
    python run_comparison.py
    python run_comparison.py --config config.yaml
    python run_comparison.py --reset   # clears ChromaDB and re-indexes
"""

import argparse
import json
import logging
import os

import yaml

from chunking import (
    FixedChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
)
from evaluation import ChunkEvaluator
from retrieval import RetrievalPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_comparison")

os.makedirs("logs", exist_ok=True)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_articles(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_summary(
    chunks_by_strategy: dict,
    eval_results: dict,
) -> None:
    print("\n" + "=" * 65)
    print("  CHUNKING COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Strategy':<20} {'Chunks':>8} {'Avg Size':>10} {'Coherence':>12} {'Boundary':>10}")
    print(f"  {'─' * 61}")

    strategies = [k for k in eval_results if not k.startswith("_")]
    for strategy in strategies:
        r = eval_results[strategy]
        print(
            f"  {strategy:<20}"
            f"  {r['chunk_count']:>8,}"
            f"  {r['size_stats']['mean']:>8.0f}"
            f"  {r['semantic_coherence']['mean_coherence']:>12.4f}"
            f"  {r['boundary_quality']['boundary_score']:>10.4f}"
        )

    print(f"  {'─' * 61}")
    winners = eval_results.get("_winners", {})
    print(f"\n  Winners:")
    print(f"  Best boundary quality:   {winners.get('boundary_quality', '—')}")
    print(f"  Best semantic coherence: {winners.get('semantic_coherence', '—')}")
    print(f"  Lowest overlap:          {winners.get('lowest_overlap', '—')}")
    print(f"  Most consistent size:    {winners.get('most_consistent_size', '—')}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run chunking strategy comparison"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset ChromaDB collections and re-index",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    articles = load_articles(config["data"]["raw_path"])
    logger.info(f"Loaded {len(articles)} articles")

    # ── 1. Chunk with all strategies ────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 1: Chunking")

    logger.info("Fixed chunking...")
    fixed_chunks = FixedChunker(config).chunk_articles(articles)
    logger.info(f"  Fixed: {len(fixed_chunks)} chunks")

    logger.info("Recursive chunking...")
    recursive_chunks = RecursiveChunker(config).chunk_articles(articles)
    logger.info(f"  Recursive: {len(recursive_chunks)} chunks")

    logger.info("Semantic chunking...")
    semantic_chunks = SemanticChunker(config).chunk_articles(articles)
    logger.info(f"  Semantic: {len(semantic_chunks)} chunks")

    logger.info("Parent-child chunking...")
    pc_chunks = ParentChildChunker(config).chunk_articles(articles)
    children = [c for c in pc_chunks if c["strategy"] == "parent_child:child"]
    logger.info(f"  Parent-child: {len(pc_chunks)} total ({len(children)} children)")

    chunks_by_strategy = {
        "fixed": fixed_chunks,
        "recursive": recursive_chunks,
        "semantic": semantic_chunks,
        "parent_child": pc_chunks,
    }

    # Save chunks to logs for dashboard
    chunks_summary = {
        strategy: [
            {k: v for k, v in c.items() if k != "text"}
            for c in chunks[:20]
        ]
        for strategy, chunks in chunks_by_strategy.items()
    }
    with open("logs/chunks_summary.json", "w") as f:
        json.dump(chunks_summary, f, indent=2, default=str)

    # ── 2. Evaluate chunk quality ────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 2: Evaluation")
    evaluator = ChunkEvaluator(config)
    eval_results = evaluator.evaluate(chunks_by_strategy)

    # ── 3. Index into ChromaDB ───────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 3: Indexing into ChromaDB")
    pipeline = RetrievalPipeline(config)

    if args.reset:
        for strategy in chunks_by_strategy:
            pipeline.reset_collection(strategy)
        logger.info("Collections reset.")

    for strategy, chunks in chunks_by_strategy.items():
        pipeline.index(chunks, strategy)

    # ── 4. Test retrieval with sample queries ────────────────────────────
    logger.info("─" * 50)
    logger.info("STEP 4: Sample Retrieval Test")

    test_queries = [
        "What are the main applications of artificial intelligence?",
        "What caused World War II?",
        "How does climate change affect sea levels?",
    ]

    retrieval_results = {}
    for query in test_queries:
        logger.info(f"  Query: {query[:60]}...")
        results = pipeline.retrieve_all_strategies(query)
        retrieval_results[query] = {
            strategy: [
                {"score": r["score"], "text": r["text"][:150]}
                for r in hits
            ]
            for strategy, hits in results.items()
        }

    with open("logs/retrieval_results.json", "w") as f:
        json.dump(retrieval_results, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    print_summary(chunks_by_strategy, eval_results)
    logger.info("Run complete. Launch dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()