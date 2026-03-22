import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChunkEvaluator:
    """
    Evaluates chunk quality across four dimensions:

    1. Size stats       — mean, std, min, max chunk sizes
    2. Boundary quality — does chunk start/end at natural boundary?
    3. Semantic coherence — cosine similarity within chunk
    4. Overlap ratio    — how much content is duplicated

    Analogy: Like a quality inspector on a production line —
    checks every batch (chunking strategy) against the same
    four standards so you can compare them objectively.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_path = config.get("evaluation", {}).get(
            "results_path", "logs/eval_results.json"
        )
        cfg = config.get("chunking", {}).get("semantic", {})
        model_name = cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------ #
    #  Individual metrics                                                   #
    # ------------------------------------------------------------------ #

    def size_stats(self, chunks: List[Dict]) -> Dict[str, float]:
        """Mean, std, min, max of chunk character counts."""
        sizes = [c["char_count"] for c in chunks]
        return {
            "mean": round(float(np.mean(sizes)), 2),
            "std": round(float(np.std(sizes)), 2),
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
            "total_chunks": len(chunks),
        }

    def boundary_quality(self, chunks: List[Dict]) -> Dict[str, float]:
        """
        Score how naturally each chunk starts and ends.

        Good boundaries: chunk starts with capital letter,
        ends with punctuation — signs of complete sentences/paragraphs.
        Bad boundaries: chunk starts mid-word or ends mid-sentence.
        """
        good_starts = 0
        good_ends = 0

        for chunk in chunks:
            text = chunk["text"].strip()
            if not text:
                continue
            if text[0].isupper():
                good_starts += 1
            if text[-1] in ".!?\n":
                good_ends += 1

        n = len(chunks)
        return {
            "good_start_ratio": round(good_starts / n, 4) if n else 0,
            "good_end_ratio": round(good_ends / n, 4) if n else 0,
            "boundary_score": round(
                (good_starts + good_ends) / (2 * n), 4
            ) if n else 0,
        }

    def semantic_coherence(self, chunks: List[Dict]) -> Dict[str, float]:
        """
        Average intra-chunk cosine similarity.

        For each chunk: embed all sentences, compute pairwise
        similarity between consecutive sentences, average them.
        High score = sentences within chunk are topically related.
        Low score = chunk contains mixed topics (poor boundary).

        Only computed on sample of 50 chunks for speed.
        """
        sample = chunks[:50]
        coherence_scores = []

        for chunk in sample:
            text = chunk.get("text", "")
            sentences = [
                s.strip()
                for s in text.replace("\n", " ").split(". ")
                if s.strip()
            ]
            if len(sentences) < 2:
                coherence_scores.append(1.0)
                continue

            embeddings = self.model.encode(
                sentences,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sims = []
            for i in range(len(embeddings) - 1):
                a, b = embeddings[i], embeddings[i + 1]
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                if norm > 0:
                    sims.append(float(np.dot(a, b) / norm))

            if sims:
                coherence_scores.append(float(np.mean(sims)))

        return {
            "mean_coherence": round(float(np.mean(coherence_scores)), 4),
            "std_coherence": round(float(np.std(coherence_scores)), 4),
            "min_coherence": round(float(np.min(coherence_scores)), 4),
        }

    def overlap_ratio(self, chunks: List[Dict]) -> Dict[str, float]:
        """
        Estimate how much content is duplicated across chunks.

        Computes total chars across all chunks vs unique chars
        (approximated by total text length with duplicates removed).
        High overlap = more redundant content stored in vector DB.
        """
        all_text = " ".join(c["text"] for c in chunks)
        total_chars = sum(c["char_count"] for c in chunks)
        unique_chars = len(set(all_text))

        ratio = (total_chars - unique_chars) / total_chars if total_chars else 0
        return {
            "total_chars": total_chars,
            "overlap_ratio": round(max(0.0, ratio), 4),
        }

    # ------------------------------------------------------------------ #
    #  Main evaluation method                                               #
    # ------------------------------------------------------------------ #

    def evaluate(
        self, chunks_by_strategy: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Run all four metrics on each chunking strategy.

        Args:
            chunks_by_strategy: dict mapping strategy name to chunk list
                e.g. {"fixed": [...], "recursive": [...], "semantic": [...]}

        Returns:
            Nested dict of results per strategy + overall winner per metric
        """
        results = {}

        for strategy, chunks in chunks_by_strategy.items():
            logger.info(
                f"Evaluating {strategy}: {len(chunks)} chunks..."
            )

            # Skip parent chunks for parent_child — evaluate children only
            if strategy == "parent_child":
                eval_chunks = [
                    c for c in chunks
                    if c.get("strategy") == "parent_child:child"
                ]
            else:
                eval_chunks = chunks

            if not eval_chunks:
                continue

            results[strategy] = {
                "chunk_count": len(eval_chunks),
                "size_stats": self.size_stats(eval_chunks),
                "boundary_quality": self.boundary_quality(eval_chunks),
                "semantic_coherence": self.semantic_coherence(eval_chunks),
                "overlap_ratio": self.overlap_ratio(eval_chunks),
            }

        # Determine winner per metric
        results["_winners"] = self._compute_winners(results)
        self._save_results(results)
        return results

    def _compute_winners(
        self, results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Pick the best strategy for each metric."""
        strategies = [k for k in results if not k.startswith("_")]
        winners = {}

        # Best boundary score
        winners["boundary_quality"] = max(
            strategies,
            key=lambda s: results[s]["boundary_quality"]["boundary_score"],
        )

        # Best semantic coherence
        winners["semantic_coherence"] = max(
            strategies,
            key=lambda s: results[s]["semantic_coherence"]["mean_coherence"],
        )

        # Lowest overlap ratio
        winners["lowest_overlap"] = min(
            strategies,
            key=lambda s: results[s]["overlap_ratio"]["overlap_ratio"],
        )

        # Most consistent chunk sizes (lowest std)
        winners["most_consistent_size"] = min(
            strategies,
            key=lambda s: results[s]["size_stats"]["std"],
        )

        return winners

    def _save_results(self, results: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {self.results_path}")