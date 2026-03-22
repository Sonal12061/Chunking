from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    """
    Semantic chunking — splits text at points where the meaning
    shifts by embedding sentences and detecting cosine similarity drops.

    Analogy: A professor reading a textbook who highlights every point
    where the topic changes — not based on word count but based on
    whether the next sentence is still talking about the same idea.
    When similarity between consecutive sentences drops below a
    threshold, a new chunk begins.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("chunking", {}).get("semantic", {})
        self.model_name = cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.breakpoint_threshold = cfg.get("breakpoint_threshold", 0.7)
        self.min_chunk_size = cfg.get("min_chunk_size", 100)
        self.max_chunk_size = cfg.get("max_chunk_size", 1000)
        self.model = SentenceTransformer(self.model_name)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences on . ! ? boundaries."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _find_breakpoints(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[int]:
        """
        Find sentence indices where a topic shift occurs.

        Computes cosine similarity between each consecutive pair of
        sentence embeddings. When similarity drops below the threshold,
        marks that index as a breakpoint — a new chunk starts there.
        """
        breakpoints = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.breakpoint_threshold:
                breakpoints.append(i)
        return breakpoints

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text at semantic breakpoints.

        Steps:
          1. Split text into sentences
          2. Embed all sentences in one batch (fast)
          3. Find consecutive pairs where similarity < threshold
          4. Cut at those points
          5. Merge tiny chunks, split oversized chunks

        Args:
            text:     Raw article text
            metadata: Optional dict attached to each chunk

        Returns:
            List of chunk dicts with text, similarity scores, and stats
        """
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # Embed all sentences in one batch
        embeddings = self.model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        breakpoints = self._find_breakpoints(sentences, embeddings)

        # Build raw chunks from breakpoints
        raw_chunks = []
        start = 0
        for bp in breakpoints:
            raw_chunks.append(sentences[start:bp])
            start = bp
        raw_chunks.append(sentences[start:])

        # Merge tiny chunks and split oversized ones
        final_chunks = []
        buffer = ""
        for raw in raw_chunks:
            chunk_text = " ".join(raw)

            if len(buffer) + len(chunk_text) < self.min_chunk_size:
                buffer += " " + chunk_text
                continue

            if buffer.strip():
                final_chunks.append(buffer.strip())
            buffer = chunk_text

        if buffer.strip():
            final_chunks.append(buffer.strip())

        # Split any chunks that exceed max_chunk_size
        checked_chunks = []
        for chunk_text in final_chunks:
            if len(chunk_text) > self.max_chunk_size:
                mid = len(chunk_text) // 2
                checked_chunks.append(chunk_text[:mid].strip())
                checked_chunks.append(chunk_text[mid:].strip())
            else:
                checked_chunks.append(chunk_text)

        # Compute per-chunk similarity score
        chunks = []
        for i, chunk_text in enumerate(checked_chunks):
            if not chunk_text.strip():
                continue

            chunk_sentences = self._split_sentences(chunk_text)
            if len(chunk_sentences) > 1:
                chunk_embeddings = self.model.encode(
                    chunk_sentences,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                sims = [
                    self._cosine_similarity(
                        chunk_embeddings[j], chunk_embeddings[j + 1]
                    )
                    for j in range(len(chunk_embeddings) - 1)
                ]
                avg_sim = float(np.mean(sims)) if sims else 1.0
            else:
                avg_sim = 1.0

            chunks.append({
                "text": chunk_text,
                "chunk_index": i,
                "strategy": "semantic",
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "avg_internal_similarity": round(avg_sim, 4),
                **metadata,
            })

        return chunks

    def chunk_articles(self, articles: List[Dict]) -> List[Dict]:
        """Chunk a list of article dicts."""
        all_chunks = []
        for article in articles:
            print(f"  Semantic chunking: {article.get('title', '')}...")
            metadata = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
            }
            chunks = self.chunk(article["content"], metadata)
            all_chunks.extend(chunks)
        return all_chunks