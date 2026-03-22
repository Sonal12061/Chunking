from typing import Any, Dict, List


class RecursiveChunker:
    """
    Recursive character chunking — tries separators in order of
    preference until chunks are small enough.

    Analogy: A smart editor who first tries to cut at chapter breaks,
    then paragraph breaks, then sentence breaks, then word breaks —
    always preferring the most natural boundary available rather than
    cutting mid-sentence like fixed chunking does.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("chunking", {}).get("recursive", {})
        self.chunk_size = cfg.get("chunk_size", 512)
        self.chunk_overlap = cfg.get("chunk_overlap", 50)
        self.separators = cfg.get(
            "separators", ["\n\n", "\n", ". ", " ", ""]
        )

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators in priority order.

        For each separator:
          1. Split the text on that separator
          2. If a piece is still too large, recurse with remaining separators
          3. If a piece is small enough, keep it
          4. Merge small adjacent pieces to avoid tiny fragments
        """
        if not separators:
            # Last resort — hard cut at chunk_size
            return [
                text[i: i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator) if separator else list(text)
        splits = [s for s in splits if s.strip()]

        chunks = []
        current = ""

        for split in splits:
            candidate = current + (separator if current else "") + split

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                # Save current accumulated chunk
                if current.strip():
                    if len(current) > self.chunk_size:
                        # Still too large — recurse with next separator
                        sub_chunks = self._split_text(current, remaining_separators)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current)

                # Start fresh with overlap
                if chunks and self.chunk_overlap > 0:
                    overlap_text = chunks[-1][-self.chunk_overlap:]
                    current = overlap_text + (separator if overlap_text else "") + split
                else:
                    current = split

        if current.strip():
            if len(current) > self.chunk_size:
                sub_chunks = self._split_text(current, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                chunks.append(current)

        return chunks

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text using recursive separator strategy.

        Args:
            text:     Raw article text
            metadata: Optional dict attached to each chunk

        Returns:
            List of chunk dicts with text, metadata, and stats
        """
        metadata = metadata or {}
        raw_chunks = self._split_text(text, self.separators)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "chunk_index": i,
                    "strategy": "recursive",
                    "char_count": len(chunk_text.strip()),
                    "word_count": len(chunk_text.strip().split()),
                    "separator_used": self._detect_separator(chunk_text),
                    **metadata,
                })

        return chunks

    def _detect_separator(self, text: str) -> str:
        """Identify which separator was likely used to end this chunk."""
        if text.endswith("\n\n"):
            return "paragraph"
        elif text.endswith("\n"):
            return "newline"
        elif text.endswith(". "):
            return "sentence"
        return "other"

    def chunk_articles(self, articles: List[Dict]) -> List[Dict]:
        """Chunk a list of article dicts."""
        all_chunks = []
        for article in articles:
            metadata = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
            }
            chunks = self.chunk(article["content"], metadata)
            all_chunks.extend(chunks)
        return all_chunks