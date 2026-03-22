from typing import Any, Dict, List


class FixedChunker:
    """
    Fixed size chunking — splits text every N characters with overlap.

    The simplest chunking strategy. Completely ignores text structure,
    sentence boundaries, and meaning.

    Analogy: Cutting a book into equal-sized pieces with scissors —
    fast and predictable but sentences and paragraphs get sliced
    in the middle with no regard for context.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("chunking", {}).get("fixed", {})
        self.chunk_size = cfg.get("chunk_size", 512)
        self.chunk_overlap = cfg.get("chunk_overlap", 50)

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text:     Raw article text
            metadata: Optional dict (title, url etc.) attached to each chunk

        Returns:
            List of chunk dicts with text, metadata, and stats
        """
        metadata = metadata or {}
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "strategy": "fixed",
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "start_char": start,
                    "end_char": min(end, len(text)),
                    **metadata,
                })
                chunk_index += 1

            # Move forward by chunk_size minus overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks

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