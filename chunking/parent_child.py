from typing import Any, Dict, List


class ParentChildChunker:
    """
    Parent-child chunking — stores two granularities of every chunk.

    Child chunks: small, used for embedding and retrieval
    Parent chunks: large, returned to LLM as context

    Analogy: A library index card (child) helps you find the right
    book fast. But when you sit down to read, you get the full
    chapter (parent) — not just the index card.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("chunking", {}).get("parent_child", {})
        self.parent_size = cfg.get("parent_size", 512)
        self.child_size = cfg.get("child_size", 128)
        self.overlap = cfg.get("overlap", 20)

    def _split_fixed(
        self, text: str, size: int, overlap: int
    ) -> List[str]:
        """Split text into fixed-size pieces with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += size - overlap
        return chunks

    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into parent chunks, then split each parent
        into child chunks. Each child stores a reference to
        its parent index.

        Returns both parent and child chunk dicts in one list.
        Strategy field distinguishes them:
          - "parent_child:parent"
          - "parent_child:child"
        """
        metadata = metadata or {}
        parent_texts = self._split_fixed(
            text, self.parent_size, self.overlap
        )

        all_chunks = []
        child_index = 0

        for parent_idx, parent_text in enumerate(parent_texts):

            # Store parent chunk
            all_chunks.append({
                "text": parent_text,
                "chunk_index": parent_idx,
                "strategy": "parent_child:parent",
                "char_count": len(parent_text),
                "word_count": len(parent_text.split()),
                "parent_index": parent_idx,
                "child_indices": [],
                **metadata,
            })

            # Split parent into children
            child_texts = self._split_fixed(
                parent_text, self.child_size, self.overlap
            )

            child_indices = []
            for child_text in child_texts:
                if child_text.strip():
                    all_chunks.append({
                        "text": child_text,
                        "chunk_index": child_index,
                        "strategy": "parent_child:child",
                        "char_count": len(child_text),
                        "word_count": len(child_text.split()),
                        "parent_index": parent_idx,
                        **metadata,
                    })
                    child_indices.append(child_index)
                    child_index += 1

            # Back-fill child indices onto parent
            all_chunks[parent_idx]["child_indices"] = child_indices

        return all_chunks

    def get_parents(self, chunks: List[Dict]) -> List[Dict]:
        """Filter only parent chunks."""
        return [
            c for c in chunks
            if c.get("strategy") == "parent_child:parent"
        ]

    def get_children(self, chunks: List[Dict]) -> List[Dict]:
        """Filter only child chunks."""
        return [
            c for c in chunks
            if c.get("strategy") == "parent_child:child"
        ]

    def get_parent_for_child(
        self, child: Dict, all_chunks: List[Dict]
    ) -> Dict:
        """
        Given a child chunk retrieved by vector search,
        fetch its parent chunk to pass to the LLM.
        This is the core parent-child retrieval pattern.
        """
        parent_idx = child.get("parent_index")
        parents = self.get_parents(all_chunks)
        for parent in parents:
            if parent["chunk_index"] == parent_idx:
                return parent
        return child  # fallback to child if parent not found

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