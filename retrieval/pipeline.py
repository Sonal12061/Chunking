import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    ChromaDB-backed retrieval pipeline.

    Stores chunks from each strategy in separate collections
    so you can compare retrieval quality across strategies
    using the same queries.

    Analogy: Four separate filing cabinets (one per strategy)
    organised by the same librarian (embedding model). You ask
    the same question to all four and compare which cabinet
    returns the most relevant files.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        retrieval_cfg = config.get("retrieval", {})
        self.db_path = retrieval_cfg.get("db_path", "logs/chromadb")
        self.top_k = retrieval_cfg.get("top_k", 5)
        model_name = retrieval_cfg.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.model = SentenceTransformer(model_name)
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collections: Dict[str, Any] = {}

    def _get_or_create_collection(self, strategy: str):
        """Get existing or create new ChromaDB collection per strategy."""
        name = f"chunks_{strategy.replace(':', '_')}"
        try:
            collection = self.client.get_collection(name)
            logger.info(f"Loaded existing collection: {name}")
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Created new collection: {name}")
        self.collections[strategy] = collection
        return collection

    def index(
        self,
        chunks: List[Dict],
        strategy: str,
        batch_size: int = 100,
    ) -> None:
        """
        Embed and store chunks in ChromaDB collection.

        For parent-child strategy, only indexes child chunks
        since children are the retrieval units.

        Args:
            chunks:     List of chunk dicts
            strategy:   Strategy name — used as collection name
            batch_size: Number of chunks to embed per batch
        """
        collection = self._get_or_create_collection(strategy)

        # For parent-child, index only children
        if strategy == "parent_child":
            index_chunks = [
                c for c in chunks
                if c.get("strategy") == "parent_child:child"
            ]
        else:
            index_chunks = chunks

        if not index_chunks:
            logger.warning(f"No chunks to index for {strategy}")
            return

        logger.info(
            f"Indexing {len(index_chunks)} chunks for '{strategy}'..."
        )

        for i in range(0, len(index_chunks), batch_size):
            batch = index_chunks[i: i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            ids = [
                f"{strategy}_{i + j}"
                for j, _ in enumerate(batch)
            ]
            metadatas = [
                {
                    "title": c.get("title", ""),
                    "strategy": c.get("strategy", strategy),
                    "chunk_index": str(c.get("chunk_index", 0)),
                    "parent_index": str(c.get("parent_index", -1)),
                    "char_count": str(c.get("char_count", 0)),
                }
                for c in batch
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        logger.info(f"Indexed {len(index_chunks)} chunks for '{strategy}'")

    def retrieve(
        self,
        query: str,
        strategy: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve top-k chunks for a query from a strategy's collection.

        Args:
            query:    Natural language query
            strategy: Which collection to search
            top_k:    Number of results (defaults to config value)

        Returns:
            List of result dicts with text, score, and metadata
        """
        k = top_k or self.top_k
        collection = self.collections.get(strategy)
        if not collection:
            collection = self._get_or_create_collection(strategy)

        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, collection.count()),
        )

        retrieved = []
        if results and results["documents"]:
            for j, doc in enumerate(results["documents"][0]):
                retrieved.append({
                    "text": doc,
                    "score": round(
                        1 - results["distances"][0][j], 4
                    ),
                    "metadata": results["metadatas"][0][j],
                    "strategy": strategy,
                })

        return retrieved

    def retrieve_all_strategies(
        self, query: str, top_k: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Run the same query against all indexed strategies.
        Used by the dashboard for side-by-side comparison.
        """
        return {
            strategy: self.retrieve(query, strategy, top_k)
            for strategy in self.collections
        }

    def reset_collection(self, strategy: str) -> None:
        """Delete and recreate a collection — useful for re-indexing."""
        name = f"chunks_{strategy.replace(':', '_')}"
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception:
            pass
        self._get_or_create_collection(strategy)