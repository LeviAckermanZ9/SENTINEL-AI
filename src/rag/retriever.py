"""
SENTINEL-AI — Two-Stage Retriever
Covers: CSR322 CO5 (RAG system design)

Stage 1: Dense retrieval via SentenceTransformer + ChromaDB (top 20).
Stage 2: Cross-encoder reranking (top 5).
All models lazy-loaded.
"""

import os
from typing import Any, Dict, List, Optional


class SentinelRetriever:
    """
    Two-stage retriever: dense cosine search + cross-encoder reranking.

    Stage 1: Embed query with all-MiniLM-L6-v2, query ChromaDB for top 20.
    Stage 2: Rerank with cross-encoder/ms-marco-MiniLM-L-6-v2, return top 5.

    Usage:
        retriever = SentinelRetriever()
        results = retriever.get_relevant_evidence("5G causes COVID")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name: str = "sentinel_factcheck_kb",
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        persist_dir: str = "./data/chromadb",
    ):
        self._embedding_model_name = embedding_model
        self._reranker_model_name = reranker_model
        self._collection_name = collection_name
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        self._persist_dir = persist_dir

        self._embedder = None
        self._reranker = None
        self._collection = None

    def _load_embedder(self):
        """Lazy-load SentenceTransformer embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _load_reranker(self):
        """Lazy-load CrossEncoder reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self._reranker_model_name)
        return self._reranker

    def _load_collection(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            import chromadb

            host = self._chroma_host or os.environ.get("CHROMA_HOST")
            port = self._chroma_port or int(os.environ.get("CHROMA_PORT", "8000"))

            if host:
                token = os.environ.get("CHROMA_TOKEN", "")
                if token:
                    client = chromadb.HttpClient(
                        host=host,
                        port=port,
                        headers={"Authorization": f"Bearer {token}"},
                    )
                else:
                    client = chromadb.HttpClient(host=host, port=port)
            else:
                client = chromadb.PersistentClient(path=self._persist_dir)

            self._collection = client.get_collection(name=self._collection_name)
        return self._collection

    def set_collection(self, collection):
        """Set collection directly (for testing with EphemeralClient)."""
        self._collection = collection

    def _stage1_dense_retrieve(self, query: str, n_results: int = 20) -> Dict[str, Any]:
        """
        Stage 1: Dense retrieval via ChromaDB cosine search.

        Args:
            query: Search query text.
            n_results: Number of results to retrieve.

        Returns:
            Dict with documents, distances, metadatas, ids.
        """
        embedder = self._load_embedder()
        collection = self._load_collection()

        query_embedding = embedder.encode(query, convert_to_numpy=True).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
        )

        # Convert distances to similarities (cosine: similarity = 1 - distance)
        distances = results["distances"][0] if results["distances"] else []
        similarities = [1.0 - d for d in distances]

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "ids": results["ids"][0] if results["ids"] else [],
            "distances": distances,
            "similarities": similarities,
        }

    def _stage2_rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Cross-encoder reranking of retrieved documents.

        Args:
            query: Original search query.
            documents: List of retrieved document texts.
            top_k: Number of top results to return after reranking.

        Returns:
            List of dicts with document, rerank_score, original_index.
        """
        if not documents:
            return []

        reranker = self._load_reranker()

        # Score all query-document pairs
        pairs = [[query, doc] for doc in documents]
        scores = reranker.predict(pairs)

        # Sort by score descending
        scored = [
            {"document": doc, "rerank_score": float(score), "original_index": i}
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]

    def get_relevant_evidence(self, query: str) -> Dict[str, Any]:
        """
        Full two-stage retrieval pipeline.

        Stage 1: Dense cosine retrieval → top 20 candidates.
        Stage 2: Cross-encoder reranking → top 5 final results.

        Args:
            query: Search query (claim or retrieval query).

        Returns:
            Dict with:
                - documents: top reranked document texts
                - metadatas: metadata for reranked docs
                - avg_cosine_similarity: mean Stage 1 similarity
                - num_retrieved: number of Stage 1 results
                - rerank_scores: Stage 2 scores
        """
        # Stage 1: Dense retrieval
        stage1 = self._stage1_dense_retrieve(query, n_results=self.top_k_retrieve)

        if not stage1["documents"]:
            return {
                "documents": [],
                "metadatas": [],
                "avg_cosine_similarity": 0.0,
                "num_retrieved": 0,
                "rerank_scores": [],
            }

        avg_sim = sum(stage1["similarities"]) / len(stage1["similarities"])

        # Stage 2: Rerank
        reranked = self._stage2_rerank(
            query, stage1["documents"], top_k=self.top_k_rerank
        )

        # Map back to metadatas
        reranked_docs = []
        reranked_metas = []
        rerank_scores = []

        for item in reranked:
            idx = item["original_index"]
            reranked_docs.append(item["document"])
            reranked_metas.append(
                stage1["metadatas"][idx] if idx < len(stage1["metadatas"]) else {}
            )
            rerank_scores.append(item["rerank_score"])

        return {
            "documents": reranked_docs,
            "metadatas": reranked_metas,
            "avg_cosine_similarity": round(avg_sim, 4),
            "num_retrieved": len(stage1["documents"]),
            "rerank_scores": rerank_scores,
        }
