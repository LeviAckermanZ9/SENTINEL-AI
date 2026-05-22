"""
SENTINEL-AI — RAG Pipeline Orchestrator
Covers: CSR322 CO5 (RAG system design)

Orchestrates: claim_extractor → retriever → rag_chain.
Returns full structured analysis with timing.
"""

import os
import time
from typing import Any, Dict, Optional

from src.rag.claim_extractor import ClaimExtractor
from src.rag.retriever import SentinelRetriever
from src.rag.rag_chain import SentinelRAGChain


class SentinelRAGPipeline:
    """
    End-to-end RAG fact-checking pipeline.

    Orchestrates: ClaimExtractor → SentinelRetriever → SentinelRAGChain.

    Usage:
        pipeline = SentinelRAGPipeline()
        result = pipeline.run("Scientists confirm 5G causes COVID-19.")
    """

    def __init__(
        self,
        use_api_llm: bool = False,
        use_heuristic: bool = False,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        persist_dir: str = "./data/chromadb",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            use_api_llm: Use API-based LLM (Mistral) instead of local (Flan-T5).
            use_heuristic: Use heuristic fact-check instead of LLM.
            chroma_host: ChromaDB host override.
            chroma_port: ChromaDB port override.
            persist_dir: Local ChromaDB path.
        """
        self.use_api_llm = use_api_llm or os.environ.get("USE_API_LLM", "false").lower() == "true"
        self.use_heuristic = use_heuristic

        self.claim_extractor = ClaimExtractor()
        self.retriever = SentinelRetriever(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            persist_dir=persist_dir,
        )
        self.rag_chain = SentinelRAGChain(use_api=self.use_api_llm)

    def run(self, text: str) -> Dict[str, Any]:
        """
        Run the full RAG fact-checking pipeline.

        Pipeline:
            1. Extract primary claim + triplet
            2. Retrieve evidence from knowledge base
            3. Run LLM fact-check chain (or heuristic fallback)

        Args:
            text: Input text to fact-check.

        Returns:
            Dict with:
                - rag_verdict: SUPPORTED / CONTRADICTED / UNVERIFIABLE
                - rag_confidence: HIGH / MEDIUM / LOW
                - rag_reasoning: explanation string
                - key_sources: source references
                - primary_claim_extracted: the claim sentence
                - claim_triplet: {subject, verb, object}
                - retrieved_docs: list of {source, verdict_label, relevance, excerpt}
                - avg_retrieval_similarity: float
                - num_sources_retrieved: int
                - rag_processing_ms: float
        """
        start_time = time.time()

        # Step 1: Extract claim
        try:
            triplet_result = self.claim_extractor.extract_claim_triplet(text)
            primary_claim = triplet_result["primary_claim"]
            retrieval_query = triplet_result["retrieval_query"]
            claim_triplet = {
                "subject": triplet_result["subject"],
                "verb": triplet_result["verb"],
                "object": triplet_result["object"],
            }
        except Exception as e:
            primary_claim = text
            retrieval_query = text
            claim_triplet = {"subject": "", "verb": "", "object": ""}

        # Step 2: Retrieve evidence
        try:
            evidence = self.retriever.get_relevant_evidence(retrieval_query)
            documents = evidence["documents"]
            metadatas = evidence["metadatas"]
            avg_sim = evidence["avg_cosine_similarity"]
            num_retrieved = evidence["num_retrieved"]
        except Exception as e:
            documents = []
            metadatas = []
            avg_sim = 0.0
            num_retrieved = 0

        # Step 3: Fact-check via LLM or heuristic
        if documents:
            try:
                if self.use_heuristic:
                    chain_result = self.rag_chain.run_without_llm(primary_claim, documents)
                else:
                    chain_result = self.rag_chain.run(primary_claim, documents)
            except Exception:
                # Fallback to heuristic
                chain_result = self.rag_chain.run_without_llm(primary_claim, documents)
        else:
            chain_result = {
                "verdict": "UNVERIFIABLE",
                "confidence": "LOW",
                "reasoning": "No relevant evidence found in knowledge base.",
                "key_sources": "",
            }

        # Format retrieved docs
        retrieved_docs = []
        rerank_scores = evidence.get("rerank_scores", []) if documents else []

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            retrieved_docs.append({
                "source": meta.get("source", "unknown"),
                "verdict_label": meta.get("verdict", "unknown"),
                "relevance": round(rerank_scores[i], 4) if i < len(rerank_scores) else 0.0,
                "excerpt": doc[:300],
            })

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "rag_verdict": chain_result["verdict"],
            "rag_confidence": chain_result["confidence"],
            "rag_reasoning": chain_result["reasoning"],
            "key_sources": chain_result["key_sources"],
            "primary_claim_extracted": primary_claim,
            "claim_triplet": claim_triplet,
            "retrieved_docs": retrieved_docs,
            "avg_retrieval_similarity": avg_sim,
            "num_sources_retrieved": num_retrieved,
            "rag_processing_ms": round(elapsed_ms, 2),
        }

    def run_batch(self, texts: list) -> list:
        """Run pipeline on multiple texts."""
        return [self.run(text) for text in texts]
