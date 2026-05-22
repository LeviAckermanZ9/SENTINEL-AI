"""
SENTINEL-AI — RAG Test Suite
Tests for: ClaimExtractor, FactCheckOutputParser, Retriever (EphemeralClient), API stubs.
Uses chromadb.EphemeralClient only — no external service required.
"""

import pytest

from src.rag.rag_chain import FactCheckOutputParser


# ================================================================
# FactCheckOutputParser Tests
# ================================================================

class TestFactCheckOutputParser:
    """Tests for FactCheckOutputParser regex parsing."""

    def setup_method(self):
        self.parser = FactCheckOutputParser()

    def test_parse_well_formed(self):
        text = (
            "VERDICT: CONTRADICTED\n"
            "CONFIDENCE: HIGH\n"
            "REASONING: The claim is false based on multiple sources.\n"
            "KEY_SOURCES: WHO, CDC, PolitiFact"
        )
        result = self.parser.parse(text)
        assert result["verdict"] == "CONTRADICTED"
        assert result["confidence"] == "HIGH"
        assert "false" in result["reasoning"].lower()
        assert "WHO" in result["key_sources"]

    def test_parse_supported(self):
        text = (
            "VERDICT: SUPPORTED\n"
            "CONFIDENCE: MEDIUM\n"
            "REASONING: Evidence aligns with the claim.\n"
            "KEY_SOURCES: Reuters"
        )
        result = self.parser.parse(text)
        assert result["verdict"] == "SUPPORTED"
        assert result["confidence"] == "MEDIUM"

    def test_parse_malformed_defaults(self):
        text = "This is just some random text without any structure."
        result = self.parser.parse(text)
        assert result["verdict"] == "UNVERIFIABLE"
        assert result["confidence"] == "LOW"

    def test_parse_empty_string(self):
        result = self.parser.parse("")
        assert result["verdict"] == "UNVERIFIABLE"
        assert result["confidence"] == "LOW"

    def test_parse_none_input(self):
        result = self.parser.parse(None)
        assert result["verdict"] == "UNVERIFIABLE"
        assert result["confidence"] == "LOW"

    def test_parse_case_insensitive(self):
        text = (
            "verdict: supported\n"
            "confidence: high\n"
            "reasoning: The evidence supports this.\n"
            "key_sources: AP News"
        )
        result = self.parser.parse(text)
        assert result["verdict"] == "SUPPORTED"
        assert result["confidence"] == "HIGH"

    def test_parse_partial_output(self):
        text = "VERDICT: CONTRADICTED\nSome other stuff here"
        result = self.parser.parse(text)
        assert result["verdict"] == "CONTRADICTED"
        assert result["confidence"] == "LOW"  # missing → default

    def test_valid_verdict_values(self):
        for verdict in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]:
            text = f"VERDICT: {verdict}\nCONFIDENCE: MEDIUM\nREASONING: Test.\nKEY_SOURCES: test"
            result = self.parser.parse(text)
            assert result["verdict"] == verdict

    def test_defaults_method(self):
        defaults = self.parser._defaults()
        assert defaults["verdict"] == "UNVERIFIABLE"
        assert defaults["confidence"] == "LOW"
        assert isinstance(defaults["reasoning"], str)


# ================================================================
# ClaimExtractor Tests (uses spaCy only — no HF download in tests)
# ================================================================

class TestClaimExtractor:
    """Tests for ClaimExtractor structure (no zero-shot model download)."""

    def test_import(self):
        from src.rag.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        assert extractor is not None

    def test_split_sentences(self):
        from src.rag.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        sentences = extractor._split_sentences(
            "First sentence here. Second sentence follows. Third one too."
        )
        assert len(sentences) >= 2

    def test_extract_triplet_from_sentence(self):
        from src.rag.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        triplet = extractor._extract_triplet_from_sentence(
            "Scientists confirmed the discovery of a new species."
        )
        assert "subject" in triplet
        assert "verb" in triplet
        assert "object" in triplet

    def test_extract_claim_triplet_returns_dict(self):
        """Test that extract_claim_triplet returns proper structure
        (skips zero-shot model — uses single sentence fast path)."""
        from src.rag.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        # Single sentence → skips zero-shot scoring
        result = extractor.extract_claim_triplet(
            "Scientists confirmed that vaccines are safe and effective."
        )
        assert "primary_claim" in result
        assert "subject" in result
        assert "verb" in result
        assert "object" in result
        assert "retrieval_query" in result
        assert isinstance(result["retrieval_query"], str)
        assert len(result["retrieval_query"]) > 0


# ================================================================
# Retriever Integration Tests (EphemeralClient — no external service)
# ================================================================

class TestRetrieverIntegration:
    """Tests using chromadb.EphemeralClient with test documents."""

    def _create_test_collection(self):
        """Create an ephemeral ChromaDB collection with 5 test docs."""
        import chromadb
        import uuid
        from sentence_transformers import SentenceTransformer

        client = chromadb.EphemeralClient()
        col_name = f"test_factcheck_{uuid.uuid4().hex[:8]}"
        collection = client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        test_docs = [
            "CLAIM: The moon landing in 1969 was real. VERDICT: TRUE EVIDENCE: NASA Apollo 11 mission successfully landed astronauts on the moon on July 20, 1969.",
            "CLAIM: 5G towers cause COVID-19. VERDICT: FALSE EVIDENCE: Multiple scientific studies have confirmed there is no link between 5G technology and coronavirus.",
            "CLAIM: The Earth is flat. VERDICT: FALSE EVIDENCE: Centuries of scientific evidence confirm the Earth is an oblate spheroid.",
            "CLAIM: Vaccines cause autism. VERDICT: FALSE EVIDENCE: The original study by Andrew Wakefield was retracted and found to be fraudulent.",
            "CLAIM: Climate change is caused by human activities. VERDICT: TRUE EVIDENCE: Over 97% of climate scientists agree that human activities are the primary cause.",
        ]

        test_metas = [
            {"source": "nasa", "verdict": "true", "speaker": "NASA", "original_claim": "Moon landing was real"},
            {"source": "who", "verdict": "false", "speaker": "WHO", "original_claim": "5G causes COVID"},
            {"source": "science", "verdict": "false", "speaker": "scientists", "original_claim": "Earth is flat"},
            {"source": "cdc", "verdict": "false", "speaker": "CDC", "original_claim": "Vaccines cause autism"},
            {"source": "ipcc", "verdict": "true", "speaker": "IPCC", "original_claim": "Climate change human-caused"},
        ]

        embeddings = embedder.encode(test_docs, convert_to_numpy=True).tolist()

        collection.add(
            ids=[f"test_{i}" for i in range(len(test_docs))],
            documents=test_docs,
            embeddings=embeddings,
            metadatas=test_metas,
        )

        return collection, embedder

    def test_collection_has_documents(self):
        collection, _ = self._create_test_collection()
        assert collection.count() == 5

    def test_cosine_query_returns_results(self):
        collection, embedder = self._create_test_collection()

        query = "Did humans land on the moon?"
        query_emb = embedder.encode(query, convert_to_numpy=True).tolist()

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
        )

        assert len(results["documents"][0]) == 3
        assert len(results["distances"][0]) == 3

    def test_moon_landing_in_top_3(self):
        collection, embedder = self._create_test_collection()

        query = "Was the 1969 moon landing real or faked?"
        query_emb = embedder.encode(query, convert_to_numpy=True).tolist()

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
        )

        docs = results["documents"][0]
        found_moon = any("moon" in doc.lower() and "landing" in doc.lower() for doc in docs)
        assert found_moon, f"Moon landing doc not in top 3. Got: {[d[:50] for d in docs]}"

    def test_5g_query_returns_relevant(self):
        collection, embedder = self._create_test_collection()

        query = "Does 5G technology spread coronavirus?"
        query_emb = embedder.encode(query, convert_to_numpy=True).tolist()

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
        )

        docs = results["documents"][0]
        found_5g = any("5g" in doc.lower() for doc in docs)
        assert found_5g, f"5G doc not found in results."

    def test_retriever_with_set_collection(self):
        """Test SentinelRetriever with manually set collection."""
        from src.rag.retriever import SentinelRetriever

        collection, _ = self._create_test_collection()

        retriever = SentinelRetriever()
        retriever.set_collection(collection)

        results = retriever._stage1_dense_retrieve("moon landing apollo", n_results=3)
        assert len(results["documents"]) > 0
        assert len(results["similarities"]) > 0
        assert all(0 <= s <= 2 for s in results["similarities"])


# ================================================================
# API Endpoint Stubs (skip if API not initialized)
# ================================================================

class TestAPIEndpoints:
    """Stub tests for API endpoints — marked skip if not available."""

    @pytest.mark.skip(reason="API tests run in Phase 6 after main.py is built")
    def test_health_endpoint(self):
        pass

    @pytest.mark.skip(reason="API tests run in Phase 6 after main.py is built")
    def test_classify_endpoint(self):
        pass

    @pytest.mark.skip(reason="API tests run in Phase 6 after main.py is built")
    def test_factcheck_endpoint(self):
        pass
