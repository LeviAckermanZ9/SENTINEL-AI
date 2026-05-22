"""
SENTINEL-AI — API Test Suite
Tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_healthy(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_has_required_fields(self):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "kb_size" in data
        assert "device" in data
        assert "rag_available" in data

    def test_health_device_is_string(self):
        response = client.get("/health")
        data = response.json()
        assert data["device"] in ("cpu", "cuda")


class TestClassifyEndpoint:
    def test_classify_valid_text(self):
        response = client.post("/classify", json={
            "text": "The president of the United States signed a new executive order on climate change today.",
            "include_explanation": False,
            "include_summary": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["classification"] in ["REAL_NEWS", "FAKE_NEWS", "SATIRE", "SPAM"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["processing_time_ms"] > 0

    def test_classify_too_short(self):
        response = client.post("/classify", json={
            "text": "Short text",
        })
        assert response.status_code == 422  # Pydantic validation

    def test_classify_returns_method(self):
        response = client.post("/classify", json={
            "text": "Breaking news: Scientists have discovered a new treatment for cancer that shows promising results.",
            "include_explanation": False,
            "include_summary": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "method" in data


class TestFactCheckEndpoint:
    def test_factcheck_rag_only(self):
        response = client.post("/fact-check/", json={
            "text": "Scientists confirm that 5G technology causes COVID-19 infections in humans.",
        })
        # May return 503 if RAG not built — both are valid
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert data["rag_verdict"] in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]

    def test_factcheck_too_short(self):
        response = client.post("/fact-check/", json={
            "text": "Short text",
        })
        assert response.status_code == 422

    def test_full_analysis(self):
        response = client.post("/fact-check/full-analysis", json={
            "text": "The World Health Organization declared a global pandemic due to the new virus variant.",
        })
        # May return 200 even without RAG (uses defaults)
        assert response.status_code == 200
        data = response.json()
        assert "fused_verdict" in data
        assert "ml_classification" in data
        assert "human_review_required" in data


class TestRootEndpoint:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SENTINEL-AI"
        assert data["version"] == "1.0.0"
