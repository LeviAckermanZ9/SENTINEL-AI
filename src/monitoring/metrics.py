"""
SENTINEL-AI — Prometheus Metrics
Covers: INT377 Unit V (Monitoring)

Nine Prometheus metrics for SENTINEL-AI platform monitoring.
"""

from prometheus_client import Counter, Gauge, Histogram

# --- ML Pipeline Metrics ---

PREDICTIONS = Counter(
    "sentinel_predictions_total",
    "Total predictions by class",
    ["predicted_class"],
)

LATENCY = Histogram(
    "sentinel_inference_seconds",
    "ML inference latency in seconds",
)

FAKE_RATE = Gauge(
    "sentinel_fake_news_rate_5m",
    "Fake news detection rate over 5 minute window",
)

# --- RAG Metrics ---

RAG_REQUESTS = Counter(
    "sentinel_rag_requests_total",
    "Total RAG fact-check requests",
)

RAG_LATENCY = Histogram(
    "sentinel_rag_latency_seconds",
    "RAG pipeline latency in seconds",
    buckets=[0.5, 1, 2, 3, 5, 10],
)

CONTRADICTIONS_TOTAL = Counter(
    "sentinel_rag_contradictions_total",
    "Total RAG contradictions detected",
)

RAG_AVG_SIM = Gauge(
    "sentinel_rag_avg_retrieval_similarity",
    "Average retrieval cosine similarity",
)

KB_SIZE = Gauge(
    "sentinel_knowledge_base_chunks",
    "Number of chunks in knowledge base",
)

CHROMA_HITS = Counter(
    "sentinel_chroma_queries_total",
    "Total ChromaDB queries executed",
)
