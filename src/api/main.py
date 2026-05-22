"""
SENTINEL-AI — FastAPI Application
Main entrypoint: lifespan loads 13 components, mounts routers + middleware.
"""

import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app

from src.api.middleware import BodySizeLimitMiddleware, ProcessTimeMiddleware
from src.api.routers import classify, explain, fact_check, health, summarize

logger = logging.getLogger("sentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all 13 pipeline components on startup.

    Order: cleaner → pos_parser → feature_extractor → mlp → bilstm →
           textcnn → autoencoder → bert_classifier → summarizer →
           ner_pipeline → sentiment → explainer → rag_pipeline
    """
    MODELS = {}
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger.info("=" * 60)
    logger.info("SENTINEL-AI Platform Starting...")
    logger.info("=" * 60)

    # 1. Cleaner
    try:
        from src.preprocessing.cleaner import TextCleaner
        MODELS["cleaner"] = TextCleaner()
        logger.info("[1/13] TextCleaner loaded")
    except Exception as e:
        logger.warning(f"[1/13] TextCleaner failed: {e}")
        MODELS["cleaner"] = None

    # 2. POS Parser
    try:
        from src.preprocessing.pos_parser import PosParser
        MODELS["pos_parser"] = PosParser()
        logger.info("[2/13] PosParser loaded")
    except Exception as e:
        logger.warning(f"[2/13] PosParser failed: {e}")
        MODELS["pos_parser"] = None

    # 3. Feature Extractor (needs fit — will be fitted in classify if needed)
    try:
        from src.preprocessing.feature_extractor import FeatureExtractor
        fe = FeatureExtractor()
        # Fit on dummy corpus so transform() works
        fe.fit(["dummy text for initialization"] * 10)
        MODELS["feature_extractor"] = fe
        logger.info("[3/13] FeatureExtractor loaded")
    except Exception as e:
        logger.warning(f"[3/13] FeatureExtractor failed: {e}")
        MODELS["feature_extractor"] = None

    # 4. MLP Classifier
    try:
        from src.models.mlp_classifier import MLPClassifier
        MODELS["mlp"] = MLPClassifier(input_dim=504, num_classes=4)
        logger.info("[4/13] MLPClassifier loaded")
    except Exception as e:
        logger.warning(f"[4/13] MLPClassifier failed: {e}")
        MODELS["mlp"] = None

    # 5. BiLSTM Attention
    try:
        from src.models.bilstm_attention import BiLSTMAttention
        MODELS["bilstm"] = BiLSTMAttention()
        logger.info("[5/13] BiLSTMAttention loaded")
    except Exception as e:
        logger.warning(f"[5/13] BiLSTMAttention failed: {e}")
        MODELS["bilstm"] = None

    # 6. TextCNN
    try:
        from src.models.cnn_moderation import TextCNN
        MODELS["textcnn"] = TextCNN()
        logger.info("[6/13] TextCNN loaded")
    except Exception as e:
        logger.warning(f"[6/13] TextCNN failed: {e}")
        MODELS["textcnn"] = None

    # 7. Autoencoder
    try:
        from src.models.autoencoder import NewsAutoencoder
        MODELS["autoencoder"] = NewsAutoencoder()
        logger.info("[7/13] NewsAutoencoder loaded")
    except Exception as e:
        logger.warning(f"[7/13] NewsAutoencoder failed: {e}")
        MODELS["autoencoder"] = None

    # 8. BERT Classifier (lazy — no model download at startup)
    try:
        from src.nlp.bert_classifier import TransformerClassifier
        MODELS["bert_classifier"] = TransformerClassifier()
        logger.info("[8/13] TransformerClassifier loaded (lazy)")
    except Exception as e:
        logger.warning(f"[8/13] TransformerClassifier failed: {e}")
        MODELS["bert_classifier"] = None

    # 9. Summarizer (lazy — no model download at startup)
    try:
        from src.nlp.bart_summarizer import SummarizerModule
        MODELS["summarizer"] = SummarizerModule()
        logger.info("[9/13] SummarizerModule loaded (lazy)")
    except Exception as e:
        logger.warning(f"[9/13] SummarizerModule failed: {e}")
        MODELS["summarizer"] = None

    # 10. NER Pipeline (lazy)
    try:
        from src.nlp.ner_pipeline import NERPipeline
        MODELS["ner_pipeline"] = NERPipeline()
        logger.info("[10/13] NERPipeline loaded (lazy)")
    except Exception as e:
        logger.warning(f"[10/13] NERPipeline failed: {e}")
        MODELS["ner_pipeline"] = None

    # 11. Sentiment Analyzer (lazy)
    try:
        from src.nlp.sentiment import SentimentAnalyzer
        MODELS["sentiment"] = SentimentAnalyzer()
        logger.info("[11/13] SentimentAnalyzer loaded (lazy)")
    except Exception as e:
        logger.warning(f"[11/13] SentimentAnalyzer failed: {e}")
        MODELS["sentiment"] = None

    # 12. Explainer
    try:
        from src.models.explainer import ModelExplainer
        MODELS["explainer"] = ModelExplainer(model=MODELS.get("mlp"))
        logger.info("[12/13] ModelExplainer loaded")
    except Exception as e:
        logger.warning(f"[12/13] ModelExplainer failed: {e}")
        MODELS["explainer"] = None

    # 13. RAG Pipeline (may fail if ChromaDB/KB not built)
    try:
        from src.rag.pipeline import SentinelRAGPipeline
        MODELS["rag_pipeline"] = SentinelRAGPipeline(use_heuristic=True)
        logger.info("[13/13] RAG Pipeline loaded")
    except Exception as e:
        logger.warning(f"[13/13] RAG Pipeline unavailable: {e}")
        MODELS["rag_pipeline"] = None

    loaded = len([v for v in MODELS.values() if v is not None])
    logger.info(f"\nStartup complete: {loaded}/13 components loaded")

    # Inject MODELS into routers
    health.MODELS = MODELS
    classify.MODELS = MODELS
    fact_check.MODELS = MODELS
    explain.MODELS = MODELS
    summarize.MODELS = MODELS

    yield

    logger.info("SENTINEL-AI shutting down...")
    MODELS.clear()


# Create FastAPI app
app = FastAPI(
    title="SENTINEL-AI",
    description="Real-Time Social Media Intelligence & Content Safety Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Middleware ---
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ProcessTimeMiddleware)
app.add_middleware(BodySizeLimitMiddleware)

# --- Prometheus metrics ASGI ---
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# --- Routers ---
app.include_router(health.router)
app.include_router(classify.router)
app.include_router(fact_check.router)
app.include_router(explain.router)
app.include_router(summarize.router)


@app.get("/")
async def root():
    return {
        "service": "SENTINEL-AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
