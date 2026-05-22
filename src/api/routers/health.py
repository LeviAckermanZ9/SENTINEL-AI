"""
SENTINEL-AI — Health Router
GET /health endpoint for liveness/readiness probes.
"""

import torch
from fastapi import APIRouter

router = APIRouter(tags=["health"])

# Will be injected by main.py lifespan
MODELS = {}


@router.get("/health")
async def health_check():
    """Health check endpoint — unauthenticated for K8s probes."""
    rag_available = MODELS.get("rag_pipeline") is not None
    kb_size = 0

    if rag_available:
        try:
            coll = MODELS["rag_pipeline"].retriever._load_collection()
            kb_size = coll.count()
        except Exception:
            kb_size = 0

    return {
        "status": "healthy",
        "models_loaded": len([k for k, v in MODELS.items() if v is not None]),
        "kb_size": kb_size,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "rag_available": rag_available,
    }
