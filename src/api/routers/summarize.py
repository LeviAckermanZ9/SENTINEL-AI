"""
SENTINEL-AI — Summarize Router
POST /summarize endpoint for BART/T5 summarization.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["summarize"])

MODELS = {}


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=5000)
    model: str = Field("bart", description="Model: 'bart' or 't5'")
    max_length: int = Field(60, ge=20, le=200)
    min_length: int = Field(20, ge=10, le=100)


class SummarizeResponse(BaseModel):
    summary: str
    model_used: str
    processing_time_ms: float


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Generate abstractive summary using BART or T5."""
    import time

    start = time.time()

    summarizer = MODELS.get("summarizer")
    if summarizer is None:
        raise HTTPException(status_code=503, detail="Summarizer not loaded.")

    try:
        result = summarizer.summarize(
            request.text,
            model=request.model,
            max_length=request.max_length,
            min_length=request.min_length,
        )
        elapsed = (time.time() - start) * 1000

        return SummarizeResponse(
            summary=result["summary_text"],
            model_used=result["model_used"],
            processing_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
