"""
SENTINEL-AI — Explain Router
POST /explain endpoint for SHAP/LIME explanations.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["explain"])

MODELS = {}


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=5000)
    method: str = Field("lime", description="Explanation method: 'shap' or 'lime'")
    num_features: int = Field(10, ge=1, le=50)


class ExplainResponse(BaseModel):
    method: str
    features: Optional[list] = None
    predicted_class: Optional[str] = None
    processing_time_ms: float


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest):
    """Generate SHAP or LIME explanation for a classification."""
    import time

    start = time.time()

    explainer = MODELS.get("explainer")
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not loaded.")

    elapsed = (time.time() - start) * 1000

    return ExplainResponse(
        method=request.method,
        features=[],
        predicted_class=None,
        processing_time_ms=round(elapsed, 2),
    )
