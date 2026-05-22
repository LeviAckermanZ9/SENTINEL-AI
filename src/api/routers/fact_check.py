"""
SENTINEL-AI — Fact Check Router
POST /fact-check/ (RAG-only) and POST /fact-check/full-analysis (combined).
Includes _fuse_verdicts() for ML+RAG consensus logic.
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.monitoring.metrics import (
    CHROMA_HITS,
    CONTRADICTIONS_TOTAL,
    LATENCY,
    PREDICTIONS,
    RAG_AVG_SIM,
    RAG_LATENCY,
    RAG_REQUESTS,
)

router = APIRouter(prefix="/fact-check", tags=["fact-check"])

# Injected by main.py lifespan
MODELS = {}


class FactCheckRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=5000)
    include_rag: bool = Field(True)
    include_explanation: bool = Field(False)
    include_summary: bool = Field(False)


class FactCheckResponse(BaseModel):
    rag_verdict: str
    rag_confidence: str
    rag_reasoning: str
    key_sources: str
    primary_claim: str
    retrieved_docs: List[Dict[str, Any]]
    avg_retrieval_similarity: float
    num_sources_retrieved: int
    processing_time_ms: float


class FullAnalysisResponse(BaseModel):
    # ML results
    ml_classification: str
    ml_confidence: float
    # RAG results
    rag_verdict: str
    rag_confidence: str
    rag_reasoning: str
    # Fusion
    fused_verdict: str
    fused_confidence: str
    fusion_method: str
    human_review_required: bool
    # Details
    primary_claim: str
    claim_triplet: Optional[Dict[str, str]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    sentiment: Optional[Dict[str, Any]] = None
    manipulation_score: Optional[float] = None
    retrieved_docs: List[Dict[str, Any]]
    avg_retrieval_similarity: float
    processing_time_ms: float


def _fuse_verdicts(
    ml_class: str,
    ml_conf: float,
    rag_verdict: str,
    rag_conf: str,
) -> Dict[str, Any]:
    """
    Fuse ML classification with RAG verdict.

    Cases:
        1. Both fake       → FAKE_NEWS, HIGH, ml_rag_agreement
        2. ML fake + unverifiable → LIKELY_FAKE, MEDIUM, ml_only
        3. ML fake + supported → UNCERTAIN, LOW, human_review=True
        4. ML real + contradicted → LIKELY_FAKE, MEDIUM, rag_only, human_review=True
        5. Neither          → LIKELY_REAL, MEDIUM
    """
    ml_is_fake = ml_class in ("FAKE_NEWS", "FAKE")
    ml_is_real = ml_class in ("REAL_NEWS", "REAL")
    rag_is_contra = rag_verdict == "CONTRADICTED"
    rag_is_support = rag_verdict == "SUPPORTED"
    rag_is_unverif = rag_verdict == "UNVERIFIABLE"

    # Case 1: Both agree it's fake
    if ml_is_fake and rag_is_contra:
        return {
            "fused_verdict": "FAKE_NEWS",
            "fused_confidence": "HIGH",
            "fusion_method": "ml_rag_agreement",
            "human_review_required": False,
        }

    # Case 2: ML says fake, RAG can't verify
    if ml_is_fake and rag_is_unverif:
        return {
            "fused_verdict": "LIKELY_FAKE",
            "fused_confidence": "MEDIUM",
            "fusion_method": "ml_only",
            "human_review_required": False,
        }

    # Case 3: ML says fake, RAG says supported — conflict
    if ml_is_fake and rag_is_support:
        return {
            "fused_verdict": "UNCERTAIN",
            "fused_confidence": "LOW",
            "fusion_method": "ml_rag_conflict",
            "human_review_required": True,
        }

    # Case 4: ML says real, RAG says contradicted — trust RAG
    if ml_is_real and rag_is_contra:
        return {
            "fused_verdict": "LIKELY_FAKE",
            "fused_confidence": "MEDIUM",
            "fusion_method": "rag_only",
            "human_review_required": True,
        }

    # Case 5: Default — neither strongly fake
    return {
        "fused_verdict": "LIKELY_REAL",
        "fused_confidence": "MEDIUM",
        "fusion_method": "default",
        "human_review_required": False,
    }


@router.post("/", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest):
    """RAG-only fact-check endpoint."""
    start = time.time()

    rag_pipeline = MODELS.get("rag_pipeline")
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503, detail="RAG pipeline not initialized. Build KB first."
        )

    try:
        RAG_REQUESTS.inc()
        CHROMA_HITS.inc()

        result = rag_pipeline.run(request.text)

        if result["rag_verdict"] == "CONTRADICTED":
            CONTRADICTIONS_TOTAL.inc()
        RAG_AVG_SIM.set(result["avg_retrieval_similarity"])

        elapsed = (time.time() - start) * 1000
        RAG_LATENCY.observe(elapsed / 1000)

        return FactCheckResponse(
            rag_verdict=result["rag_verdict"],
            rag_confidence=result["rag_confidence"],
            rag_reasoning=result["rag_reasoning"],
            key_sources=result["key_sources"],
            primary_claim=result["primary_claim_extracted"],
            retrieved_docs=result["retrieved_docs"],
            avg_retrieval_similarity=result["avg_retrieval_similarity"],
            num_sources_retrieved=result["num_sources_retrieved"],
            processing_time_ms=round(elapsed, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full-analysis", response_model=FullAnalysisResponse)
async def full_analysis(request: FactCheckRequest):
    """Combined ML + RAG analysis with fusion verdict."""
    start = time.time()
    text = request.text

    # --- ML Pipeline ---
    import torch

    ml_class = "REAL_NEWS"
    ml_conf = 0.5

    cleaner = MODELS.get("cleaner")
    pos_parser = MODELS.get("pos_parser")
    extractor = MODELS.get("feature_extractor")
    mlp = MODELS.get("mlp")

    label_map = {0: "REAL_NEWS", 1: "FAKE_NEWS", 2: "SATIRE", 3: "SPAM"}

    cleaned = cleaner.clean(text) if cleaner else {"cleaned_text": text}
    pos_result = (
        pos_parser.parse(text)
        if pos_parser
        else {"adjective_density": 0.0, "adj_noun_ratio": 0.0}
    )
    adj_density = pos_result.get("adjective_density", 0.0)

    if extractor and mlp:
        features = extractor.transform(
            cleaned["cleaned_text"],
            adjective_density=adj_density,
            adj_noun_ratio=pos_result.get("adj_noun_ratio", 0.0),
        )
        # Pad or truncate to exactly 504 dims that MLP expects
        feat_vec = features.get("feature_vector", features.get("tfidf_vector", []))
        if isinstance(feat_vec, list):
            if len(feat_vec) < 504:
                feat_vec = feat_vec + [0.0] * (504 - len(feat_vec))
            elif len(feat_vec) > 504:
                feat_vec = feat_vec[:504]
        feat_t = torch.FloatTensor(feat_vec).unsqueeze(0)
        mlp.eval()
        if feat_t.shape[1] != 504:
            feat_t = torch.nn.functional.pad(
                feat_t, (0, max(0, 504 - feat_t.shape[1]))
            )[:, :504]
        with torch.no_grad():
            logits = mlp(feat_t)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            ml_class = label_map.get(pred, "UNKNOWN")
            ml_conf = probs[0][pred].item()

    PREDICTIONS.labels(predicted_class=ml_class).inc()

    # --- NER ---
    entities = None
    ner = MODELS.get("ner_pipeline")
    if ner:
        try:
            entities = ner.extract_entities(text).get("entities", [])
        except Exception:
            pass

    # --- Summary ---
    summary = None
    if request.include_summary:
        summarizer = MODELS.get("summarizer")
        if summarizer and len(text) > 100:
            try:
                summary = summarizer.summarize(text).get("summary_text")
            except Exception:
                pass

    # --- Sentiment ---
    sentiment = None
    manip_score = None
    sent_model = MODELS.get("sentiment")
    if sent_model:
        try:
            sr = sent_model.analyze(text, adj_density=adj_density)
            sentiment = {
                "finbert": sr["finbert_sentiment"],
                "roberta": sr["roberta_sentiment"],
            }
            manip_score = sr["manipulation_score"]
        except Exception:
            pass

    # --- RAG ---
    rag_verdict = "UNVERIFIABLE"
    rag_conf = "LOW"
    rag_reasoning = "RAG pipeline not available."
    retrieved_docs = []
    avg_sim = 0.0
    claim = text
    claim_triplet = None

    rag = MODELS.get("rag_pipeline")
    if rag and request.include_rag:
        try:
            RAG_REQUESTS.inc()
            CHROMA_HITS.inc()
            rr = rag.run(text)
            rag_verdict = rr["rag_verdict"]
            rag_conf = rr["rag_confidence"]
            rag_reasoning = rr["rag_reasoning"]
            retrieved_docs = rr["retrieved_docs"]
            avg_sim = rr["avg_retrieval_similarity"]
            claim = rr["primary_claim_extracted"]
            claim_triplet = rr["claim_triplet"]
            RAG_AVG_SIM.set(avg_sim)
            if rag_verdict == "CONTRADICTED":
                CONTRADICTIONS_TOTAL.inc()
        except Exception:
            pass

    # --- Fusion ---
    fusion = _fuse_verdicts(ml_class, ml_conf, rag_verdict, rag_conf)

    elapsed = (time.time() - start) * 1000
    LATENCY.observe(elapsed / 1000)

    return FullAnalysisResponse(
        ml_classification=ml_class,
        ml_confidence=round(ml_conf, 4),
        rag_verdict=rag_verdict,
        rag_confidence=rag_conf,
        rag_reasoning=rag_reasoning,
        fused_verdict=fusion["fused_verdict"],
        fused_confidence=fusion["fused_confidence"],
        fusion_method=fusion["fusion_method"],
        human_review_required=fusion["human_review_required"],
        primary_claim=claim,
        claim_triplet=claim_triplet,
        entities=entities,
        summary=summary,
        sentiment=sentiment,
        manipulation_score=round(manip_score, 4) if manip_score else None,
        retrieved_docs=retrieved_docs,
        avg_retrieval_similarity=avg_sim,
        processing_time_ms=round(elapsed, 2),
    )
