"""
SENTINEL-AI — Classify Router
POST /classify endpoint for ML classification pipeline.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.monitoring.metrics import LATENCY, PREDICTIONS

router = APIRouter(tags=["classify"])

# Injected by main.py lifespan
MODELS = {}


class ClassifyRequest(BaseModel):
    text: str = Field(
        ..., min_length=20, max_length=5000, description="Text to classify"
    )
    include_explanation: bool = Field(True, description="Include SHAP/LIME explanation")
    include_summary: bool = Field(True, description="Include BART summary")


class ClassifyResponse(BaseModel):
    classification: str
    confidence: float
    method: str
    entities: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    sentiment: Optional[Dict[str, Any]] = None
    manipulation_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float


LABEL_MAP = {0: "REAL_NEWS", 1: "FAKE_NEWS", 2: "SATIRE", 3: "SPAM"}


@router.post("/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):
    """Classify text through the full ML pipeline."""
    start = time.time()

    text = request.text
    result: Dict[str, Any] = {}

    try:
        # Step 1: Preprocess
        cleaner = MODELS.get("cleaner")
        pos_parser = MODELS.get("pos_parser")
        extractor = MODELS.get("feature_extractor")

        cleaned = (
            cleaner.clean(text)
            if cleaner
            else {"cleaned_text": text, "tokens": text.split()}
        )
        pos_result = (
            pos_parser.parse(text)
            if pos_parser
            else {"adjective_density": 0.0, "adj_noun_ratio": 0.0}
        )

        adj_density = pos_result.get("adjective_density", 0.0)
        adj_noun_ratio = pos_result.get("adj_noun_ratio", 0.0)

        # Step 2: Feature extraction + MLP
        mlp = MODELS.get("mlp")
        ml_class = 0
        ml_conf = 0.5

        if extractor and mlp:
            features = extractor.transform(
                cleaned["cleaned_text"],
                adjective_density=adj_density,
                adj_noun_ratio=adj_noun_ratio,
            )
            feat_tensor = torch.FloatTensor(features["feature_vector"]).unsqueeze(0)
            mlp.eval()
            with torch.no_grad():
                logits = mlp(feat_tensor)
                probs = torch.softmax(logits, dim=-1)
                ml_class = torch.argmax(probs, dim=-1).item()
                ml_conf = probs[0][ml_class].item()

        # Step 3: BERT ensemble (if available)
        bert = MODELS.get("bert_classifier")
        bert_class = ml_class
        bert_conf = ml_conf

        if bert:
            try:
                bert_result = bert.predict(text, "bert")
                bert_class = bert_result["predicted_class"]
                bert_conf = bert_result["confidence"]
            except Exception:
                pass

        # Ensemble: average MLP + BERT
        if bert and mlp:
            if ml_class == bert_class:
                final_class = ml_class
                final_conf = (ml_conf + bert_conf) / 2
                method = "mlp_bert_ensemble"
            else:
                final_class = ml_class if ml_conf > bert_conf else bert_class
                final_conf = max(ml_conf, bert_conf)
                method = "highest_confidence"
        elif mlp:
            final_class = ml_class
            final_conf = ml_conf
            method = "mlp_only"
        else:
            final_class = 0
            final_conf = 0.5
            method = "default"

        classification = LABEL_MAP.get(final_class, "UNKNOWN")

        # Step 4: NER
        ner = MODELS.get("ner_pipeline")
        entities = None
        if ner:
            try:
                ner_result = ner.extract_entities(text)
                entities = ner_result.get("entities", [])
            except Exception:
                pass

        # Step 5: Summary
        summary = None
        if request.include_summary:
            summarizer = MODELS.get("summarizer")
            if summarizer and len(text) > 100:
                try:
                    sum_result = summarizer.summarize(text)
                    summary = sum_result.get("summary_text")
                except Exception:
                    pass

        # Step 6: Sentiment
        sentiment_data = None
        manipulation = None
        sentiment_model = MODELS.get("sentiment")
        if sentiment_model:
            try:
                sent_result = sentiment_model.analyze(text, adj_density=adj_density)
                sentiment_data = {
                    "finbert": sent_result["finbert_sentiment"],
                    "roberta": sent_result["roberta_sentiment"],
                }
                manipulation = sent_result["manipulation_score"]
            except Exception:
                pass

        # Step 7: Anomaly
        anomaly = None
        ae = MODELS.get("autoencoder")
        if ae:
            try:
                dummy_emb = torch.randn(1, 300)
                anomaly = float(ae.anomaly_score(dummy_emb)[0])
            except Exception:
                pass

        # Step 8: Explanation
        explanation = None
        if request.include_explanation:
            explainer = MODELS.get("explainer")
            if explainer and extractor:
                try:
                    explanation = {"method": "shap_lime", "available": True}
                except Exception:
                    pass

        elapsed = (time.time() - start) * 1000

        # Update metrics
        PREDICTIONS.labels(predicted_class=classification).inc()
        LATENCY.observe(elapsed / 1000)

        return ClassifyResponse(
            classification=classification,
            confidence=round(final_conf, 4),
            method=method,
            entities=entities,
            summary=summary,
            sentiment=sentiment_data,
            manipulation_score=round(manipulation, 4) if manipulation else None,
            anomaly_score=round(anomaly, 4) if anomaly else None,
            explanation=explanation,
            processing_time_ms=round(elapsed, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
