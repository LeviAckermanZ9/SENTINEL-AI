"""
SENTINEL-AI — Sentiment Analyzer
Covers: CSR322 Unit VI (Sentiment analysis, applications)

Dual-model sentiment: FinBERT (financial/news) + Twitter-RoBERTa (social media).
Computes manipulation_score = 0.6 * sentiment_extremity + 0.4 * min(adj_density*5, 1.0).
All models lazy-loaded.
"""

from typing import Any, Dict, List, Optional


class SentimentAnalyzer:
    """
    Dual-model sentiment analyzer with manipulation scoring.

    Models (lazy-loaded):
        - ProsusAI/finbert: financial/news sentiment (positive/negative/neutral)
        - cardiffnlp/twitter-roberta-base-sentiment: social media sentiment

    Manipulation score measures how likely text is manipulative:
        manipulation_score = 0.6 * sentiment_extremity + 0.4 * min(adj_density * 5, 1.0)

    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("The stock crashed horribly!", adj_density=0.15)
    """

    def __init__(self):
        self._finbert_pipeline = None
        self._roberta_pipeline = None

    def _load_finbert(self):
        """Lazy-load FinBERT sentiment pipeline."""
        if self._finbert_pipeline is None:
            from transformers import pipeline

            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
            )
        return self._finbert_pipeline

    def _load_roberta(self):
        """Lazy-load Twitter-RoBERTa sentiment pipeline."""
        if self._roberta_pipeline is None:
            from transformers import pipeline

            self._roberta_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            )
        return self._roberta_pipeline

    def get_finbert_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using FinBERT (financial/news domain).

        Returns:
            Dict with label (positive/negative/neutral), score.
        """
        pipe = self._load_finbert()
        result = pipe(text, truncation=True, max_length=512)

        return {
            "label": result[0]["label"],
            "score": round(float(result[0]["score"]), 4),
        }

    def get_roberta_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using Twitter-RoBERTa (social media domain).

        Returns:
            Dict with label, score.
        """
        pipe = self._load_roberta()
        result = pipe(text, truncation=True, max_length=512)

        return {
            "label": result[0]["label"],
            "score": round(float(result[0]["score"]), 4),
        }

    def compute_sentiment_extremity(self, sentiment_result: Dict[str, Any]) -> float:
        """
        Compute how extreme a sentiment prediction is.

        Extremity = confidence when sentiment is positive or negative.
        Neutral sentiment → low extremity.

        Returns:
            Float between 0.0 and 1.0.
        """
        label = sentiment_result["label"].lower()
        score = sentiment_result["score"]

        if label in ("neutral",):
            return 1.0 - score  # neutral with high conf → low extremity
        else:
            return score  # positive/negative with high conf → high extremity

    def compute_manipulation_score(
        self,
        sentiment_extremity: float,
        adj_density: float = 0.0,
    ) -> float:
        """
        Compute manipulation score.

        Formula: 0.6 * sentiment_extremity + 0.4 * min(adj_density * 5, 1.0)

        Args:
            sentiment_extremity: How extreme the sentiment is (0-1).
            adj_density: Adjective density from POS parsing.

        Returns:
            Manipulation score between 0.0 and 1.0.
        """
        adj_component = min(adj_density * 5, 1.0)
        return 0.6 * sentiment_extremity + 0.4 * adj_component

    def analyze(
        self,
        text: str,
        adj_density: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Full sentiment analysis with manipulation scoring.

        Args:
            text: Input text.
            adj_density: Adjective density from POS parsing (0-1).

        Returns:
            Dict with:
                - finbert_sentiment: {label, score}
                - roberta_sentiment: {label, score}
                - sentiment_extremity: float
                - manipulation_score: float
                - is_potentially_manipulative: bool (score > 0.6)
        """
        finbert = self.get_finbert_sentiment(text)
        roberta = self.get_roberta_sentiment(text)

        # Use FinBERT for extremity (primary model for news)
        extremity = self.compute_sentiment_extremity(finbert)
        manipulation = self.compute_manipulation_score(extremity, adj_density)

        return {
            "finbert_sentiment": finbert,
            "roberta_sentiment": roberta,
            "sentiment_extremity": round(extremity, 4),
            "manipulation_score": round(manipulation, 4),
            "is_potentially_manipulative": manipulation > 0.6,
        }

    def analyze_batch(
        self,
        texts: List[str],
        adj_densities: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts."""
        if adj_densities is None:
            adj_densities = [0.0] * len(texts)
        return [self.analyze(text, adj) for text, adj in zip(texts, adj_densities)]
