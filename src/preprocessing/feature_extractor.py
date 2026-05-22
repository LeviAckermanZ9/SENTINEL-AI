"""
SENTINEL-AI — Feature Extractor Module
Covers: CSR322 Unit III (TF-IDF, n-gram, clustering)

Provides TF-IDF vectorization, KMeans topic clustering, bigram perplexity proxy,
and final 504-dim feature vector assembly for the MLP classifier.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class FeatureExtractor:
    """
    TF-IDF + KMeans feature extraction pipeline for SENTINEL-AI.

    Produces a 504-dimensional feature vector:
        - 500 TF-IDF features (unigrams + bigrams)
        - 1 adjective density
        - 1 adjective/noun ratio
        - 1 bigram perplexity proxy
        - 1 topic cluster ID

    Usage:
        extractor = FeatureExtractor()
        extractor.fit(corpus)
        features = extractor.transform("Some text to analyze")
    """

    def __init__(
        self,
        max_features: int = 500,
        ngram_range: tuple = (1, 2),
        n_clusters: int = 8,
        random_state: int = 42,
    ):
        """
        Initialize the FeatureExtractor.

        Args:
            max_features: Max TF-IDF vocabulary size.
            ngram_range: N-gram range for TF-IDF (default: unigrams + bigrams).
            n_clusters: Number of KMeans topic clusters.
            random_state: Random seed for reproducibility.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.tfidf = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words="english",
            sublinear_tf=True,
        )
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.is_fitted = False

    def fit(self, corpus: List[str]) -> "FeatureExtractor":
        """
        Fit TF-IDF vectorizer and KMeans on a corpus.

        Args:
            corpus: List of text strings to train on.

        Returns:
            self (for method chaining).
        """
        tfidf_matrix = self.tfidf.fit_transform(corpus)
        self.kmeans.fit(tfidf_matrix.toarray())
        self.is_fitted = True
        return self

    def compute_bigram_perplexity_proxy(self, text: str) -> float:
        """
        Compute a bigram perplexity proxy score.

        Uses the ratio of unique bigrams to total bigrams as a proxy
        for lexical diversity / perplexity. Higher values indicate
        more diverse (potentially more natural) language.

        Args:
            text: Input text string.

        Returns:
            Float perplexity proxy score between 0.0 and 1.0.
        """
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        return len(unique_bigrams) / len(bigrams) if bigrams else 0.0

    def transform(
        self,
        text: str,
        adjective_density: float = 0.0,
        adj_noun_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Transform a single text into a 504-dim feature dictionary.

        Args:
            text: Input text string.
            adjective_density: Pre-computed adjective density from PosParser.
            adj_noun_ratio: Pre-computed adj/noun ratio from PosParser.

        Returns:
            Dictionary with:
                - tfidf_vector: numpy array (500-dim)
                - topic_cluster: int cluster assignment
                - bigram_perplexity_proxy: float
                - feature_vector: numpy array (504-dim, full concatenated vector)
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor not fitted. Call fit() first.")

        # TF-IDF features (500-dim)
        tfidf_vector = self.tfidf.transform([text]).toarray()[0]

        # Topic cluster
        topic_cluster = int(self.kmeans.predict(tfidf_vector.reshape(1, -1))[0])

        # Bigram perplexity proxy
        perplexity = self.compute_bigram_perplexity_proxy(text)

        # Assemble 504-dim feature vector
        extra_features = np.array(
            [
                adjective_density,
                adj_noun_ratio,
                perplexity,
                float(topic_cluster),
            ]
        )
        feature_vector = np.concatenate([tfidf_vector, extra_features])

        return {
            "tfidf_vector": tfidf_vector,
            "topic_cluster": topic_cluster,
            "bigram_perplexity_proxy": perplexity,
            "feature_vector": feature_vector,
        }

    def transform_batch(
        self,
        texts: List[str],
        adjective_densities: Optional[List[float]] = None,
        adj_noun_ratios: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Transform a batch of texts into feature matrices.

        Args:
            texts: List of text strings.
            adjective_densities: List of adjective density values.
            adj_noun_ratios: List of adj/noun ratio values.

        Returns:
            Dictionary with:
                - tfidf_matrix: numpy array (N, 500)
                - topic_clusters: numpy array (N,)
                - feature_matrix: numpy array (N, 504)
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor not fitted. Call fit() first.")

        if adjective_densities is None:
            adjective_densities = [0.0] * len(texts)
        if adj_noun_ratios is None:
            adj_noun_ratios = [0.0] * len(texts)

        tfidf_matrix = self.tfidf.transform(texts).toarray()
        topic_clusters = self.kmeans.predict(tfidf_matrix)
        perplexities = [self.compute_bigram_perplexity_proxy(t) for t in texts]

        extra = np.column_stack(
            [
                adjective_densities,
                adj_noun_ratios,
                perplexities,
                topic_clusters.astype(float),
            ]
        )
        feature_matrix = np.hstack([tfidf_matrix, extra])

        return {
            "tfidf_matrix": tfidf_matrix,
            "topic_clusters": topic_clusters,
            "feature_matrix": feature_matrix,
        }

    def get_feature_names(self) -> List[str]:
        """Return list of all 504 feature names."""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor not fitted. Call fit() first.")
        tfidf_names = list(self.tfidf.get_feature_names_out())
        extra_names = [
            "adjective_density",
            "adj_noun_ratio",
            "bigram_perplexity",
            "topic_cluster",
        ]
        return tfidf_names + extra_names

    @property
    def feature_dim(self) -> int:
        """Total feature dimension (504)."""
        return self.max_features + 4

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"FeatureExtractor(tfidf_features={self.max_features}, "
            f"clusters={self.n_clusters}, status={status})"
        )
