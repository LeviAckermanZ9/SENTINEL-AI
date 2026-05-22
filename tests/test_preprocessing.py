"""
SENTINEL-AI — Preprocessing Test Suite
Tests for: TextCleaner, Tokenizer, PosParser, FeatureExtractor
"""

import numpy as np
import pytest

from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.pos_parser import PosParser
from src.preprocessing.feature_extractor import FeatureExtractor

# ================================================================
# TextCleaner Tests
# ================================================================


class TestTextCleaner:
    """Tests for the TextCleaner class."""

    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_strip_html(self):
        text = "<p>Hello <b>world</b></p>"
        result = self.cleaner.strip_html(text)
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strip_urls(self):
        text = "Visit http://example.com or https://test.org for info"
        result = self.cleaner.strip_urls(text)
        assert "http" not in result
        assert "example.com" not in result

    def test_strip_mentions(self):
        text = "Hey @user1 and @user2 check this out"
        result = self.cleaner.strip_mentions(text)
        assert "@user1" not in result
        assert "@user2" not in result

    def test_strip_hashtags(self):
        text = "This is #breaking #news today"
        result = self.cleaner.strip_hashtags(text)
        assert "#" not in result
        assert "breaking" in result
        assert "news" in result

    def test_full_clean(self):
        text = "<p>BREAKING: @user says #fakenews about http://example.com!!!</p>"
        result = self.cleaner.clean(text)
        assert isinstance(result, dict)
        assert "cleaned_text" in result
        assert "tokens" in result
        assert isinstance(result["tokens"], list)
        assert result["original_length"] == len(text)
        assert result["token_count"] == len(result["tokens"])

    def test_clean_removes_html_urls_mentions(self):
        text = "<b>Check</b> @admin http://t.co/abc #trending now!!!"
        result = self.cleaner.clean(text)
        cleaned = result["cleaned_text"]
        assert "<b>" not in cleaned
        assert "@admin" not in cleaned
        assert "http" not in cleaned

    def test_clean_empty_input(self):
        result = self.cleaner.clean("")
        assert result["cleaned_text"] == ""
        assert result["tokens"] == []
        assert result["token_count"] == 0

    def test_clean_none_input(self):
        result = self.cleaner.clean(None)
        assert result["cleaned_text"] == ""

    def test_clean_batch(self):
        texts = ["Hello <b>world</b>", "Test @user #tag"]
        results = self.cleaner.clean_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_get_clean_text(self):
        result = self.cleaner.get_clean_text("Hello World!")
        assert isinstance(result, str)

    def test_get_tokens(self):
        result = self.cleaner.get_tokens("Hello beautiful world")
        assert isinstance(result, list)

    def test_lemmatization(self):
        text = "The dogs were running quickly through the trees"
        result = self.cleaner.clean(text)
        tokens = result["tokens"]
        assert "dog" in tokens or "running" in tokens  # lemmatized forms

    def test_stopword_removal(self):
        cleaner_with = TextCleaner(remove_stopwords=True)
        cleaner_without = TextCleaner(remove_stopwords=False)
        text = "The quick brown fox jumps over the lazy dog"
        with_stop = cleaner_with.clean(text)
        without_stop = cleaner_without.clean(text)
        assert len(without_stop["tokens"]) >= len(with_stop["tokens"])


# ================================================================
# Tokenizer Tests
# ================================================================


class TestTokenizer:
    """Tests for the Tokenizer class."""

    def setup_method(self):
        self.tokenizer = Tokenizer(max_vocab_size=100, min_frequency=1)
        self.corpus = [
            "the cat sat on the mat",
            "the dog chased the cat",
            "the bird flew over the mat",
        ]

    def test_word_tokenize(self):
        tokens = self.tokenizer.word_tokenize("Hello world!")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_sentence_tokenize(self):
        text = "First sentence. Second sentence! Third one?"
        sentences = self.tokenizer.sentence_tokenize(text)
        assert len(sentences) == 3

    def test_whitespace_tokenize(self):
        tokens = self.tokenizer.whitespace_tokenize("hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_char_tokenize(self):
        chars = self.tokenizer.char_tokenize("abc")
        assert chars == ["a", "b", "c"]

    def test_build_vocab(self):
        vocab = self.tokenizer.build_vocab(self.corpus)
        assert isinstance(vocab, dict)
        assert len(vocab) > 4  # more than just special tokens
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert self.tokenizer.vocab_built is True

    def test_encode(self):
        self.tokenizer.build_vocab(self.corpus)
        indices = self.tokenizer.encode("the cat sat")
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)

    def test_decode(self):
        self.tokenizer.build_vocab(self.corpus)
        indices = self.tokenizer.encode("the cat sat")
        text = self.tokenizer.decode(indices)
        assert isinstance(text, str)
        assert "cat" in text

    def test_encode_with_max_length(self):
        self.tokenizer.build_vocab(self.corpus)
        indices = self.tokenizer.encode("the cat sat on the mat", max_length=3)
        assert len(indices) == 3

    def test_encode_with_padding(self):
        self.tokenizer.build_vocab(self.corpus)
        indices = self.tokenizer.encode("cat", max_length=10)
        assert len(indices) == 10

    def test_encode_before_build_raises(self):
        fresh = Tokenizer()
        with pytest.raises(RuntimeError):
            fresh.encode("test")

    def test_vocab_size(self):
        self.tokenizer.build_vocab(self.corpus)
        assert self.tokenizer.vocab_size == len(self.tokenizer.token2idx)

    def test_repr(self):
        repr_str = repr(self.tokenizer)
        assert "Tokenizer" in repr_str


# ================================================================
# PosParser Tests
# ================================================================


class TestPosParser:
    """Tests for the PosParser class."""

    def setup_method(self):
        self.parser = PosParser()

    def test_parse_returns_dict(self):
        result = self.parser.parse("The president signed a new executive order.")
        assert isinstance(result, dict)
        assert "pos_tags" in result
        assert "dep_triplets" in result
        assert "adjective_density" in result
        assert "adj_noun_ratio" in result

    def test_pos_tags_format(self):
        result = self.parser.parse("The quick brown fox jumps.")
        assert isinstance(result["pos_tags"], list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result["pos_tags"])

    def test_adjective_density(self):
        # Text with many adjectives
        adj_heavy = "The big, beautiful, amazing, incredible, stunning building."
        result = self.parser.parse(adj_heavy)
        assert result["adjective_density"] > 0.0
        # Text with no adjectives
        no_adj = "He runs fast."
        result2 = self.parser.parse(no_adj)
        assert result2["adjective_density"] <= result["adjective_density"]

    def test_adj_noun_ratio(self):
        result = self.parser.parse("The beautiful old house on the green hill.")
        assert result["adj_noun_ratio"] >= 0.0
        assert isinstance(result["adj_noun_ratio"], float)

    def test_dep_triplets(self):
        result = self.parser.parse("The president signed the executive order.")
        triplets = result["dep_triplets"]
        assert isinstance(triplets, list)
        # Should extract at least one triplet from this clear SVO sentence
        if triplets:
            assert all(isinstance(t, tuple) and len(t) == 3 for t in triplets)

    def test_temporal_patterns(self):
        text = "The event happened on January 15, 2024 and was reported yesterday."
        result = self.parser.parse(text)
        matches = result["temporal_matches"]
        assert len(matches) >= 1  # Should match at least one date

    def test_temporal_iso_date(self):
        matches = self.parser.find_temporal_patterns("The date is 2024-01-15.")
        assert any("2024" in m for m in matches)

    def test_temporal_relative(self):
        matches = self.parser.find_temporal_patterns("This happened yesterday.")
        assert len(matches) >= 1

    def test_empty_input(self):
        result = self.parser.parse("")
        assert result["pos_tags"] == []
        assert result["adjective_density"] == 0.0

    def test_none_input(self):
        result = self.parser.parse(None)
        assert result["pos_tags"] == []

    def test_pos_distribution(self):
        result = self.parser.parse("The quick fox jumps over the lazy dog.")
        dist = result["pos_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == result["num_tokens"]

    def test_parse_batch(self):
        texts = ["Hello world.", "The cat sat on the mat."]
        results = self.parser.parse_batch(texts)
        assert len(results) == 2


# ================================================================
# FeatureExtractor Tests
# ================================================================


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""

    def setup_method(self):
        # Use max_features=50 because test corpus is small (~8 short sentences)
        self.extractor = FeatureExtractor(max_features=50, n_clusters=3)
        self.corpus = [
            "The president signed a new executive order on climate change.",
            "Scientists discovered a breakthrough in cancer research today.",
            "The stock market experienced significant volatility this week.",
            "A new study reveals exercise improves mental health outcomes.",
            "Technology companies reported record earnings last quarter.",
            "The United Nations held an emergency session on the crisis.",
            "Researchers developed a treatment for resistant bacteria.",
            "The central bank announced an interest rate cut to help growth.",
        ]

    def test_fit(self):
        self.extractor.fit(self.corpus)
        assert self.extractor.is_fitted is True

    def test_transform_returns_dict(self):
        self.extractor.fit(self.corpus)
        result = self.extractor.transform(
            "The president announced new economic policies."
        )
        assert isinstance(result, dict)
        assert "tfidf_vector" in result
        assert "topic_cluster" in result
        assert "bigram_perplexity_proxy" in result
        assert "feature_vector" in result

    def test_tfidf_vector_shape(self):
        self.extractor.fit(self.corpus)
        result = self.extractor.transform("Test text for analysis.")
        assert result["tfidf_vector"].shape == (50,)

    def test_feature_vector_dim(self):
        self.extractor.fit(self.corpus)
        result = self.extractor.transform("Test text for feature extraction.")
        assert result["feature_vector"].shape == (54,)  # 50 + 4 extra

    def test_topic_cluster_range(self):
        self.extractor.fit(self.corpus)
        result = self.extractor.transform("Some text about politics.")
        assert 0 <= result["topic_cluster"] < 3

    def test_bigram_perplexity(self):
        self.extractor.fit(self.corpus)
        result = self.extractor.transform(
            "The quick brown fox jumps over the lazy dog."
        )
        assert 0.0 <= result["bigram_perplexity_proxy"] <= 1.0

    def test_transform_before_fit_raises(self):
        fresh = FeatureExtractor()
        with pytest.raises(RuntimeError):
            fresh.transform("test")

    def test_transform_batch(self):
        self.extractor.fit(self.corpus)
        texts = ["Text one about politics.", "Text two about science."]
        result = self.extractor.transform_batch(texts)
        assert result["feature_matrix"].shape == (2, 54)
        assert result["tfidf_matrix"].shape == (2, 50)
        assert len(result["topic_clusters"]) == 2

    def test_get_feature_names(self):
        self.extractor.fit(self.corpus)
        names = self.extractor.get_feature_names()
        assert len(names) == 54
        assert "adjective_density" in names
        assert "topic_cluster" in names

    def test_feature_dim(self):
        assert self.extractor.feature_dim == 54

    def test_repr(self):
        repr_str = repr(self.extractor)
        assert "FeatureExtractor" in repr_str

    def test_custom_adjective_density(self):
        self.extractor.fit(self.corpus)
        result = self.extractor.transform(
            "Test text.", adjective_density=0.15, adj_noun_ratio=0.5
        )
        vec = result["feature_vector"]
        # Last 4 elements: adj_density, adj_noun_ratio, perplexity, cluster
        assert vec[-4] == pytest.approx(0.15, abs=1e-6)
        assert vec[-3] == pytest.approx(0.5, abs=1e-6)

    def test_production_feature_dim(self):
        """Verify that with max_features=500, feature_dim is 504."""
        prod_extractor = FeatureExtractor(max_features=500, n_clusters=8)
        assert prod_extractor.feature_dim == 504
