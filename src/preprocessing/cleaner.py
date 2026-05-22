"""
SENTINEL-AI — Text Cleaner Module
Covers: CSR322 Unit I (NLP pipeline, tokenization)

Provides comprehensive text preprocessing for social media posts and news articles:
- HTML tag removal
- URL, mention, hashtag stripping
- Lowercasing, punctuation removal
- NLTK tokenization + WordNet lemmatization
"""

import re
import string
from typing import Dict, List, Tuple

import nltk

# Ensure required NLTK data is available
for resource in [
    "punkt",
    "punkt_tab",
    "wordnet",
    "omw-1.4",
    "stopwords",
    "averaged_perceptron_tagger",
]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextCleaner:
    """
    End-to-end text cleaning pipeline for social media / news text.

    Pipeline steps:
        1. Strip HTML tags
        2. Remove URLs
        3. Remove @mentions
        4. Remove #hashtags (keeps the word, strips the #)
        5. Lowercase
        6. Remove punctuation
        7. Tokenize (NLTK word_tokenize)
        8. Remove stopwords (optional)
        9. Lemmatize (WordNet)

    Usage:
        cleaner = TextCleaner()
        result = cleaner.clean("Some <b>HTML</b> text @user http://link.com")
        # result == {"cleaned_text": "...", "tokens": [...]}
    """

    # Compiled regex patterns for performance
    RE_HTML = re.compile(r"<[^>]+>")
    RE_URL = re.compile(r"https?://\S+|www\.\S+")
    RE_MENTION = re.compile(r"@\w+")
    RE_HASHTAG = re.compile(r"#(\w+)")  # captures word after #
    RE_MULTIPLE_SPACES = re.compile(r"\s+")
    RE_NUMBERS = re.compile(r"\b\d+\b")

    def __init__(self, remove_stopwords: bool = True, min_token_length: int = 2):
        """
        Initialize the TextCleaner.

        Args:
            remove_stopwords: Whether to filter English stopwords.
            min_token_length: Minimum token length to keep after cleaning.
        """
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()

    def strip_html(self, text: str) -> str:
        """Remove all HTML/XML tags from text."""
        return self.RE_HTML.sub("", text)

    def strip_urls(self, text: str) -> str:
        """Remove all URLs (http, https, www) from text."""
        return self.RE_URL.sub("", text)

    def strip_mentions(self, text: str) -> str:
        """Remove all @mentions from text."""
        return self.RE_MENTION.sub("", text)

    def strip_hashtags(self, text: str) -> str:
        """Replace #hashtag with just the word (remove # symbol)."""
        return self.RE_HASHTAG.sub(r"\1", text)

    def strip_punctuation(self, text: str) -> str:
        """Remove all punctuation characters."""
        return text.translate(str.maketrans("", "", string.punctuation))

    def strip_numbers(self, text: str) -> str:
        """Remove standalone numbers (keeps numbers within words)."""
        return self.RE_NUMBERS.sub("", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into single space."""
        return self.RE_MULTIPLE_SPACES.sub(" ", text).strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK word_tokenize."""
        return word_tokenize(text)

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize a list of tokens using WordNet lemmatizer."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by:
        - Removing stopwords (if enabled)
        - Removing tokens shorter than min_token_length
        - Keeping only alphabetic tokens
        """
        filtered = []
        for token in tokens:
            if not token.isalpha():
                continue
            if len(token) < self.min_token_length:
                continue
            if self.remove_stopwords and token in self.stop_words:
                continue
            filtered.append(token)
        return filtered

    def clean(self, text: str) -> Dict[str, object]:
        """
        Run the full cleaning pipeline on input text.

        Args:
            text: Raw input text (social media post, news article, etc.)

        Returns:
            Dictionary with:
                - cleaned_text (str): Fully cleaned and rejoined text
                - tokens (List[str]): List of cleaned, lemmatized tokens
                - original_length (int): Character count of original text
                - cleaned_length (int): Character count of cleaned text
                - token_count (int): Number of tokens after cleaning
        """
        if not text or not isinstance(text, str):
            return {
                "cleaned_text": "",
                "tokens": [],
                "original_length": 0,
                "cleaned_length": 0,
                "token_count": 0,
            }

        original_length = len(text)

        # Sequential cleaning steps
        text = self.strip_html(text)
        text = self.strip_urls(text)
        text = self.strip_mentions(text)
        text = self.strip_hashtags(text)
        text = text.lower()
        text = self.strip_punctuation(text)
        text = self.strip_numbers(text)
        text = self.normalize_whitespace(text)

        # Tokenize
        tokens = self.tokenize(text)

        # Filter and lemmatize
        tokens = self.filter_tokens(tokens)
        tokens = self.lemmatize_tokens(tokens)

        # Rejoin cleaned text
        cleaned_text = " ".join(tokens)

        return {
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "original_length": original_length,
            "cleaned_length": len(cleaned_text),
            "token_count": len(tokens),
        }

    def clean_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        """
        Clean a batch of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned result dictionaries.
        """
        return [self.clean(text) for text in texts]

    def get_clean_text(self, text: str) -> str:
        """Convenience method: returns only the cleaned text string."""
        return self.clean(text)["cleaned_text"]

    def get_tokens(self, text: str) -> List[str]:
        """Convenience method: returns only the token list."""
        return self.clean(text)["tokens"]
