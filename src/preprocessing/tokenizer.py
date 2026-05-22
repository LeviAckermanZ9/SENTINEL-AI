"""
SENTINEL-AI — Tokenizer Module
Covers: CSR322 Unit I (NLP pipeline, tokenization)

Provides multiple tokenization strategies:
- NLTK word_tokenize (default)
- NLTK sentence tokenize
- Whitespace tokenizer
- Character-level tokenizer
- Vocabulary builder for deep learning models
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize


class Tokenizer:
    """
    Multi-strategy tokenizer with vocabulary management for SENTINEL-AI.

    Supports:
        - Word-level tokenization (NLTK)
        - Sentence-level tokenization (NLTK)
        - Whitespace tokenization
        - Character-level tokenization
        - Vocabulary building with frequency cutoffs
        - Token-to-index and index-to-token mappings

    Usage:
        tokenizer = Tokenizer(max_vocab_size=30000)
        tokenizer.build_vocab(corpus)
        indices = tokenizer.encode("Some text to encode")
        text = tokenizer.decode(indices)
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(
        self,
        max_vocab_size: int = 30000,
        min_frequency: int = 2,
        lowercase: bool = True,
    ):
        """
        Initialize the Tokenizer.

        Args:
            max_vocab_size: Maximum vocabulary size (including special tokens).
            min_frequency: Minimum token frequency to include in vocabulary.
            lowercase: Whether to lowercase text before tokenizing.
        """
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase

        # Initialize vocabulary with special tokens
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.token_counts: Counter = Counter()
        self.vocab_built = False

    def word_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words using NLTK word_tokenize.

        Args:
            text: Input text string.

        Returns:
            List of word tokens.
        """
        if self.lowercase:
            text = text.lower()
        return word_tokenize(text)

    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using NLTK sent_tokenize.

        Args:
            text: Input text string.

        Returns:
            List of sentence strings.
        """
        return sent_tokenize(text)

    def whitespace_tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace-based tokenization.

        Args:
            text: Input text string.

        Returns:
            List of whitespace-separated tokens.
        """
        if self.lowercase:
            text = text.lower()
        return text.split()

    def char_tokenize(self, text: str) -> List[str]:
        """
        Character-level tokenization.

        Args:
            text: Input text string.

        Returns:
            List of individual characters.
        """
        if self.lowercase:
            text = text.lower()
        return list(text)

    def build_vocab(self, corpus: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from a corpus of texts.

        Counts all word frequencies across the corpus, filters by min_frequency,
        and creates token-to-index / index-to-token mappings.

        Args:
            corpus: List of text strings to build vocabulary from.

        Returns:
            token2idx mapping dictionary.
        """
        # Count all tokens
        self.token_counts = Counter()
        for text in corpus:
            tokens = self.word_tokenize(text)
            self.token_counts.update(tokens)

        # Filter by minimum frequency
        filtered = {
            token: count
            for token, count in self.token_counts.items()
            if count >= self.min_frequency
        }

        # Sort by frequency (descending) and limit to max_vocab_size
        sorted_tokens = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        max_tokens = self.max_vocab_size - len(self.special_tokens)
        sorted_tokens = sorted_tokens[:max_tokens]

        # Build mappings
        self.token2idx = {}
        self.idx2token = {}

        for idx, token in enumerate(self.special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        for i, (token, _) in enumerate(sorted_tokens):
            idx = i + len(self.special_tokens)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        self.vocab_built = True
        return self.token2idx

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """
        Encode text into a list of token indices.

        Args:
            text: Input text to encode.
            max_length: Optional max sequence length (truncates or pads).
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of integer token indices.
        """
        if not self.vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")

        tokens = self.word_tokenize(text)
        unk_idx = self.token2idx[self.UNK_TOKEN]

        indices = [self.token2idx.get(token, unk_idx) for token in tokens]

        # Add special tokens if requested
        if add_special_tokens:
            bos_idx = self.token2idx[self.BOS_TOKEN]
            eos_idx = self.token2idx[self.EOS_TOKEN]
            indices = [bos_idx] + indices + [eos_idx]

        # Handle max_length: truncate or pad
        if max_length is not None:
            pad_idx = self.token2idx[self.PAD_TOKEN]
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [pad_idx] * (max_length - len(indices))

        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token indices back into text.

        Args:
            indices: List of integer token indices.
            skip_special_tokens: Whether to skip PAD/UNK/BOS/EOS in output.

        Returns:
            Decoded text string.
        """
        if not self.vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")

        special_set = set(self.special_tokens) if skip_special_tokens else set()
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, self.UNK_TOKEN)
            if token not in special_set:
                tokens.append(token)
        return " ".join(tokens)

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of input text strings.
            max_length: Optional max sequence length.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of encoded index lists.
        """
        return [
            self.encode(text, max_length=max_length, add_special_tokens=add_special_tokens)
            for text in texts
        ]

    @property
    def vocab_size(self) -> int:
        """Return current vocabulary size."""
        return len(self.token2idx)

    @property
    def pad_token_id(self) -> int:
        """Return PAD token index."""
        return self.token2idx.get(self.PAD_TOKEN, 0)

    @property
    def unk_token_id(self) -> int:
        """Return UNK token index."""
        return self.token2idx.get(self.UNK_TOKEN, 1)

    def get_token_frequency(self, token: str) -> int:
        """Return frequency count for a specific token."""
        if self.lowercase:
            token = token.lower()
        return self.token_counts.get(token, 0)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        status = "built" if self.vocab_built else "not built"
        return f"Tokenizer(vocab_size={self.vocab_size}, status={status})"
