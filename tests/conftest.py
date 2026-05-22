"""
SENTINEL-AI Test Configuration
Shared fixtures and utilities for the test suite.
"""

import os
import sys

import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_text():
    """Sample text for testing the analysis pipeline."""
    return (
        "Scientists at the World Health Organization confirmed today that "
        "5G towers do not spread COVID-19. The claim, which circulated widely "
        "on social media platforms including Twitter and Facebook, has been "
        "debunked by multiple independent research groups across 15 countries."
    )


@pytest.fixture
def sample_fake_text():
    """Sample fake news text for testing."""
    return (
        "BREAKING: Government officials secretly admit that vaccines contain "
        "mind-control microchips designed by Bill Gates. Sources within the CDC "
        "have leaked classified documents proving a global conspiracy to track "
        "every citizen through 5G-enabled nanobots injected during vaccination."
    )


@pytest.fixture
def sample_satire_text():
    """Sample satire text for testing."""
    return (
        "In a stunning development, local man discovers that his houseplant "
        "has been filing taxes independently for the past three years. The fern, "
        "named Gerald, reportedly claimed several dependents and received a "
        "substantial refund from the IRS."
    )


@pytest.fixture
def sample_short_text():
    """Text below minimum length for validation testing."""
    return "Too short text."


@pytest.fixture
def sample_corpus():
    """Corpus of texts for training/testing feature extractors."""
    return [
        "The president signed a new executive order on climate change policy.",
        "Scientists discovered a new species of deep-sea fish near hydrothermal vents.",
        "The stock market experienced significant volatility amid trade war concerns.",
        "A new study reveals that regular exercise improves mental health outcomes.",
        "Technology companies reported record earnings in the fourth quarter.",
        "The United Nations held an emergency session on the humanitarian crisis.",
        "Researchers developed a breakthrough treatment for antibiotic-resistant bacteria.",
        "The central bank announced an unexpected interest rate cut to stimulate growth.",
    ]


@pytest.fixture
def label_map():
    """Standard label mapping for SENTINEL-AI classification."""
    return {0: "REAL", 1: "FAKE", 2: "SATIRE", 3: "SPAM"}
