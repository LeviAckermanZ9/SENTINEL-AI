"""
SENTINEL-AI — POS Parser Module
Covers: CSR322 Unit II (POS tagging, dependency parsing, regex, FSA)

Provides spaCy-based POS tagging, dependency parsing, claim triplets,
adjective metrics, and regex FSA for temporal patterns.
"""

import re
from typing import Any, Dict, List, Tuple

import spacy


class PosParser:
    """SpaCy POS tagger + dependency parser for fake news linguistic analysis."""

    TEMPORAL_PATTERNS = [
        re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
        re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b"),
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:yesterday|today|tomorrow|tonight|"
            r"last\s+(?:week|month|year|night)|"
            r"next\s+(?:week|month|year)|"
            r"this\s+(?:week|month|year|morning|afternoon|evening)|"
            r"\d+\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\s+ago)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.nlp = spacy.load(model_name)

    def get_pos_tags(self, doc) -> List[Tuple[str, str]]:
        """Extract (token, POS) tuples."""
        return [(token.text, token.pos_) for token in doc]

    def get_dep_triplets(self, doc) -> List[Tuple[str, str, str]]:
        """Extract (subject, verb, object) triplets from dependency parse."""
        triplets = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb = token.text
                subjects = []
                for child in token.lefts:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        compound = " ".join(
                            [c.text for c in child.lefts if c.dep_ == "compound"]
                            + [child.text]
                        )
                        subjects.append(compound)
                objects = []
                for child in token.rights:
                    if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                        compound = " ".join(
                            [
                                c.text
                                for c in child.lefts
                                if c.dep_ in ("compound", "amod")
                            ]
                            + [child.text]
                        )
                        objects.append(compound)
                    elif child.dep_ == "prep":
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                objects.append(f"{child.text} {gc.text}")
                for subj in subjects:
                    if objects:
                        for obj in objects:
                            triplets.append((subj, verb, obj))
                    else:
                        triplets.append((subj, verb, ""))
        return triplets

    def compute_adjective_density(self, doc) -> float:
        """ADJ_count / total_tokens."""
        total = len(doc)
        if total == 0:
            return 0.0
        return sum(1 for t in doc if t.pos_ == "ADJ") / total

    def compute_adj_noun_ratio(self, doc) -> float:
        """Adjective-to-noun ratio."""
        nouns = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
        adjs = sum(1 for t in doc if t.pos_ == "ADJ")
        return adjs / nouns if nouns > 0 else 0.0

    def find_temporal_patterns(self, text: str) -> List[str]:
        """Find date/temporal expressions via regex FSA."""
        matches = []
        for pattern in self.TEMPORAL_PATTERNS:
            matches.extend(pattern.findall(text))
        return matches

    def get_pos_distribution(self, doc) -> Dict[str, int]:
        """POS tag frequency distribution."""
        dist: Dict[str, int] = {}
        for token in doc:
            dist[token.pos_] = dist.get(token.pos_, 0) + 1
        return dist

    def parse(self, text: str) -> Dict[str, Any]:
        """Run full POS parsing pipeline. Returns pos_tags, dep_triplets, metrics."""
        if not text or not isinstance(text, str):
            return {
                "pos_tags": [],
                "dep_triplets": [],
                "adjective_density": 0.0,
                "adj_noun_ratio": 0.0,
                "temporal_matches": [],
                "pos_distribution": {},
                "num_sentences": 0,
                "num_tokens": 0,
            }
        doc = self.nlp(text)
        return {
            "pos_tags": self.get_pos_tags(doc),
            "dep_triplets": self.get_dep_triplets(doc),
            "adjective_density": self.compute_adjective_density(doc),
            "adj_noun_ratio": self.compute_adj_noun_ratio(doc),
            "temporal_matches": self.find_temporal_patterns(text),
            "pos_distribution": self.get_pos_distribution(doc),
            "num_sentences": len(list(doc.sents)),
            "num_tokens": len(doc),
        }

    def parse_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Parse a batch of texts using spaCy pipe."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=32):
            results.append(
                {
                    "pos_tags": self.get_pos_tags(doc),
                    "dep_triplets": self.get_dep_triplets(doc),
                    "adjective_density": self.compute_adjective_density(doc),
                    "adj_noun_ratio": self.compute_adj_noun_ratio(doc),
                    "temporal_matches": self.find_temporal_patterns(doc.text),
                    "pos_distribution": self.get_pos_distribution(doc),
                    "num_sentences": len(list(doc.sents)),
                    "num_tokens": len(doc),
                }
            )
        return results
