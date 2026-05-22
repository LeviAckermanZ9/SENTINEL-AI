"""
SENTINEL-AI — Claim Extractor
Covers: CSR322 CO5 (RAG system design)

Extracts primary factual claims from text using zero-shot scoring
and builds structured claim triplets for retrieval queries.
All models lazy-loaded.
"""

from typing import Any, Dict, List, Optional, Tuple


class ClaimExtractor:
    """
    Extract and structure factual claims from input text.

    Uses facebook/bart-large-mnli zero-shot to score each sentence
    on 'factual claim' → returns highest-scoring sentence.
    Then uses spaCy to extract (subject, verb, object) triplets.

    Usage:
        extractor = ClaimExtractor()
        claim = extractor.extract_primary_claim("Some multi-sentence text.")
        triplet = extractor.extract_claim_triplet("Scientists confirmed the finding.")
    """

    def __init__(self):
        self._zero_shot = None
        self._nlp = None

    def _load_zero_shot(self):
        """Lazy-load zero-shot pipeline."""
        if self._zero_shot is None:
            from transformers import pipeline
            self._zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
            )
        return self._zero_shot

    def _load_spacy(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess, sys
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        nlp = self._load_spacy()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        return sentences if sentences else [text.strip()]

    def extract_primary_claim(self, text: str) -> Dict[str, Any]:
        """
        Extract the primary factual claim from text.

        Splits into sentences, scores each with zero-shot on 'factual claim'
        label, returns highest-scoring sentence.

        Args:
            text: Input text (may be multi-sentence).

        Returns:
            Dict with:
                - primary_claim: highest-scoring sentence
                - claim_score: confidence score
                - all_sentences: scored sentence list
                - num_sentences: total sentences found
        """
        sentences = self._split_sentences(text)

        if len(sentences) == 1:
            return {
                "primary_claim": sentences[0],
                "claim_score": 1.0,
                "all_sentences": [{"sentence": sentences[0], "score": 1.0}],
                "num_sentences": 1,
            }

        pipe = self._load_zero_shot()
        scored = []

        for sent in sentences:
            result = pipe(sent, candidate_labels=["factual claim", "opinion", "question"])
            claim_idx = result["labels"].index("factual claim")
            score = result["scores"][claim_idx]
            scored.append({"sentence": sent, "score": round(score, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "primary_claim": scored[0]["sentence"],
            "claim_score": scored[0]["score"],
            "all_sentences": scored,
            "num_sentences": len(scored),
        }

    def _extract_triplet_from_sentence(self, text: str) -> Dict[str, str]:
        """Extract (subject, verb, object) from a single sentence via spaCy."""
        nlp = self._load_spacy()
        doc = nlp(text)

        subject = ""
        verb = ""
        obj = ""

        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb = token.text

                # Subjects
                for child in token.lefts:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        compound = " ".join(
                            [c.text for c in child.lefts if c.dep_ == "compound"]
                            + [child.text]
                        )
                        subject = compound
                        break

                # Objects
                for child in token.rights:
                    if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                        compound = " ".join(
                            [c.text for c in child.lefts if c.dep_ in ("compound", "amod")]
                            + [child.text]
                        )
                        obj = compound
                        break
                    elif child.dep_ == "prep":
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                obj = f"{child.text} {gc.text}"
                                break
                        if obj:
                            break

                break  # Use first ROOT verb

        return {"subject": subject, "verb": verb, "object": obj}

    def extract_claim_triplet(self, text: str) -> Dict[str, Any]:
        """
        Extract primary claim and its structured triplet.

        Args:
            text: Input text.

        Returns:
            Dict with:
                - primary_claim: extracted claim sentence
                - subject: claim subject
                - verb: claim verb
                - object: claim object
                - retrieval_query: formatted query for retriever
        """
        # Get primary claim
        claim_result = self.extract_primary_claim(text)
        primary_claim = claim_result["primary_claim"]

        # Extract triplet
        triplet = self._extract_triplet_from_sentence(primary_claim)

        # Build retrieval query
        parts = [triplet["subject"], triplet["verb"], triplet["object"]]
        query_parts = [p for p in parts if p]
        retrieval_query = " ".join(query_parts) if query_parts else primary_claim

        return {
            "primary_claim": primary_claim,
            "subject": triplet["subject"],
            "verb": triplet["verb"],
            "object": triplet["object"],
            "retrieval_query": retrieval_query,
        }
