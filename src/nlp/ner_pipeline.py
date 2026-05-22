"""
SENTINEL-AI — NER Pipeline
Covers: CSR322 Unit VI (NER, applications)

Combines HuggingFace BERT-NER + spaCy dependency parsing for entity
extraction and claim triplet building. All models lazy-loaded.
"""

from typing import Any, Dict, List, Optional, Tuple


class NERPipeline:
    """
    Named Entity Recognition combining transformer NER + spaCy parsing.

    Models (lazy-loaded):
        - dslim/bert-base-NER (HuggingFace, aggregation_strategy='simple')
        - en_core_web_sm (spaCy, for dependency-based claim triplets)

    Usage:
        ner = NERPipeline()
        result = ner.extract_entities("Elon Musk announced Tesla opens in Berlin.")
        # result = {entities: [...], spacy_entities: [...], claim_triplets: [...]}
    """

    def __init__(self, ner_model: str = "dslim/bert-base-NER"):
        self._ner_model_name = ner_model
        self._ner_pipeline = None
        self._nlp = None

    def _load_ner(self):
        """Lazy-load HuggingFace NER pipeline."""
        if self._ner_pipeline is None:
            from transformers import pipeline

            self._ner_pipeline = pipeline(
                "ner",
                model=self._ner_model_name,
                aggregation_strategy="simple",
            )
        return self._ner_pipeline

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
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract_hf_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using HuggingFace BERT-NER.

        Returns list of dicts with: entity_group (PER/ORG/LOC/MISC),
        word, score, start, end.
        """
        pipe = self._load_ner()
        raw_entities = pipe(text)

        entities = []
        for ent in raw_entities:
            entities.append(
                {
                    "entity_group": ent["entity_group"],
                    "word": ent["word"],
                    "score": round(float(ent["score"]), 4),
                    "start": ent["start"],
                    "end": ent["end"],
                }
            )

        return entities

    def extract_spacy_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using spaCy NER.

        Returns list of dicts with: text, label, start_char, end_char.
        """
        nlp = self._load_spacy()
        doc = nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

        return entities

    def extract_claim_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Build claim triplets from dependency parsing.

        Walks ROOT verbs → collects nsubj + dobj/attr → (subject, verb, object).
        """
        nlp = self._load_spacy()
        doc = nlp(text)

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

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Full entity extraction: HuggingFace NER + spaCy NER + claim triplets.

        Args:
            text: Input text.

        Returns:
            Dict with:
                - entities: HuggingFace BERT-NER results (PER/ORG/LOC/MISC)
                - spacy_entities: spaCy NER results
                - claim_triplets: (subject, verb, object) tuples
                - entity_count: total HF entities found
                - unique_entity_types: set of entity types found
        """
        hf_entities = self.extract_hf_entities(text)
        spacy_entities = self.extract_spacy_entities(text)
        claim_triplets = self.extract_claim_triplets(text)

        entity_types = set(e["entity_group"] for e in hf_entities)

        return {
            "entities": hf_entities,
            "spacy_entities": spacy_entities,
            "claim_triplets": claim_triplets,
            "entity_count": len(hf_entities),
            "unique_entity_types": list(entity_types),
        }

    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract entities from a batch of texts."""
        return [self.extract_entities(text) for text in texts]
