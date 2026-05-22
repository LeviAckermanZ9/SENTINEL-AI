"""
SENTINEL-AI — RAG Chain
Covers: CSR322 CO5, CO6 (RAG design, production trade-offs)

LangChain LCEL chain: prompt | llm | parser.
Supports local Flan-T5-XL or Mistral API via env toggle.
FactCheckOutputParser with regex parsing + safe defaults.
"""

import os
import re
from typing import Any, Dict, Optional


class FactCheckOutputParser:
    """
    Parse LLM fact-check output into structured fields.

    Expected format:
        VERDICT: SUPPORTED / CONTRADICTED / UNVERIFIABLE
        CONFIDENCE: HIGH / MEDIUM / LOW
        REASONING: 2-3 sentences
        KEY_SOURCES: comma-separated list

    Defaults to UNVERIFIABLE/LOW on malformed output.
    """

    VALID_VERDICTS = {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}
    VALID_CONFIDENCES = {"HIGH", "MEDIUM", "LOW"}

    def parse(self, text: str) -> Dict[str, str]:
        """
        Parse LLM output text into structured dict.

        Args:
            text: Raw LLM output string.

        Returns:
            Dict with verdict, confidence, reasoning, key_sources.
        """
        if not text or not isinstance(text, str):
            return self._defaults()

        result = {}

        # VERDICT
        verdict_match = re.search(
            r"VERDICT\s*:\s*(SUPPORTED|CONTRADICTED|UNVERIFIABLE)", text, re.IGNORECASE
        )
        if verdict_match:
            result["verdict"] = verdict_match.group(1).upper()
        else:
            result["verdict"] = "UNVERIFIABLE"

        # Validate verdict
        if result["verdict"] not in self.VALID_VERDICTS:
            result["verdict"] = "UNVERIFIABLE"

        # CONFIDENCE
        conf_match = re.search(
            r"CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE
        )
        if conf_match:
            result["confidence"] = conf_match.group(1).upper()
        else:
            result["confidence"] = "LOW"

        if result["confidence"] not in self.VALID_CONFIDENCES:
            result["confidence"] = "LOW"

        # REASONING
        reasoning_match = re.search(
            r"REASONING\s*:\s*(.+?)(?=KEY_SOURCES|$)", text, re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            result["reasoning"] = text.strip()[:500]

        # KEY_SOURCES
        sources_match = re.search(
            r"KEY_SOURCES\s*:\s*(.+?)$", text, re.IGNORECASE | re.DOTALL
        )
        if sources_match:
            result["key_sources"] = sources_match.group(1).strip()
        else:
            result["key_sources"] = ""

        return result

    def _defaults(self) -> Dict[str, str]:
        """Return safe default values for malformed output."""
        return {
            "verdict": "UNVERIFIABLE",
            "confidence": "LOW",
            "reasoning": "Unable to parse LLM output.",
            "key_sources": "",
        }


# Prompt template text
FACT_CHECK_PROMPT_TEMPLATE = """You are a professional fact-checker working for a reputable news verification organization. Your task is to assess the veracity of a claim based on the provided evidence documents.

EVIDENCE DOCUMENTS:
{context_docs}

CLAIM TO ASSESS:
{claim}

Based on the evidence provided, respond with the following format exactly:
VERDICT: [SUPPORTED / CONTRADICTED / UNVERIFIABLE]
CONFIDENCE: [HIGH / MEDIUM / LOW]
REASONING: [2-3 sentences explaining your assessment]
KEY_SOURCES: [comma-separated list of key source references]"""


class SentinelRAGChain:
    """
    LangChain-based RAG chain for fact-checking.

    Supports two LLM backends (env-controlled):
        - Local: google/flan-t5-xl via HuggingFace pipeline
        - API: mistralai/Mistral-7B-Instruct-v0.3 via HuggingFaceEndpoint

    Chain: prompt | llm | parser (LangChain LCEL).

    Usage:
        chain = SentinelRAGChain(use_api=False)
        result = chain.run(claim="5G causes COVID", evidence_docs=["doc1", "doc2"])
    """

    def __init__(self, use_api: bool = False):
        self.use_api = (
            use_api or os.environ.get("USE_API_LLM", "false").lower() == "true"
        )
        self.parser = FactCheckOutputParser()
        self._llm = None
        self._chain = None

    def _load_local_llm(self):
        """Load local Flan-T5-XL via HuggingFace pipeline."""
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline as hf_pipeline
        import torch

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        pipe = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-xl",
            device_map="auto",
            torch_dtype=dtype,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )

        return HuggingFacePipeline(pipeline=pipe)

    def _load_api_llm(self):
        """Load Mistral via HuggingFaceEndpoint (requires HF_TOKEN)."""
        from langchain_huggingface import HuggingFaceEndpoint

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            raise ValueError("HF_TOKEN env var required for API LLM mode.")

        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            huggingfacehub_api_token=hf_token,
            max_new_tokens=256,
            temperature=0.1,
        )

    def _load_llm(self):
        """Load the appropriate LLM backend."""
        if self._llm is None:
            if self.use_api:
                self._llm = self._load_api_llm()
            else:
                self._llm = self._load_local_llm()
        return self._llm

    def _build_chain(self):
        """Build LangChain LCEL chain: prompt | llm | parser."""
        if self._chain is None:
            from langchain.prompts import PromptTemplate

            prompt = PromptTemplate(
                input_variables=["context_docs", "claim"],
                template=FACT_CHECK_PROMPT_TEMPLATE,
            )

            llm = self._load_llm()

            # LCEL chain
            self._chain = prompt | llm
        return self._chain

    def run(
        self,
        claim: str,
        evidence_docs: list,
    ) -> Dict[str, Any]:
        """
        Run the fact-check chain.

        Args:
            claim: Claim text to verify.
            evidence_docs: List of evidence document strings.

        Returns:
            Parsed dict with verdict, confidence, reasoning, key_sources.
        """
        chain = self._build_chain()

        # Format evidence
        context = "\n\n".join(
            f"[Doc {i+1}]: {doc}" for i, doc in enumerate(evidence_docs)
        )

        try:
            raw_output = chain.invoke(
                {
                    "context_docs": context,
                    "claim": claim,
                }
            )

            # Handle different output types
            if hasattr(raw_output, "content"):
                output_text = raw_output.content
            elif isinstance(raw_output, str):
                output_text = raw_output
            else:
                output_text = str(raw_output)

            result = self.parser.parse(output_text)
            result["raw_output"] = output_text

        except Exception as e:
            result = self.parser._defaults()
            result["raw_output"] = ""
            result["error"] = str(e)

        return result

    def run_without_llm(
        self,
        claim: str,
        evidence_docs: list,
    ) -> Dict[str, Any]:
        """
        Heuristic fact-check without LLM (fallback when no LLM available).

        Uses keyword matching against evidence documents.
        """
        claim_lower = claim.lower()
        support_count = 0
        contradict_count = 0

        contradict_keywords = [
            "false",
            "debunked",
            "incorrect",
            "myth",
            "hoax",
            "pants-fire",
            "misleading",
        ]
        support_keywords = [
            "true",
            "confirmed",
            "verified",
            "supports",
            "accurate",
            "correct",
        ]

        for doc in evidence_docs:
            doc_lower = doc.lower()
            for kw in contradict_keywords:
                if kw in doc_lower:
                    contradict_count += 1
            for kw in support_keywords:
                if kw in doc_lower:
                    support_count += 1

        if contradict_count > support_count and contradict_count >= 2:
            verdict = "CONTRADICTED"
            confidence = "MEDIUM"
        elif support_count > contradict_count and support_count >= 2:
            verdict = "SUPPORTED"
            confidence = "MEDIUM"
        else:
            verdict = "UNVERIFIABLE"
            confidence = "LOW"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": f"Heuristic analysis: {support_count} support signals, {contradict_count} contradict signals across {len(evidence_docs)} documents.",
            "key_sources": "",
            "raw_output": "",
        }
