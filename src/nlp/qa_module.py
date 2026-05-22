"""
SENTINEL-AI — QA Module
Covers: CSR322 Unit V (QA systems)

Standalone QA module wrapping deepset/roberta-base-squad2.
Lazy-loaded on first call.
"""

from typing import Any, Dict, List, Optional


class QAModule:
    """
    Extractive Question Answering using RoBERTa-SQuAD2.

    Model (lazy-loaded): deepset/roberta-base-squad2

    Usage:
        qa = QAModule()
        answer = qa.answer("Who is the president?", "The president is John Smith.")
    """

    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self._model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load QA pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "question-answering",
                model=self._model_name,
            )
        return self._pipeline

    def answer(
        self,
        question: str,
        context: str,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """
        Answer a question given a context passage.

        Args:
            question: Question to answer.
            context: Passage containing the answer.
            top_k: Number of top answers to return.

        Returns:
            Dict with answer, confidence, start, end.
        """
        pipe = self._load_pipeline()

        if top_k == 1:
            result = pipe(question=question, context=context)
            return {
                "answer": result["answer"],
                "confidence": round(float(result["score"]), 4),
                "start": result["start"],
                "end": result["end"],
            }
        else:
            results = pipe(question=question, context=context, top_k=top_k)
            return {
                "answers": [
                    {
                        "answer": r["answer"],
                        "confidence": round(float(r["score"]), 4),
                        "start": r["start"],
                        "end": r["end"],
                    }
                    for r in results
                ],
                "top_answer": results[0]["answer"],
                "top_confidence": round(float(results[0]["score"]), 4),
            }

    def answer_batch(
        self,
        questions: List[str],
        contexts: List[str],
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions with corresponding contexts."""
        return [
            self.answer(q, c) for q, c in zip(questions, contexts)
        ]

    def is_answerable(
        self,
        question: str,
        context: str,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Check if a question is answerable from the context.

        Args:
            question: Question to check.
            context: Context passage.
            threshold: Minimum confidence to consider answerable.

        Returns:
            Dict with answerable (bool), answer, confidence.
        """
        result = self.answer(question, context)
        answerable = result["confidence"] >= threshold

        return {
            "answerable": answerable,
            "answer": result["answer"] if answerable else None,
            "confidence": result["confidence"],
        }
