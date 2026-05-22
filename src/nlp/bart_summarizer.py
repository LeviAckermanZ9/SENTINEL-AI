"""
SENTINEL-AI — Summarizer Module
Covers: CSR322 Unit V (Encoder-decoder, BART, T5, QA)

SummarizerModule: BART + T5 summarization, ROUGE evaluation, QA on articles.
All models lazy-loaded on first use.
"""

from typing import Any, Dict, List, Optional


class SummarizerModule:
    """
    Abstractive summarization with BART and T5, plus QA capability.

    Models (lazy-loaded):
        - facebook/bart-large-cnn: primary summarizer
        - t5-small: comparison summarizer
        - deepset/roberta-base-squad2: QA pipeline

    Usage:
        summarizer = SummarizerModule()
        summary = summarizer.summarize("Long article text...")
        rouge = summarizer.evaluate_rouge(predictions, references)
        answer = summarizer.qa_on_article("What happened?", "Article context...")
    """

    def __init__(self):
        self._bart_pipeline = None
        self._t5_pipeline = None
        self._qa_pipeline = None

    def _load_bart(self):
        """Lazy-load BART summarization pipeline."""
        if self._bart_pipeline is None:
            from transformers import pipeline

            self._bart_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
            )
        return self._bart_pipeline

    def _load_t5(self):
        """Lazy-load T5 summarization pipeline."""
        if self._t5_pipeline is None:
            from transformers import pipeline

            self._t5_pipeline = pipeline(
                "summarization",
                model="t5-small",
            )
        return self._t5_pipeline

    def _load_qa(self):
        """Lazy-load QA pipeline."""
        if self._qa_pipeline is None:
            from transformers import pipeline

            self._qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
            )
        return self._qa_pipeline

    def summarize(
        self,
        text: str,
        model: str = "bart",
        max_length: int = 60,
        min_length: int = 20,
    ) -> Dict[str, Any]:
        """
        Generate abstractive summary.

        Args:
            text: Input text to summarize.
            model: 'bart' or 't5'.
            max_length: Maximum summary length in tokens.
            min_length: Minimum summary length in tokens.

        Returns:
            Dict with summary_text, model_used.
        """
        if model == "bart":
            pipe = self._load_bart()
        elif model == "t5":
            pipe = self._load_t5()
        else:
            raise ValueError(f"Unknown model '{model}'. Use 'bart' or 't5'.")

        # T5 requires prefix
        input_text = text
        if model == "t5":
            input_text = f"summarize: {text}"

        result = pipe(
            input_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

        return {
            "summary_text": result[0]["summary_text"],
            "model_used": model,
        }

    def summarize_compare(
        self,
        text: str,
        max_length: int = 60,
        min_length: int = 20,
    ) -> Dict[str, Any]:
        """
        Compare BART and T5 summarizations side-by-side.

        Args:
            text: Input text to summarize.
            max_length: Max summary length.
            min_length: Min summary length.

        Returns:
            Dict with bart_summary, t5_summary.
        """
        bart_result = self.summarize(
            text, model="bart", max_length=max_length, min_length=min_length
        )
        t5_result = self.summarize(
            text, model="t5", max_length=max_length, min_length=min_length
        )

        return {
            "bart_summary": bart_result["summary_text"],
            "t5_summary": t5_result["summary_text"],
        }

    def evaluate_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate summarization quality using ROUGE metrics.

        Args:
            predictions: List of predicted summaries.
            references: List of reference summaries.

        Returns:
            Dict with rouge1, rouge2, rougeL scores.
        """
        import evaluate as hf_evaluate

        rouge = hf_evaluate.load("rouge")
        results = rouge.compute(
            predictions=predictions,
            references=references,
        )

        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
        }

    def qa_on_article(
        self,
        question: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Answer a question about an article using extractive QA.

        Uses deepset/roberta-base-squad2.

        Args:
            question: Question to answer.
            context: Article/passage to search for the answer.

        Returns:
            Dict with answer, confidence, start, end.
        """
        pipe = self._load_qa()

        result = pipe(question=question, context=context)

        return {
            "answer": result["answer"],
            "confidence": result["score"],
            "start": result["start"],
            "end": result["end"],
        }

    def qa_batch(
        self,
        questions: List[str],
        contexts: List[str],
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions about corresponding contexts."""
        return [self.qa_on_article(q, c) for q, c in zip(questions, contexts)]
