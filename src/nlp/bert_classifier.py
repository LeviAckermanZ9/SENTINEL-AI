"""
SENTINEL-AI — Transformer Classifier
Covers: CSR322 Units IV (BERT, HuggingFace, fine-tuning)

TransformerClassifier: BERT/RoBERTa/DistilBERT fine-tuning + zero-shot baseline.
All models lazy-loaded on first use.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np


class TransformerClassifier:
    """
    Multi-model transformer classifier for fake news detection.

    Supports three fine-tunable models and a zero-shot baseline:
        - bert-base-uncased → BertForSequenceClassification
        - roberta-base → RobertaForSequenceClassification
        - distilbert-base-uncased → DistilBertForSequenceClassification
        - facebook/bart-large-mnli → zero-shot baseline

    All models lazy-loaded (no downloads on import).

    Usage:
        classifier = TransformerClassifier()
        result = classifier.zero_shot_classify("Some news text")
        trainer = classifier.fine_tune("bert", train_dataset, eval_dataset)
    """

    MODELS = {
        "bert": {
            "name": "bert-base-uncased",
            "class": "BertForSequenceClassification",
            "tokenizer": "BertTokenizer",
        },
        "roberta": {
            "name": "roberta-base",
            "class": "RobertaForSequenceClassification",
            "tokenizer": "RobertaTokenizer",
        },
        "distilbert": {
            "name": "distilbert-base-uncased",
            "class": "DistilBertForSequenceClassification",
            "tokenizer": "DistilBertTokenizer",
        },
    }

    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    CANDIDATE_LABELS = ["real news", "fake news", "satire", "spam"]
    LABEL_MAP = {0: "REAL", 1: "FAKE", 2: "SATIRE", 3: "SPAM"}

    def __init__(self):
        self._zero_shot_pipeline = None
        self._loaded_models: Dict[str, Any] = {}
        self._loaded_tokenizers: Dict[str, Any] = {}

    def _load_zero_shot(self):
        """Lazy-load zero-shot classification pipeline."""
        if self._zero_shot_pipeline is None:
            from transformers import pipeline

            self._zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model=self.ZERO_SHOT_MODEL,
            )
        return self._zero_shot_pipeline

    def _load_model_and_tokenizer(self, model_key: str, num_labels: int = 4):
        """Lazy-load a fine-tuning model and tokenizer."""
        if model_key not in self.MODELS:
            raise ValueError(
                f"Unknown model key '{model_key}'. Choose from: {list(self.MODELS.keys())}"
            )

        if model_key not in self._loaded_models:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            config = self.MODELS[model_key]
            model_name = config["name"]

            self._loaded_tokenizers[model_key] = AutoTokenizer.from_pretrained(
                model_name
            )
            self._loaded_models[model_key] = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_labels
                )
            )

        return self._loaded_models[model_key], self._loaded_tokenizers[model_key]

    def zero_shot_classify(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Classify text using zero-shot BART-MNLI baseline.

        Args:
            text: Input text to classify.
            candidate_labels: Custom labels (default: real/fake/satire/spam).

        Returns:
            Dict with labels, scores, predicted_label, confidence.
        """
        if candidate_labels is None:
            candidate_labels = self.CANDIDATE_LABELS

        pipe = self._load_zero_shot()
        result = pipe(text, candidate_labels)

        return {
            "labels": result["labels"],
            "scores": result["scores"],
            "predicted_label": result["labels"][0],
            "confidence": result["scores"][0],
        }

    def _zero_shot_baseline(
        self,
        texts: List[str],
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run zero-shot baseline on a batch of texts.

        Args:
            texts: List of texts.
            candidate_labels: Custom candidate labels.

        Returns:
            Dict with predictions, accuracy stats.
        """
        if candidate_labels is None:
            candidate_labels = self.CANDIDATE_LABELS

        pipe = self._load_zero_shot()
        predictions = []

        for text in texts:
            result = pipe(text, candidate_labels)
            predictions.append(
                {
                    "text": text[:100],
                    "predicted": result["labels"][0],
                    "confidence": result["scores"][0],
                    "all_scores": dict(zip(result["labels"], result["scores"])),
                }
            )

        return {"predictions": predictions, "total": len(predictions)}

    def fine_tune(
        self,
        model_key: str,
        train_dataset=None,
        eval_dataset=None,
        num_labels: int = 4,
        output_dir: str = "./models/transformer",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        save_strategy: str = "epoch",
        metric_for_best_model: str = "f1",
    ):
        """
        Fine-tune a transformer model on labeled data.

        Args:
            model_key: One of 'bert', 'roberta', 'distilbert'.
            train_dataset: HuggingFace Dataset for training.
            eval_dataset: HuggingFace Dataset for evaluation.
            num_labels: Number of classification labels.
            output_dir: Directory for checkpoints.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            warmup_ratio: Warmup ratio.
            weight_decay: Weight decay.
            save_strategy: When to save checkpoints.
            metric_for_best_model: Metric for best model selection.

        Returns:
            Configured HuggingFace Trainer instance.
        """
        from transformers import TrainingArguments, Trainer
        import evaluate as hf_evaluate

        model, tokenizer = self._load_model_and_tokenizer(model_key, num_labels)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, model_key),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy=save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            logging_steps=50,
            report_to="none",
        )

        # Metric computation
        f1_metric = hf_evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            f1 = f1_metric.compute(
                predictions=preds, references=labels, average="macro"
            )
            accuracy = (preds == labels).mean()
            return {"f1": f1["f1"], "accuracy": accuracy}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        return trainer

    def tokenize_dataset(self, dataset, model_key: str, max_length: int = 128):
        """
        Tokenize a HuggingFace dataset for fine-tuning.

        Args:
            dataset: HuggingFace Dataset with 'statement' and 'label' columns.
            model_key: Model key for tokenizer selection.
            max_length: Max sequence length.

        Returns:
            Tokenized dataset.
        """
        _, tokenizer = self._load_model_and_tokenizer(model_key)

        def tokenize_fn(examples):
            return tokenizer(
                examples["statement"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return tokenized

    def predict(self, text: str, model_key: str) -> Dict[str, Any]:
        """
        Predict class for a single text using a loaded model.

        Args:
            text: Input text.
            model_key: Which model to use.

        Returns:
            Dict with predicted_class, confidence, probabilities.
        """
        import torch

        model, tokenizer = self._load_model_and_tokenizer(model_key)
        model.eval()

        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        return {
            "predicted_class": pred_class,
            "predicted_label": self.LABEL_MAP.get(pred_class, "UNKNOWN"),
            "confidence": confidence,
            "probabilities": probs[0].tolist(),
        }
