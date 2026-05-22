# %% [markdown]
# # 04 — Transformer Models
# Module C: BERT, RoBERTa, DistilBERT, BART, T5, NER, Sentiment
# **Covers:** CSR322 Units IV, V, VI

# %%
from src.nlp.bert_classifier import TransformerClassifier
from src.nlp.bart_summarizer import SummarizerModule
from src.nlp.ner_pipeline import NERPipeline
from src.nlp.sentiment import SentimentAnalyzer

# %%
# Zero-shot classification baseline
classifier = TransformerClassifier()
result = classifier.zero_shot_classify("The president announced new tariffs on Chinese goods.")
print(f"Zero-shot result: {result}")

# %%
# BART summarization
summarizer = SummarizerModule()
text = "The World Health Organization announced today that global COVID-19 cases have declined significantly..."
summary = summarizer.summarize(text)
print(f"Summary: {summary}")

# %%
# NER extraction
ner = NERPipeline()
entities = ner.extract_entities("Elon Musk announced that Tesla will open a new factory in Berlin, Germany.")
print(f"Entities: {entities}")

# %%
# Sentiment analysis
sentiment_analyzer = SentimentAnalyzer()
result = sentiment_analyzer.analyze("The stock market crashed dramatically today amid fears of recession.")
print(f"Sentiment: {result}")
