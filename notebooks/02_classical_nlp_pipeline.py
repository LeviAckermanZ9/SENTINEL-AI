# %% [markdown]
# # 02 — Classical NLP Pipeline
# Module A: Text cleaning, POS parsing, TF-IDF feature extraction
# **Covers:** CSR322 Units I, II, III

# %%
import sys
sys.path.insert(0, "..")
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.pos_parser import PosParser
from src.preprocessing.feature_extractor import FeatureExtractor

# %%
# Initialize components
cleaner = TextCleaner()
parser = PosParser()
extractor = FeatureExtractor()

# %%
# Test cleaning
sample = "<p>BREAKING: @user says #fakenews about http://example.com!!!</p>"
cleaned = cleaner.clean(sample)
print(f"Original: {sample}")
print(f"Cleaned: {cleaned}")

# %%
# Test POS parsing
pos_result = parser.parse("The president signed a controversial executive order yesterday.")
print(f"POS tags: {pos_result['pos_tags'][:5]}")
print(f"Dep triplets: {pos_result['dep_triplets']}")
print(f"Adjective density: {pos_result['adjective_density']:.3f}")

# %%
# Train feature extractor on corpus and transform
corpus = ["The president signed a new law.", "Scientists discovered a cure.", "Markets crashed today."]
extractor.fit(corpus)
features = extractor.transform("The president announced new economic policies.")
print(f"TF-IDF vector shape: {features['tfidf_vector'].shape}")
print(f"Topic cluster: {features['topic_cluster']}")
