# %% [markdown]
# # 01 — Data Exploration
# SENTINEL-AI: Exploring datasets for fake news detection
#
# **Datasets:**
# - LIAR (12.8K PolitiFact statements)
# - FEVER v1.0 (Wikipedia-grounded claims)
# - Financial PhraseBank (4.8K sentiment-labeled sentences)

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# %%
# Load LIAR dataset
print("Loading LIAR dataset...")
liar = load_dataset("liar")
print(f"Train: {len(liar['train'])}, Val: {len(liar['validation'])}, Test: {len(liar['test'])}")

# %%
# Explore label distribution
labels = liar["train"]["label"]
label_names = ["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"]
pd.Series(labels).value_counts().plot(kind="bar", title="LIAR Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/liar_label_distribution.png", dpi=150)
plt.show()

# %%
# Text length analysis
lengths = [len(s.split()) for s in liar["train"]["statement"]]
print(f"Avg length: {np.mean(lengths):.1f} words, Max: {max(lengths)}, Min: {min(lengths)}")

# %%
# Map 6 labels → 4 classes
LABEL_MAP = {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}  # real/fake mapping
# 0=false→fake, 1=half-true→real, 2=mostly-true→real, 3=true→real, 4=barely-true→fake, 5=pants-fire→fake
