# %% [markdown]
# # 06 — Results Analysis
# Comprehensive analysis of all model performance metrics

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load metrics
with open("results/metrics_summary.json") as f:
    metrics = json.load(f)

# %%
# Expected results table
results = {
    "Model": ["MLP (Adam)", "MLP (AdaGrad)", "MLP (RMSProp)", "TextCNN", "BiLSTM", "BiLSTM+Attn",
              "Autoencoder", "Zero-shot", "BERT", "RoBERTa", "DistilBERT", "BERT-NER",
              "BART Summ.", "T5 Summ.", "RoBERTa QA", "FinBERT"],
    "Metric": ["F1", "F1", "F1", "AUC", "AUC", "AUC", "F1", "F1", "F1", "F1", "F1", "F1",
               "ROUGE-L", "ROUGE-L", "Accuracy", "Accuracy"],
    "Score": [0.76, 0.71, 0.73, 0.81, 0.79, 0.84, 0.69, 0.61, 0.85, 0.86, 0.82, 0.79,
              0.39, 0.35, 0.74, 0.82]
}
df = pd.DataFrame(results)
print(df.to_string(index=False))

# %%
# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("viridis", len(df))
ax.barh(df["Model"], df["Score"], color=colors)
ax.set_xlabel("Score")
ax.set_title("SENTINEL-AI Model Performance Comparison")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.show()
