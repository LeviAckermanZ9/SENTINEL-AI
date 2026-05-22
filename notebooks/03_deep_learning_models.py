# %% [markdown]
# # 03 — Deep Learning Models
# Module B: MLP, TextCNN, BiLSTM+Attention, Autoencoder
# **Covers:** CSR311 Units I, II, III, IV, VI

# %%
import torch
from src.models.mlp_classifier import MLPClassifier, train_mlp
from src.models.cnn_moderation import TextCNN, ImageCNN
from src.models.bilstm_attention import BiLSTMAttention
from src.models.autoencoder import NewsAutoencoder, train_word2vec

# %%
# MLP forward pass test
mlp = MLPClassifier(input_dim=504, num_classes=4)
x = torch.randn(8, 504)
out = mlp(x)
print(f"MLP output shape: {out.shape}")  # (8, 4)

# %%
# TextCNN forward pass test
textcnn = TextCNN(vocab_size=30000, embed_dim=100, num_classes=4)
x = torch.randint(0, 30000, (4, 50))
out = textcnn(x)
print(f"TextCNN output shape: {out.shape}")  # (4, 4)

# %%
# BiLSTM + Attention test
bilstm = BiLSTMAttention(vocab_size=30000, embed_dim=100, hidden_dim=256, num_classes=3)
x = torch.randint(0, 30000, (4, 20))
logits, attn = bilstm(x, return_attention=True)
print(f"BiLSTM output: {logits.shape}, Attention: {attn.shape}")
print(f"Attention sums to 1: {attn.sum(dim=-1)}")

# %%
# Autoencoder anomaly detection
ae = NewsAutoencoder(input_dim=300)
x = torch.randn(4, 300)
recon, _ = ae(x)
scores = ae.anomaly_score(x)
print(f"Reconstruction shape: {recon.shape}, Anomaly scores: {scores}")
