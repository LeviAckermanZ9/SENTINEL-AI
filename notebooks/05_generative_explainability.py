# %% [markdown]
# # 05 — Generative Models + Explainability
# Module D: cGAN, VAE, SHAP, LIME, Federated Learning
# **Covers:** CSR311 Unit V

# %%
import torch
from src.models.gan_augmentor import Generator, Discriminator
from src.models.vae_explorer import VAE
from src.models.explainer import ModelExplainer

# %%
# cGAN test
gen = Generator(noise_dim=100, num_classes=4, output_dim=504)
disc = Discriminator(input_dim=504, num_classes=4)
noise = torch.randn(8, 100)
labels = torch.randint(0, 4, (8,))
fake_data = gen(noise, labels)
validity = disc(fake_data, labels)
print(f"Generated shape: {fake_data.shape}, Validity shape: {validity.shape}")

# %%
# VAE test
vae = VAE(input_dim=768, latent_dim=32)
x = torch.randn(4, 768)
recon, mu, log_var = vae(x)
print(f"VAE recon: {recon.shape}, mu: {mu.shape}, log_var: {log_var.shape}")

# %%
# Explainer (SHAP/LIME) — requires trained model
# explainer = ModelExplainer(model, feature_names)
# shap_values = explainer.shap_explain(X_test)
# lime_exp = explainer.lime_explain(text, predict_fn)
print("Explainer ready — requires trained model for demonstration.")
