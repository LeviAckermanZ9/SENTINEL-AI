"""
SENTINEL-AI — Model Test Suite
Tests forward pass shapes and key behaviors for all deep learning models.
"""

import numpy as np
import pytest
import torch

from src.models.mlp_classifier import MLPClassifier
from src.models.bilstm_attention import BiLSTMAttention
from src.models.cnn_moderation import TextCNN, ImageCNN
from src.models.autoencoder import NewsAutoencoder
from src.models.gan_augmentor import Generator, Discriminator
from src.models.vae_explorer import VAE


# ================================================================
# MLP Tests
# ================================================================

class TestMLPClassifier:
    def test_forward_shape(self):
        model = MLPClassifier(input_dim=504, num_classes=4)
        x = torch.randn(8, 504)
        out = model(x)
        assert out.shape == (8, 4)

    def test_different_batch_sizes(self):
        model = MLPClassifier(input_dim=504, num_classes=4)
        model.eval()  # BatchNorm requires >1 in train mode
        for bs in [1, 4, 16, 32]:
            x = torch.randn(bs, 504)
            out = model(x)
            assert out.shape == (bs, 4)

    def test_predict(self):
        model = MLPClassifier(input_dim=504, num_classes=4)
        x = torch.randn(8, 504)
        preds = model.predict(x)
        assert preds.shape == (8,)
        assert all(0 <= p < 4 for p in preds)

    def test_predict_proba(self):
        model = MLPClassifier(input_dim=504, num_classes=4)
        x = torch.randn(8, 504)
        proba = model.predict_proba(x)
        assert proba.shape == (8, 4)
        # Probabilities should sum to ~1.0
        sums = proba.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_custom_dims(self):
        model = MLPClassifier(input_dim=256, num_classes=3)
        x = torch.randn(4, 256)
        out = model(x)
        assert out.shape == (4, 3)


# ================================================================
# BiLSTM Tests
# ================================================================

class TestBiLSTMAttention:
    def test_forward_shape(self):
        model = BiLSTMAttention(vocab_size=30000, embed_dim=100, hidden_dim=256, num_classes=3)
        x = torch.randint(0, 30000, (4, 20))
        logits = model(x)
        assert logits.shape == (4, 3)

    def test_return_attention(self):
        model = BiLSTMAttention(vocab_size=30000, embed_dim=100, hidden_dim=256, num_classes=3)
        x = torch.randint(0, 30000, (4, 20))
        logits, attn = model(x, return_attention=True)
        assert logits.shape == (4, 3)
        assert attn.shape == (4, 20)

    def test_attention_sums_to_one(self):
        model = BiLSTMAttention(vocab_size=30000, embed_dim=100, hidden_dim=256, num_classes=3)
        x = torch.randint(0, 30000, (4, 20))
        _, attn = model(x, return_attention=True)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_different_seq_lengths(self):
        model = BiLSTMAttention(vocab_size=30000, embed_dim=100, hidden_dim=256, num_classes=3)
        for seq_len in [10, 50, 100]:
            x = torch.randint(0, 30000, (2, seq_len))
            logits, attn = model(x, return_attention=True)
            assert logits.shape == (2, 3)
            assert attn.shape == (2, seq_len)


# ================================================================
# TextCNN Tests
# ================================================================

class TestTextCNN:
    def test_forward_shape(self):
        model = TextCNN(vocab_size=30000, embed_dim=100, num_classes=4)
        x = torch.randint(0, 30000, (4, 50))
        out = model(x)
        assert out.shape == (4, 4)

    def test_different_batch_sizes(self):
        model = TextCNN(vocab_size=30000, embed_dim=100, num_classes=4)
        for bs in [1, 8, 16]:
            x = torch.randint(0, 30000, (bs, 50))
            out = model(x)
            assert out.shape == (bs, 4)

    def test_short_sequence(self):
        model = TextCNN(vocab_size=30000, embed_dim=100, num_classes=4)
        # Min sequence must be >= max filter size (5)
        x = torch.randint(0, 30000, (4, 6))
        out = model(x)
        assert out.shape == (4, 4)


# ================================================================
# ImageCNN Tests
# ================================================================

class TestImageCNN:
    def test_forward_shape(self):
        model = ImageCNN(in_channels=3, num_classes=4)
        x = torch.randn(4, 3, 64, 64)
        out = model(x)
        assert out.shape == (4, 4)

    def test_grad_cam(self):
        model = ImageCNN(in_channels=3, num_classes=4)
        model.train()  # Need gradients
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        # Backward on class 0
        out[0, 0].backward()
        heatmap = model.grad_cam(class_idx=0)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_activations_stored(self):
        model = ImageCNN(in_channels=3, num_classes=4)
        x = torch.randn(2, 3, 32, 32)
        _ = model(x)
        assert model.activations is not None
        assert model.activations.shape[0] == 2
        assert model.activations.shape[1] == 256  # 256 channels from last block


# ================================================================
# Autoencoder Tests
# ================================================================

class TestAutoencoder:
    def test_forward_shape(self):
        model = NewsAutoencoder(input_dim=300, latent_dim=32)
        x = torch.randn(4, 300)
        recon, latent = model(x)
        assert recon.shape == (4, 300)
        assert latent.shape == (4, 32)

    def test_anomaly_score(self):
        model = NewsAutoencoder(input_dim=300, latent_dim=32)
        x = torch.randn(4, 300)
        scores = model.anomaly_score(x)
        assert scores.shape == (4,)
        assert (scores >= 0).all()

    def test_encode_decode(self):
        model = NewsAutoencoder(input_dim=300, latent_dim=32)
        x = torch.randn(8, 300)
        latent = model.encode(x)
        assert latent.shape == (8, 32)
        decoded = model.decode(latent)
        assert decoded.shape == (8, 300)

    def test_get_latent(self):
        model = NewsAutoencoder(input_dim=300, latent_dim=32)
        x = torch.randn(4, 300)
        latent = model.get_latent(x)
        assert isinstance(latent, np.ndarray)
        assert latent.shape == (4, 32)


# ================================================================
# GAN Tests
# ================================================================

class TestGAN:
    def test_generator_shape(self):
        gen = Generator(noise_dim=100, num_classes=4, output_dim=504)
        noise = torch.randn(8, 100)
        labels = torch.randint(0, 4, (8,))
        fake = gen(noise, labels)
        assert fake.shape == (8, 504)

    def test_discriminator_shape(self):
        disc = Discriminator(input_dim=504, num_classes=4)
        x = torch.randn(8, 504)
        labels = torch.randint(0, 4, (8,))
        validity = disc(x, labels)
        assert validity.shape == (8, 1)

    def test_generator_output_range(self):
        gen = Generator(noise_dim=100, num_classes=4, output_dim=504)
        noise = torch.randn(4, 100)
        labels = torch.randint(0, 4, (4,))
        fake = gen(noise, labels)
        # Sigmoid output: should be in [0, 1]
        assert fake.min() >= 0.0
        assert fake.max() <= 1.0

    def test_discriminator_output_range(self):
        disc = Discriminator(input_dim=504, num_classes=4)
        x = torch.randn(4, 504)
        labels = torch.randint(0, 4, (4,))
        validity = disc(x, labels)
        # Sigmoid output: should be in [0, 1]
        assert validity.min() >= 0.0
        assert validity.max() <= 1.0


# ================================================================
# VAE Tests
# ================================================================

class TestVAE:
    def test_forward_shape(self):
        vae = VAE(input_dim=768, latent_dim=32)
        x = torch.randn(4, 768)
        recon, mu, loss = vae(x)
        assert recon.shape == (4, 768)
        assert mu.shape == (4, 32)

    def test_loss_positive(self):
        vae = VAE(input_dim=768, latent_dim=32)
        x = torch.randn(4, 768)
        _, _, loss = vae(x)
        assert loss.item() > 0

    def test_encode(self):
        vae = VAE(input_dim=768, latent_dim=32)
        x = torch.randn(4, 768)
        mu, log_var = vae.encode(x)
        assert mu.shape == (4, 32)
        assert log_var.shape == (4, 32)

    def test_reparameterize(self):
        vae = VAE(input_dim=768, latent_dim=32)
        mu = torch.zeros(4, 32)
        log_var = torch.zeros(4, 32)
        z = vae.reparameterize(mu, log_var)
        assert z.shape == (4, 32)

    def test_sample(self):
        vae = VAE(input_dim=768, latent_dim=32)
        samples = vae.sample(num_samples=10)
        assert samples.shape == (10, 768)

    def test_get_latent(self):
        vae = VAE(input_dim=768, latent_dim=32)
        x = torch.randn(4, 768)
        latent = vae.get_latent(x)
        assert isinstance(latent, np.ndarray)
        assert latent.shape == (4, 32)
