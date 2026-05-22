"""
SENTINEL-AI — Training Pipeline
9-step training pipeline with seed control and results logging.
Run: python scripts/train_all.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_FILE = "results/metrics_summary.json"


def log_result(step_name: str, metrics: dict):
    """Append step results to metrics_summary.json incrementally."""
    os.makedirs("results", exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {
            "steps": [],
            "metadata": {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        }

    data["steps"].append(
        {
            "step": step_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
        }
    )
    data["metadata"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  ✓ Logged results for: {step_name}")


def main():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    total_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'=' * 60}")
    print(f"SENTINEL-AI Training Pipeline")
    print(f"{'=' * 60}\n")

    # Step 1: Generate synthetic training data
    print("[Step 1/9] Generating synthetic training data...")
    n_train, n_val = 800, 200
    X_train = np.random.randn(n_train, 504).astype(np.float32)
    y_train = np.random.randint(0, 4, n_train)
    X_val = np.random.randn(n_val, 504).astype(np.float32)
    y_val = np.random.randint(0, 4, n_val)
    log_result(
        "synthetic_data",
        {"train_samples": n_train, "val_samples": n_val, "features": 504},
    )

    # Step 2: Train MLP with 3 optimizers
    print("\n[Step 2/9] Training MLP Classifier (3 optimizers)...")
    from src.models.mlp_classifier import train_mlp

    mlp_results = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        patience=5,
        save_dir="./models",
        device=device,
    )
    log_result(
        "mlp_training",
        {
            opt: {"f1": r["f1"], "best_epoch": r["best_epoch"]}
            for opt, r in mlp_results.items()
        },
    )

    # Step 3: Train Autoencoder on "real" class
    print("\n[Step 3/9] Training Autoencoder (real news only)...")
    from src.models.autoencoder import train_autoencoder

    X_real = np.random.randn(400, 300).astype(np.float32)
    ae_results = train_autoencoder(
        X_real,
        epochs=20,
        save_path="./models/autoencoder.pt",
        device=device,
    )
    log_result("autoencoder_training", {"best_loss": ae_results["best_loss"]})

    # Step 4: Train GAN
    print("\n[Step 4/9] Training Conditional GAN...")
    from src.models.gan_augmentor import train_cgan

    gan_results = train_cgan(
        X_train,
        y_train,
        epochs=30,
        save_dir="./models",
        device=device,
    )
    log_result(
        "gan_training",
        {
            "final_g_loss": gan_results["g_losses"][-1],
            "final_d_loss": gan_results["d_losses"][-1],
        },
    )

    # Step 5: Train VAE
    print("\n[Step 5/9] Training VAE...")
    from src.models.vae_explorer import train_vae

    X_bert = np.random.randn(400, 768).astype(np.float32)
    vae_results = train_vae(
        X_bert,
        epochs=20,
        save_dir="./models",
        device=device,
    )
    log_result("vae_training", {"final_loss": vae_results["train_losses"][-1]})

    # Step 6: Feature extraction test
    print("\n[Step 6/9] Testing Feature Extractor pipeline...")
    from src.preprocessing.cleaner import TextCleaner
    from src.preprocessing.pos_parser import PosParser
    from src.preprocessing.feature_extractor import FeatureExtractor

    cleaner = TextCleaner()
    parser = PosParser()
    extractor = FeatureExtractor()

    test_corpus = [
        "The president signed a controversial executive order today.",
        "Scientists discovered a breakthrough in cancer treatment.",
        "Breaking news: major earthquake strikes coastal region.",
        "New study reveals alarming effects of social media use.",
        "Government announces new economic stimulus package.",
    ] * 4  # 20 docs

    extractor.fit([cleaner.get_clean_text(t) for t in test_corpus])

    sample = "The government denied all allegations of corruption."
    cleaned = cleaner.clean(sample)
    pos = parser.parse(sample)
    features = extractor.transform(
        cleaned["cleaned_text"],
        adjective_density=pos["adjective_density"],
        adj_noun_ratio=pos["adj_noun_ratio"],
    )
    log_result(
        "feature_extraction",
        {
            "vector_dim": len(features["feature_vector"]),
            "topic_cluster": features["topic_cluster"],
        },
    )

    # Step 7: BiLSTM forward pass test
    print("\n[Step 7/9] Testing BiLSTM Attention...")
    from src.models.bilstm_attention import BiLSTMAttention

    bilstm = BiLSTMAttention()
    x = torch.randint(0, 30000, (4, 20))
    logits, attn = bilstm(x, return_attention=True)
    attn_sum = attn.sum(dim=-1).mean().item()
    log_result(
        "bilstm_test",
        {"attn_sum_mean": round(attn_sum, 4), "output_shape": list(logits.shape)},
    )

    # Step 8: TextCNN + ImageCNN forward pass test
    print("\n[Step 8/9] Testing CNN Models...")
    from src.models.cnn_moderation import TextCNN, ImageCNN

    text_cnn = TextCNN()
    x_text = torch.randint(0, 30000, (4, 50))
    text_out = text_cnn(x_text)

    img_cnn = ImageCNN()
    x_img = torch.randn(1, 3, 64, 64, requires_grad=True)
    img_out = img_cnn(x_img)
    img_out[0, 0].backward()
    heatmap = img_cnn.grad_cam(class_idx=0)

    log_result(
        "cnn_test",
        {
            "textcnn_shape": list(text_out.shape),
            "imagecnn_shape": list(img_out.shape),
            "gradcam_shape": list(heatmap.shape),
        },
    )

    # Step 9: GAN sample generation
    print("\n[Step 9/9] Generating GAN samples...")
    from src.models.gan_augmentor import generate_samples, Generator

    gen = Generator()
    samples = generate_samples(gen, class_label=1, num_samples=50)
    log_result("gan_generation", {"samples_shape": list(samples.shape)})

    total_time = time.time() - total_start
    log_result("pipeline_complete", {"total_time_seconds": round(total_time, 2)})

    print(f"\n{'=' * 60}")
    print(f"Training pipeline complete in {total_time:.1f}s")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
