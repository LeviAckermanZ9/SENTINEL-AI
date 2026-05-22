<p align="center">
  <h1 align="center">🛡️ SENTINEL-AI</h1>
  <p align="center">
    <strong>Real-Time Social Media Intelligence & Content Safety Platform</strong>
  </p>
  <p align="center">
    LPU B.Tech CSE (AI/ML) · CSR311 · CSR322 · INT377 · Session 2025-26
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.x-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/transformers-HuggingFace-yellow?style=flat-square" alt="Transformers">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white" alt="K8s">
  <img src="https://img.shields.io/badge/Terraform-7B42BC?style=flat-square&logo=terraform&logoColor=white" alt="Terraform">
</p>

---

## Overview

SENTINEL-AI is a production-grade, cloud-deployed AI platform that analyzes social media posts and news articles through a **7-module AI pipeline**:

```
Input → [A] Classical NLP → [B] Deep Learning → [C] Transformers
     → [D] Generative+XAI → [E] FastAPI → [F] DevOps/Cloud
                          → [G] RAG Fact-Check (parallel branch)
```

**Output:** Verdict (REAL / FAKE / SATIRE / SPAM), confidence score, SHAP explanation, named entities, BART summary, sentiment & manipulation score, anomaly score, and RAG-grounded evidence with sources.

---

## Architecture

| Module | Component | Key Technologies |
|--------|-----------|-----------------|
| **A** | Classical NLP Preprocessing | NLTK, spaCy, TF-IDF, KMeans |
| **B** | Deep Learning Models | MLP, TextCNN, BiLSTM+Attention, Autoencoder |
| **C** | Transformer NLP | BERT, RoBERTa, DistilBERT, BART, T5, FinBERT |
| **D** | Generative + XAI | cGAN, VAE, SHAP, LIME, FedAvg |
| **E** | API Serving | FastAPI, JWT Auth, Prometheus |
| **F** | DevOps & Cloud | Docker, K8s, Terraform, GitHub Actions, Grafana |
| **G** | RAG Fact-Check | ChromaDB, LangChain, Flan-T5/Mistral, RAGAS |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_USERNAME/sentinel-ai.git && cd sentinel-ai
cp .env.example .env

# 2. Start infrastructure
docker-compose up -d chromadb prometheus grafana alertmanager

# 3. Build knowledge base (one-time, ~15 min)
docker-compose --profile init up kb-builder

# 4. Start full platform
docker-compose up -d sentinel-api sentinel-ui

# 5. Verify
curl http://localhost:8000/health
```

### Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | — |
| Dashboard | http://localhost:8501 | — |
| Grafana | http://localhost:3000 | admin / sentinel_admin |
| ChromaDB | http://localhost:8100/api/v1/heartbeat | — |
| Prometheus | http://localhost:9090 | — |

---

## Usage

### Classify a post

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists confirm 5G towers spread COVID-19.", "include_explanation": true, "include_summary": true}'
```

### Full analysis with RAG fact-check

```bash
curl -X POST http://localhost:8000/fact-check/full-analysis \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm 5G towers spread COVID-19.", "include_rag": true, "include_explanation": true, "include_summary": true}' \
  | python -m json.tool
```

---

## Performance Results

| Component | Metric | Result |
|-----------|--------|--------|
| MLP (Adam) | Macro-F1 | 0.76 |
| BERT fine-tuned | Macro-F1 | 0.85 |
| RoBERTa fine-tuned | Macro-F1 | 0.86 |
| BiLSTM + Attention | AUC-ROC | 0.84 |
| BART Summarization | ROUGE-L | 0.39 |
| BERT-NER | Entity F1 | 0.79 |
| FinBERT Sentiment | Accuracy | 0.82 |
| RAGAS Faithfulness | Score | 0.83 |
| API P95 (ML only) | Latency | 312ms |
| API P95 (RAG) | Latency | 1,240ms |

---

## Training

```bash
# Train all models (requires GPU recommended)
python -m scripts.train_all

# Individual steps
python -c "from scripts.train_all import step1_word2vec; step1_word2vec()"
```

All metrics are logged to `results/metrics_summary.json`.

---

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run full test suite
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-fail-under=70 -v
```

---

## Deployment

### Docker

```bash
docker build -t sentinel-ai .
docker run -p 8000:8000 sentinel-ai
```

### Kubernetes

```bash
minikube start --memory=4096 --cpus=2
kubectl apply -f k8s/
kubectl get pods -n sentinel-prod
```

### Terraform (AWS)

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

---

## Project Structure

```
sentinel-ai/
├── .github/workflows/     # CI/CD pipelines
├── src/
│   ├── preprocessing/     # Module A: Classical NLP
│   ├── models/            # Module B+D: Deep Learning + Generative/XAI
│   ├── nlp/               # Module C: Transformer NLP
│   ├── rag/               # Module G: RAG Fact-Check
│   ├── api/               # Module E: FastAPI
│   ├── dashboard/         # Streamlit UI
│   └── monitoring/        # Prometheus metrics
├── notebooks/             # Jupyter notebooks (7)
├── scripts/               # Training pipeline
├── tests/                 # Test suite
├── terraform/             # AWS IaC
├── k8s/                   # Kubernetes manifests
├── monitoring/            # Prometheus + Grafana configs
├── data/                  # Datasets + vector store
├── models/                # Trained checkpoints
└── results/               # Metrics + visualizations
```

---

## Academic Coverage

This project covers **21 academic checkpoints** across three subjects:

- **CSR311** (Deep Learning): Units I–VI — MLP, CNN, LSTM, GAN, VAE, Word2Vec, SHAP/LIME
- **CSR322** (NLP): Units I–VI + CO4/CO5/CO6 — Tokenization, POS, TF-IDF, BERT, BART, NER, RAG
- **INT377** (DevOps): Units I–VI — Docker, K8s, Terraform, CI/CD, Monitoring, Security

---

## License

This project is developed for academic purposes at Lovely Professional University.

---

<p align="center">
  <strong>SENTINEL-AI</strong> · Built with 🔥 by LPU CSE (AI/ML) · 2025-26
</p>
