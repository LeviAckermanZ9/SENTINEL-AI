# ============================================
# SENTINEL-AI — Multi-stage Production Dockerfile
# ============================================

# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production — lean runtime image
FROM python:3.11-slim AS production

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords'); nltk.download('omw-1.4')"

# Copy application source
COPY src/ /app/src/
COPY scripts/ /app/scripts/
WORKDIR /app

# Create non-root user
RUN groupadd -r sentinel && useradd -r -g sentinel -u 1001 sentinel \
    && chown -R sentinel:sentinel /app

# Create necessary directories
RUN mkdir -p /app/models /app/results /app/data && chown -R sentinel:sentinel /app

USER sentinel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
