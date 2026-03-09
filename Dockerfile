# ─────────────────────────────────────────────────────────────────────────────
# AAA Northeast Member Analysis — Docker Image
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   docker build -t aaa-northeast .
#   docker run --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/reports:/app/reports \
#     aaa-northeast
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — never run ML workloads as root
RUN useradd --create-home --shell /bin/bash aaa
WORKDIR /app
RUN chown aaa:aaa /app

# Install Python dependencies (layer-cached: only rebuilt when requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip --quiet \
    && pip install --no-cache-dir -r requirements.txt

# Copy source (after deps — faster cache invalidation)
COPY --chown=aaa:aaa src/       ./src/
COPY --chown=aaa:aaa tests/     ./tests/
COPY --chown=aaa:aaa configs/   ./configs/
COPY --chown=aaa:aaa notebooks/ ./notebooks/

# Create runtime directories
RUN mkdir -p data/raw data/processed data/external \
             models/artifacts reports/figures \
    && chown -R aaa:aaa data models reports

USER aaa

# Smoke test at build time — fails fast if config is broken
RUN python -c "from src.config import get_config; get_config(); print('Config OK')"

# Default: run the full pipeline
CMD ["python", "-m", "src.pipelines.train", "--stage", "all"]
```

---

## FILE 21 — `requirements.txt`
```
# ── Core data stack ───────────────────────────────────────────────────────────
numpy>=1.24,<2.0
pandas>=2.0,<3.0
pyarrow>=14.0

# ── Machine learning ──────────────────────────────────────────────────────────
scikit-learn>=1.3,<2.0
xgboost>=2.0,<3.0
joblib>=1.3

# ── Visualisation ─────────────────────────────────────────────────────────────
matplotlib>=3.7,<4.0
seaborn>=0.13
scipy>=1.11

# ── Configuration ─────────────────────────────────────────────────────────────
pyyaml>=6.0

# ── Notebooks ─────────────────────────────────────────────────────────────────
jupyter>=1.0
nbformat>=5.9
ipykernel>=6.0

# ── Testing ───────────────────────────────────────────────────────────────────
pytest>=7.4
pytest-cov>=4.1