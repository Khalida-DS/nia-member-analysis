# AAA Northeast Member Analysis

> **Cross-sell propensity scoring · ERS cost prediction · Member segmentation**

[![CI](https://github.com/Khalida-DS/aaa-northeast-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/aaa-northeast-analysis/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

---

## Business Problem

AAA Northeast serves ~6 million members across six states. Cross-sell campaigns cost **$5–15 per contact** and achieve less than 5% conversion when untargeted. This project replaces scatter-shot outreach with data-driven household scoring.

| Question | Approach | Business Metric |
|----------|----------|-----------------|
| Who is most likely to buy each product? | Per-product binary classifiers | **Lift @ Top 10% ≥ 2.0×** |
| How much will a household cost in ERS next year? | Regression on log-transformed cost | R² > 0.25 |
| What behavioural segments exist? | K-Means on propensity scores | Silhouette ≥ 0.30 |
| What should we recommend to each household? | Cluster → segment action table | Full household coverage |

---

## Quick Start
```bash
# 1. Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/aaa-northeast-analysis
cd aaa-northeast-analysis

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add raw data
cp /path/to/member_sample.csv data/raw/member_sample.csv

# 5. Run the full pipeline
python -m src.pipelines.train --stage all

# 6. Run tests
pytest tests/ -v --cov=src
```

---

## Repository Structure
```
aaa-northeast-analysis/
│
├── configs/
│   └── settings.yaml          ← all constants — one place to change everything
│
├── src/
│   ├── config.py              ← typed settings loader
│   ├── features/
│   │   └── preprocessing.py   ← 10-step raw→household pipeline
│   ├── models/
│   │   ├── classifier.py      ← per-product propensity models
│   │   ├── regressor.py       ← ERS cost prediction
│   │   └── clustering.py      ← K-Means segmentation
│   ├── evaluation/
│   │   ├── metrics.py         ← AUC, F1, Lift@10%, RMSE, R²
│   │   └── plots.py           ← all figures (one style, one place)
│   └── pipelines/
│       └── train.py           ← CLI orchestrator (--stage all/preprocess/…)
│
├── tests/                     ← 49+ unit tests, one file per src module
├── notebooks/                 ← EDA and results notebooks
├── data/raw/                  ← source CSV (gitignored)
├── models/artifacts/          ← .pkl + metadata JSON (gitignored)
├── reports/figures/           ← generated charts (gitignored)
│
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Pipeline Stages
```bash
python -m src.pipelines.train --stage preprocess   # raw CSV → parquet feature store
python -m src.pipelines.train --stage classify     # train 7 propensity classifiers
python -m src.pipelines.train --stage regress      # train ERS cost regressor
python -m src.pipelines.train --stage cluster      # fit K-Means, build action table
python -m src.pipelines.train --stage all          # run everything in sequence
```

---

## Docker
```bash
docker build -t aaa-northeast .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  aaa-northeast
```

---

## Methodology

**Preprocessing** — Raw data arrives at the individual×service-call grain (21,344 rows). The pipeline aggregates to household level (~3,500 rows) because products are household decisions and marketing targets households, not individuals. ERS cost aggregation happens *before* deduplication to preserve full cost history.

**Classification** — Separate model per product (buyer profiles differ). `class_weight='balanced'` handles imbalance without synthetic upsampling. 70/10/20 split with a dedicated validation set for hyperparameter selection keeps the test set truly held out. Lift@10% is the primary business metric.

**Regression** — `np.log1p` transform before fitting handles the zero-inflated, right-skewed cost distribution. `np.expm1` recovers dollar-scale predictions. R²=0.15–0.30 is realistic; higher values signal leakage.

**Clustering** — K-Means on predicted propensity scores groups members by purchase *likelihood*, not just demographics. Optimal k chosen by silhouette score. Each cluster maps to one recommended product in the action table.

---

## Data Dictionary

See [`data/README.md`](data/README.md) for full column descriptions and exclusion rationale.

---

## Contributing

1. Branch from `develop`
2. Add tests for any new function
3. All tests must pass: `pytest tests/ -v`
4. Open a PR targeting `develop`