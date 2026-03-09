# NorthShield Insurance Association — Member Intelligence Platform

> **Which members should we call? About which product? In what order?**
> This system answers all three questions — automatically, for every household.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://nia-member-analysis-gcfpuvp2upf6m8ebrgvq3j.streamlit.app/)
[![CI](https://github.com/Khalida-DS/nia-member-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Khalida-DS/nia-member-analysis/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://www.python.org)

---

## The Problem — In Plain English

NorthShield Insurance Association (NIA) has **3,511 member households**. Most of them hold only 1 or 2 products,
even though they are eligible for 9 — insurance, credit cards, travel services, and more.

The marketing team used to contact **everyone about everything**. That meant:

- 📬 31,599 outreach contacts per campaign cycle
- 💸 $5–15 cost per contact
- 😕 Less than 5% of contacts converting to a sale
- 🗑️ 95% of the marketing budget wasted

**This project changes that.** Instead of guessing, we use member behaviour data to predict
who is likely to buy what — and focus the marketing budget only on the households most
likely to say yes.

---

## The Solution — What This System Does

Think of it like a smart assistant that reads the history of every household and says:
*"This family has used claims service 4 times this year and earns $110K —
they are very likely to want NIA insurance next."*

The system does three things:

### 1. 🎯 Propensity Scoring — Who will buy what?
For each of 6 products, a model scores every household from 0 to 1.
A score of 0.75 means "75% likely to buy." Marketing contacts the top scorers first.

### 2. 💰 Cost Prediction — What will this household cost us?
A separate model predicts how much claims service each household will use next year.
This helps budget planning and identifies households that need a premium service plan.

### 3. 🗂️ Segmentation — How do we group households for campaigns?
Households are grouped into 8 segments based on their purchase likelihood profiles.
Each segment gets one clear recommended product and a campaign priority level.

---

## Results — By The Numbers

| What We Measure | Result | What It Means in Plain English |
|---|---|---|
| **Best model quality (AUC)** | 0.89 | Near-perfect ability to separate buyers from non-buyers |
| **Best campaign efficiency (Lift@10%)** | 5.71× | Contact 351 households and find as many buyers as contacting 2,005 at random |
| **Cost prediction accuracy (R²)** | 0.59 | Model explains 59% of cost variation using only member behaviour |
| **Households scored** | 3,511 | Every household in the member base gets a score and a recommendation |
| **Households ready for offers** | 1,924 (55%) | Active campaign targets with propensity scores above threshold |
| **Households in nurture queue** | 1,587 (45%) | Not ready now — re-score in 6 months after engagement |

---

## The 8 Marketing Segments

| Priority | Segment | Size | Recommended Offer | Why |
|---|---|---|---|---|
| 🔴 Highest | High High Claims Members | 177 hh | INS Client | 97% use claims service — insurance is the natural next step |
| 🔴 Highest | Long-Tenure Loyalists | 257 hh | FSV Credit Card | 42-year average tenure, already hold 2+ products |
| 🔴 Highest | High-Income Prospects | 265 hh | INS Client | $108K average income, financially ready |
| 🟡 Second Wave | Long-Tenure Single | 497 hh | INS Client | Room to grow from 1 product to 2 |
| 🟡 Second Wave | Active Claims Users | 463 hh | INS Client | High engagement, moderate conversion readiness |
| 🟡 Second Wave | Established Members | 265 hh | INS Client | Stable members not yet cross-sold |
| ⏸ Nurture | Dormant Long-Tenure | 1,071 hh | No offer yet | Low propensity — preserve relationship, re-score in 6 months |
| ⏸ Nurture | Disengaged Claims Users | 516 hh | No offer yet | High claims usage but no cross-sell signal yet |

---

## 🖥️ Live Demo

The interactive app lets you explore all results without running any code.

**[→ Open the Live Demo](https://khalida-ds-nia-member-analysis-app.streamlit.app)**

The app has 5 pages:

- **Overview** — KPIs, pipeline architecture, recommendation distribution
- **Model Performance** — AUC scores, lift charts, leakage investigation walkthrough
- **Segment Explorer** — Browse all 8 segments with profiles and business rationale
- **Household Lookup** — Score any household live against all 6 product models
- **Technical Deep Dive** — Architecture decisions, test suite, before/after comparisons

---

## How the Pipeline Works

Raw data comes in as 21,344 individual service call records.
The pipeline transforms them into a clean household-level dataset and trains the models.

```
Raw CSV (21,344 service call rows)
    │
    ├─ Aggregate costs per member        ← sum claims costs BEFORE deduplication
    ├─ Filter cancelled members          ← remove noise from the data
    ├─ Encode income bands to numbers    ← "$50K–$75K" becomes 62500
    ├─ Deduplicate to one row per member ← attach aggregated cost history
    ├─ Aggregate to one row per household← products are household decisions
    ├─ Engineer 50+ new features         ← total_calls, is_high_income, tenure...
    └─ Build model matrix                ← 140 final features, zero nulls
    │
    ├─→ 6 Classification models          → AUC 0.82–0.89
    ├─→ 1 Regression model               → R² = 0.59
    └─→ Clustering + Action Table        → 8 segments, 1 recommendation each
```

**Run everything in one command:**
```bash
python -m src.pipelines.train --stage all
```

---

## Technical Highlights (For Data Scientists & Engineers)

- **No data leakage** — Three-round investigation removed cost-year columns, then
  a proxy feature (`cost_trend`) discovered through feature importance inspection
- **Cluster on propensity, not demographics** — K-Means on predicted probabilities
  produces marketing-actionable segments, not demographic descriptions
- **68 unit tests** — Every function verified before the pipeline runs on real data
- **Stage-by-stage CLI** — `--stage preprocess/classify/regress/cluster/all`
  so each component can be developed and debugged independently
- **YAML config** — Every constant (k=8, test_size=0.20, income bands) lives in
  `configs/settings.yaml` — one file to change, everything updates
- **Typed dataclass config** — `cfg.training.test_size` catches typos at import
  time; raw dict access fails silently at runtime

---

## Project Structure

```
nia-member-analysis/
├── configs/settings.yaml        ← single source of truth for all constants
├── src/
│   ├── config.py                ← typed settings loader
│   ├── features/preprocessing.py← 10-step raw → household pipeline
│   ├── models/
│   │   ├── classifier.py        ← per-product propensity models
│   │   ├── regressor.py         ← claims cost prediction (Huber)
│   │   └── clustering.py        ← K-Means on propensity scores
│   ├── evaluation/
│   │   ├── metrics.py           ← AUC, Lift@k, RMSE, R²
│   │   └── plots.py             ← all visualisations
│   └── pipelines/train.py       ← CLI orchestrator
├── tests/                       ← 68 unit tests
├── notebooks/                   ← 4 analysis notebooks
├── app.py                       ← Streamlit portfolio demo
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Khalida-DS/nia-member-analysis
cd nia-member-analysis

# Create environment
conda create -n nia-member-analysis python=3.11
conda activate nia-member-analysis

# Install dependencies
pip install -r requirements.txt

# Add your data
cp /path/to/member_sample.csv data/raw/member_sample.csv

# Run pipeline
python -m src.pipelines.train --stage all

# Run tests
pytest tests/ -v

# Launch demo app
streamlit run app.py
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ML Models | scikit-learn, XGBoost |
| Data | pandas, numpy, pyarrow |
| Visualisation | matplotlib, seaborn |
| App | Streamlit |
| Testing | pytest, pytest-cov |
| CI/CD | GitHub Actions |
| Containerisation | Docker |

---

## About This Project

This project was built as a complete end-to-end machine learning system —
from raw transactional data through to a deployed interactive application.
Every design decision prioritises production readiness: reproducible pipelines,
honest model evaluation, and results that a non-technical stakeholder can act on.

**Author:** Khalida — [GitHub](https://github.com/Khalida-DS)
