"""
src/pipelines/train.py
======================
End-to-end training pipeline orchestrator.

Usage
-----
  python -m src.pipelines.train                     # run all stages
  python -m src.pipelines.train --stage preprocess  # preprocessing only
  python -m src.pipelines.train --stage classify    # classification only
  python -m src.pipelines.train --stage regress     # regression only
  python -m src.pipelines.train --stage cluster     # clustering only

Run individual stages when iterating: you don't need to re-run preprocessing
every time you tweak a classifier hyperparameter.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


@contextmanager
def _timer(name: str):
    t0 = time.time()
    log.info(f"▶  {name}")
    yield
    log.info(f"✔  {name}  ({time.time() - t0:.1f}s)")


# ── Stage functions ───────────────────────────────────────────────────────────

def stage_preprocess(cfg) -> tuple:
    from src.features.preprocessing import run_preprocessing

    with _timer("Preprocessing"):
        X, df_hh = run_preprocessing(cfg.paths.raw_data, cfg=cfg)

        Path(cfg.paths.processed_data).parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cfg.paths.processed_data)
        df_hh.to_parquet(cfg.paths.household_data)

        log.info(f"   Feature matrix:  {X.shape}")
        log.info(f"   Households:      {df_hh.shape}")

    return X, df_hh


def stage_classify(X: pd.DataFrame, df_hh: pd.DataFrame, cfg) -> pd.DataFrame:
    from src.models.classifier import train_all_products

    with _timer("Classification"):
        summary = train_all_products(X, df_hh, cfg=cfg)
        if not summary.empty:
            log.info(f"\n{summary[['product','model','roc_auc','lift_at_10']].to_string()}")

    return summary


def stage_regress(X: pd.DataFrame, df_hh: pd.DataFrame, cfg) -> dict:
    from src.models.regressor import train_cost_regressor

    target_col = cfg.regression.target_col
    if target_col not in df_hh.columns:
        candidates = [c for c in df_hh.columns if "Total Cost" in c]
        if candidates:
            target_col = candidates[0]
            log.warning(f"'{cfg.regression.target_col}' not found; using '{target_col}'")
        else:
            log.error("No cost target column found — skipping regression")
            return {}

    y = df_hh[target_col].fillna(0).values

    # ── Remove ALL cost-related columns from X before regression ──────────
    # These columns are components of the target — keeping them is leakage.
    # Cost 2014, Cost 2015, ERS Member Cost Year 1, avg_cost_per_call, etc.
    # all derive from the same service call records that produce Total Cost.
    leakage_patterns = [
    "Cost 20",           # annual cost columns: Cost 2014, Cost 2015...
    "ERS Member Cost",   # raw annual ERS cost columns
    "avg_cost_per_call", # derived from cost amounts
    "Total Cost",        # the target itself
    "cost_trend",        # derived from annual cost columns — leakage
     ]
    
    leak_cols = [
        c for c in X.columns
        if any(pattern in c for pattern in leakage_patterns)
    ]
    if leak_cols:
        log.info(f"   Removing {len(leak_cols)} leakage columns from regression features:")
        for c in leak_cols:
            log.info(f"     - {c}")
    X_reg = X.drop(columns=leak_cols)

    with _timer("Regression"):
        results = train_cost_regressor(X_reg, y, cfg=cfg)

    return results


def stage_cluster(X: pd.DataFrame, df_hh: pd.DataFrame, cfg) -> None:
    from src.models.clustering import run_clustering

    # ── Build propensity score matrix ──────────────────────────────────────
    # Load saved classifiers and predict probabilities for each product.
    # This is what we should cluster on — likelihood to buy, not current ownership.
    import joblib, os, glob
    import numpy as np

    cfg_data     = cfg.data
    product_names = cfg_data.product_names   # clean names: 'INS Client' etc.
    model_dir    = str(cfg.paths.abs('model_dir'))

    propensity_cols = {}
    for product in product_names:
        # Find the best saved model for this product
        safe     = product.lower().replace(" ", "_")
        pattern  = os.path.join(model_dir, f"{safe}_*.pkl")
        matches  = sorted(glob.glob(pattern))
        if not matches:
            log.warning(f"No saved model for '{product}' — skipping")
            continue
        # Pick the first match (models are saved with best estimator name)
        model_path = matches[0]
        try:
            model  = joblib.load(model_path)
            proba  = model.predict_proba(X)[:, 1]   # probability of buying
            propensity_cols[f"propensity_{safe}"] = proba
            log.info(f"   Loaded propensity scores: {product}  "
                     f"(mean={proba.mean():.3f}  max={proba.max():.3f})")
        except Exception as e:
            log.error(f"Could not score '{product}': {e}")

    if len(propensity_cols) < 2:
        log.error("Not enough propensity models found — cannot cluster")
        return

    # Build propensity DataFrame — one column per product
    import pandas as pd
    X_prop = pd.DataFrame(propensity_cols, index=X.index)

    log.info(f"   Clustering on {X_prop.shape[1]} propensity dimensions "
             f"across {X_prop.shape[0]} households")

    with _timer("Clustering"):
        return run_clustering(X_prop, df_hh, cfg=cfg)


def stage_recommend(
    df_hh:        pd.DataFrame,
    labels:       "np.ndarray",
    action_table: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Attach cluster label and recommended product to each household."""
    with _timer("Recommendations"):
        rec_df = df_hh.copy()
        rec_df["cluster"] = labels

        cluster_to_prod = (
            action_table.set_index("cluster")["recommended_product"].to_dict()
        )
        rec_df["recommended_product"] = rec_df["cluster"].map(cluster_to_prod)

        out_path = cfg.paths.recommendations
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        rec_df.to_parquet(out_path)

        log.info(f"   Recommendations saved to {out_path}")
        log.info(f"   Distribution:\n{rec_df['recommended_product'].value_counts().to_string()}")

    return rec_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AAA Northeast — Training Pipeline")
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "preprocess", "classify", "regress", "cluster", "recommend"],
    )
    args = parser.parse_args()

    from src.config import get_config
    cfg = get_config(Path(args.config))
    log.info(f"Project: {cfg.name}  v{cfg.version}  seed={cfg.random_seed}")

    # Create output directories
    os.makedirs(cfg.paths.model_dir,  exist_ok=True)
    os.makedirs(cfg.paths.report_dir, exist_ok=True)

    # ── Load or compute features ───────────────────────────────────────────────
    if args.stage in ("all", "preprocess"):
        X, df_hh = stage_preprocess(cfg)
    else:
        log.info("Loading pre-computed features from parquet...")
        X     = pd.read_parquet(cfg.paths.processed_data)
        df_hh = pd.read_parquet(cfg.paths.household_data)

    if args.stage in ("all", "classify"):
        stage_classify(X, df_hh, cfg)

    if args.stage in ("all", "regress"):
        stage_regress(X, df_hh, cfg)

    labels = action_table = None
    if args.stage in ("all", "cluster"):
        labels, profile, action_table = stage_cluster(X, df_hh, cfg)

    if args.stage in ("all", "recommend"):
        if labels is None:
            raise RuntimeError("Run --stage cluster first to obtain labels.")
        stage_recommend(df_hh, labels, action_table, cfg)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()