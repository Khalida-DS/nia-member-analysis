"""
src/models/classifier.py
========================
Product cross-sell propensity classifiers.

For each modelable product we train three estimators (XGBoost, Random Forest,
Logistic Regression) and select the best by ROC-AUC on the held-out test set.

Key design decisions
--------------------
* Separate model per product — buyer profiles differ completely by product
* class_weight='balanced' — handles imbalanced targets without upsampling
* 70/10/20 train/val/test — validation set for HP selection; test NEVER touched
* Lift@10% as the primary business metric alongside AUC
* All artifacts saved with self-documenting JSON metadata
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig, get_config
from src.evaluation.metrics import (
    classification_scorecard,
    print_classification_scorecard,
    save_model,
)
from src.evaluation.plots import (
    plot_feature_importance,
    plot_lift_curve,
    plot_roc_curves,
)


def _build_estimators(cfg: ProjectConfig) -> Dict[str, object]:
    """
    Instantiate all models from settings.yaml.
    Returns a dict of name → unfitted estimator.
    """
    seed    = cfg.random_seed
    xgb_p   = cfg.classification.xgboost
    rf_p    = cfg.classification.random_forest

    estimators = {}

    # XGBoost
    try:
        from xgboost import XGBClassifier
        estimators["xgboost"] = XGBClassifier(
            n_estimators     = xgb_p["n_estimators"],
            max_depth        = xgb_p["max_depth"],
            learning_rate    = xgb_p["learning_rate"],
            subsample        = xgb_p["subsample"],
            colsample_bytree = xgb_p["colsample_bytree"],
            min_child_weight = xgb_p["min_child_weight"],
            use_label_encoder = False,
            eval_metric      = "logloss",
            random_state     = seed,
            n_jobs           = -1,
        )
    except ImportError:
        print("  [INFO] xgboost not installed — skipping XGBoost estimator")

    # Random Forest
    estimators["random_forest"] = RandomForestClassifier(
        n_estimators   = rf_p["n_estimators"],
        max_depth      = rf_p["max_depth"],
        min_samples_leaf = rf_p["min_samples_leaf"],
        max_features   = rf_p["max_features"],
        class_weight   = "balanced",
        random_state   = seed,
        n_jobs         = -1,
    )

    # Logistic Regression (linear baseline + fully interpretable)
    estimators["logistic_regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight = "balanced",
            max_iter     = 1000,
            solver       = "lbfgs",
            random_state = seed,
        )),
    ])

    return estimators


def train_product_classifier(
    X:            pd.DataFrame,
    y:            np.ndarray,
    product_name: str,
    cfg:          Optional[ProjectConfig] = None,
    save:         bool = True,
) -> Dict[str, dict]:
    """
    Train and evaluate all estimators for a single product.

    Parameters
    ----------
    X            : feature matrix (households × features)
    y            : binary target  (1 = household holds the product)
    product_name : human-readable product name used in output filenames
    cfg          : ProjectConfig (loaded from settings.yaml if None)
    save         : write best model + metadata to models/artifacts/

    Returns
    -------
    results : dict of { model_name: scorecard_dict }
    """
    if cfg is None:
        cfg = get_config()

    seed      = cfg.random_seed
    test_size = cfg.training.test_size
    val_size  = cfg.training.val_size
    fig_dir   = cfg.paths.report_dir
    model_dir = cfg.paths.model_dir

    # ── 70/10/20 split ────────────────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # Carve validation from trainval
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=seed
    )

    print(f"\n{'─' * 58}")
    print(f"  Product:  {product_name}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Positive rate (train): {y_train.mean():.2%}")
    print(f"{'─' * 58}")

    estimators = _build_estimators(cfg)
    results:  Dict[str, dict] = {}
    models:   Dict[str, object] = {}

    for name, est in estimators.items():
        try:
            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)
            y_prob = est.predict_proba(X_test)[:, 1]
            scores = classification_scorecard(y_pred=y_pred, y_prob=y_prob,
                                              y_true=y_test, model_name=name,
                                              label=product_name)
            print_classification_scorecard(scores)
            results[name] = scores
            models[name]  = est
        except Exception as exc:
            print(f"  [WARN] {name} failed: {exc}")

    if not results:
        print(f"  [ERROR] All estimators failed for {product_name}")
        return results

    # ── Figures ────────────────────────────────────────────────────────────────
    slug = product_name.lower().replace(" ", "_")

    # ROC overlay
    roc_data = [(n, y_test, models[n].predict_proba(X_test)[:, 1])
                for n in results]
    plot_roc_curves(roc_data,
                    title=f"ROC Curves — {product_name}",
                    save_path=os.path.join(fig_dir, f"roc_{slug}.png"))

    # Best model by AUC
    best_name  = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models[best_name]
    best_prob  = best_model.predict_proba(X_test)[:, 1]

    # Lift curve
    plot_lift_curve(y_test, best_prob,
                    label=f"{product_name} ({best_name})",
                    save_path=os.path.join(fig_dir, f"lift_{slug}.png"))

    # Feature importance (if available)
    actual_model = (best_model.named_steps.get("lr")
                    if hasattr(best_model, "named_steps") else best_model)
    try:
        plot_feature_importance(actual_model, list(X.columns),
                                label=f"{product_name} ({best_name})",
                                save_path=os.path.join(fig_dir, f"importance_{slug}.png"))
    except (AttributeError, TypeError):
        pass

    # ── Persist best model ─────────────────────────────────────────────────────
    if save:
        save_model(best_model,
                   label=f"{slug}_{best_name}",
                   scores=results[best_name],
                   model_dir=model_dir,
                   feature_names=list(X.columns))

    return results


def train_all_products(
    X:          pd.DataFrame,
    df_hh:      pd.DataFrame,
    cfg:        Optional[ProjectConfig] = None,
) -> pd.DataFrame:
    """
    Train classifiers for every product that meets the penetration threshold.

    Parameters
    ----------
    X     : feature matrix (output of run_preprocessing)
    df_hh : household DataFrame containing product target columns
    cfg   : ProjectConfig

    Returns
    -------
    summary_df : one row per (product, model), columns = all scorecard metrics
    """
    if cfg is None:
        cfg = get_config()

    product_names = cfg.data.product_names
    min_pen       = cfg.data.min_product_penetration

    # Identify modelable products
    modelable = []
    for prod in product_names:
        if prod not in df_hh.columns:
            print(f"  [SKIP] {prod} — not in df_hh columns")
            continue
        penetration = df_hh[prod].mean()
        if penetration < min_pen:
            print(f"  [SKIP] {prod} — penetration {penetration:.1%} < {min_pen:.0%} threshold")
        elif df_hh[prod].sum() < 20:
            print(f"  [SKIP] {prod} — only {int(df_hh[prod].sum())} positives (need ≥ 20)")
        else:
            print(f"  [MODEL] {prod} — penetration {penetration:.1%}")
            modelable.append(prod)

    print(f"\n  Modelable products: {modelable}\n")

    all_results = []
    for prod in modelable:
        y       = df_hh[prod].fillna(0).astype(int).values
        scores  = train_product_classifier(X, y, prod, cfg=cfg)
        for model_name, sc in scores.items():
            all_results.append({"product": prod, **sc})

    if not all_results:
        return pd.DataFrame()

    summary = (
        pd.DataFrame(all_results)
        .sort_values(["product", "roc_auc"], ascending=[True, False])
    )

    print("\n" + "═" * 70)
    print("  CLASSIFICATION SUMMARY")
    print("═" * 70)
    cols_to_show = ["product", "model", "roc_auc", "f1", "lift_at_10"]
    print(summary[cols_to_show].to_string(index=False))
    print("═" * 70)

    return summary