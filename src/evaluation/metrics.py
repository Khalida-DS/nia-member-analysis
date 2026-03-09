"""
src/evaluation/metrics.py
=========================
Centralised evaluation functions used by EVERY model in this project.

Why centralised?
----------------
If each model defines its own AUC calculation, you get subtle differences
(threshold choices, probability calibration) that make cross-model comparisons
meaningless.  One module, one standard.

Primary business metric: Lift @ Top 10%
----------------------------------------
Marketing has a fixed budget for outreach.  They contact the top-N households
by predicted score.  Lift@10% tells them: if we contact the top 10% of
households, how many times more buyers do we find vs. random outreach?
Target: > 2.0×   (doubles yield per dollar spent)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ── Classification ────────────────────────────────────────────────────────────

def lift_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k:      float = 0.10,
) -> float:
    """
    Lift at top-k% of population contacted.

    Example
    -------
    k=0.10, 3500 households, 700 buyers (20% base rate):
    - Random: contact 350 → find 70 buyers
    - Model Lift@10%=2.5: contact 350 → find 175 buyers

    Parameters
    ----------
    y_true : binary ground truth labels
    y_prob : predicted probabilities for the positive class
    k      : fraction of population to contact (default 0.10 = top 10%)

    Returns
    -------
    float : lift multiplier (1.0 = same as random, 2.0 = twice as good)
    """
    df = (
        pd.DataFrame({"y": y_true, "p": y_prob})
        .sort_values("p", ascending=False)
    )
    n_contact  = max(1, int(len(df) * k))
    rate_model = df.head(n_contact)["y"].mean()
    rate_base  = df["y"].mean()
    return float(rate_model / rate_base) if rate_base > 0 else 0.0


def classification_scorecard(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_prob:     np.ndarray,
    model_name: str,
    label:      str = "",
) -> Dict[str, Any]:
    """
    Compute the full classification scorecard.

    Returns a dict with: roc_auc, f1, precision, recall, lift_at_10,
    plus the raw sklearn classification_report string.
    """
    return {
        "model":        model_name,
        "label":        label,
        "roc_auc":      round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1":           round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision":    round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":       round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "lift_at_10":   round(lift_at_k(y_true, y_prob, k=0.10), 3),
        "class_report": classification_report(y_true, y_pred, zero_division=0),
    }


def print_classification_scorecard(scores: Dict[str, Any]) -> None:
    lbl = f" | {scores['label']}" if scores["label"] else ""
    print(f"\n{'═' * 55}")
    print(f"  {scores['model']}{lbl}")
    print(f"{'═' * 55}")
    print(f"  ROC-AUC:     {scores['roc_auc']:.4f}   (target > 0.70)")
    print(f"  F1:          {scores['f1']:.4f}")
    print(f"  Precision:   {scores['precision']:.4f}")
    print(f"  Recall:      {scores['recall']:.4f}")
    print(f"  Lift @ 10%:  {scores['lift_at_10']:.3f}×  (target > 2.0×)")
    print(f"\n{scores['class_report']}")


# ── Regression ────────────────────────────────────────────────────────────────

def regression_scorecard(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str,
    label:      str = "",
) -> Dict[str, Any]:
    """Compute RMSE, MAE, R² for a regression model."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "model": model_name,
        "label": label,
        "rmse":  round(rmse, 4),
        "mae":   round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2":    round(float(r2_score(y_true, y_pred)), 4),
    }


def print_regression_scorecard(scores: Dict[str, Any]) -> None:
    lbl = f" | {scores['label']}" if scores["label"] else ""
    print(f"\n{'═' * 45}")
    print(f"  {scores['model']}{lbl}")
    print(f"{'═' * 45}")
    print(f"  RMSE:  {scores['rmse']:.4f}")
    print(f"  MAE:   {scores['mae']:.4f}")
    print(f"  R²:    {scores['r2']:.4f}   (target > 0.25)")


# ── Model persistence ──────────────────────────────────────────────────────────

def save_model(
    model:         Any,
    label:         str,
    scores:        Dict[str, Any],
    model_dir:     str,
    feature_names: Optional[List[str]] = None,
) -> str:
    """
    Serialize model (.pkl) + self-documenting metadata (.json).

    The metadata JSON means anyone can inspect a saved model without
    loading the pickle: what product it was trained for, when, with
    what metrics, on what features.

    Returns
    -------
    str : path to the saved .pkl file
    """
    import joblib

    os.makedirs(model_dir, exist_ok=True)
    slug = label.lower().replace(" ", "_")

    pkl_path  = os.path.join(model_dir, f"{slug}.pkl")
    meta_path = os.path.join(model_dir, f"{slug}_metadata.json")

    joblib.dump(model, pkl_path)

    meta = {
        "label":         label,
        "trained_at":    datetime.utcnow().isoformat() + "Z",
        "feature_count": len(feature_names) if feature_names else None,
        "feature_names": feature_names,
        **{k: v for k, v in scores.items() if k != "class_report"},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Model saved: {pkl_path}")
    return pkl_path