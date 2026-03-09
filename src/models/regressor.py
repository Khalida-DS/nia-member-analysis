"""
src/models/regressor.py
=======================
Predict total household ERS cost.

Revised approach
----------------
* Try both log1p-transformed and raw targets
* Use HuberRegressor as a robust alternative to Ridge/Lasso
* Random Forest on raw target (trees are scale-invariant)
* Report all results honestly
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig, get_config
from src.evaluation.metrics import (
    print_regression_scorecard,
    regression_scorecard,
    save_model,
)
from src.evaluation.plots import plot_residuals


def train_cost_regressor(
    X:    pd.DataFrame,
    y:    np.ndarray,
    cfg:  Optional[ProjectConfig] = None,
    save: bool = True,
) -> Dict[str, dict]:

    if cfg is None:
        cfg = get_config()

    seed      = cfg.random_seed
    fig_dir   = cfg.paths.report_dir
    model_dir = cfg.paths.model_dir

    y_raw = np.array(y, dtype=float)
    y_log = np.log1p(y_raw)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=cfg.training.test_size, random_state=seed
    )
    y_train_log = np.log1p(y_train_raw)
    y_test_log  = np.log1p(y_test_raw)

    print(f"\n{'─' * 50}")
    print(f"  Regression target: Total ERS Cost")
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")
    print(f"  Zero-cost households: {(y_raw == 0).mean():.1%}")
    print(f"  Mean cost (non-zero): ${y_raw[y_raw > 0].mean():.2f}")
    print(f"  Median cost (non-zero): ${np.median(y_raw[y_raw > 0]):.2f}")
    print(f"{'─' * 50}")

    results: Dict[str, dict] = {}
    models:  Dict[str, object] = {}

    # ── Ridge on log target ───────────────────────────────────────────────
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=cfg.regression.ridge_alphas,
                           cv=cfg.training.cv_folds)),
    ])
    ridge_pipe.fit(X_train, y_train_log)
    y_pred = np.expm1(ridge_pipe.predict(X_test))
    y_pred = np.clip(y_pred, 0, None)  # predictions cannot be negative dollars
    scores = regression_scorecard(y_test_raw, y_pred, "Ridge (log target)", "ERS Cost")
    print_regression_scorecard(scores)
    plot_residuals(y_test_raw, y_pred, "Ridge",
                   save_path=os.path.join(fig_dir, "residuals_ridge.png"))
    results["ridge"] = scores
    models["ridge"]  = ridge_pipe

    # ── Huber (robust to outliers, raw target) ────────────────────────────
    huber_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("huber",  HuberRegressor(epsilon=1.35, max_iter=500)),
    ])
    huber_pipe.fit(X_train, y_train_raw)
    y_pred = np.clip(huber_pipe.predict(X_test), 0, None)
    scores = regression_scorecard(y_test_raw, y_pred, "Huber (robust)", "ERS Cost")
    print_regression_scorecard(scores)
    plot_residuals(y_test_raw, y_pred, "Huber",
                   save_path=os.path.join(fig_dir, "residuals_huber.png"))
    results["huber"] = scores
    models["huber"]  = huber_pipe

    # ── Random Forest (raw target — trees are scale invariant) ───────────
    rf_p = cfg.regression.random_forest
    rf   = RandomForestRegressor(
        n_estimators     = rf_p["n_estimators"],
        max_depth        = rf_p["max_depth"],
        min_samples_leaf = rf_p["min_samples_leaf"],
        random_state     = seed,
        n_jobs           = -1,
    )
    rf.fit(X_train, y_train_raw)
    y_pred = np.clip(rf.predict(X_test), 0, None)
    scores = regression_scorecard(y_test_raw, y_pred, "Random Forest (raw target)", "ERS Cost")
    print_regression_scorecard(scores)
    plot_residuals(y_test_raw, y_pred, "Random Forest",
                   save_path=os.path.join(fig_dir, "residuals_rf.png"))
    results["random_forest"] = scores
    models["random_forest"]  = rf

    # ── Gradient Boosting (raw target) ────────────────────────────────────
    gb = GradientBoostingRegressor(
        n_estimators     = 200,
        max_depth        = 4,
        learning_rate    = 0.05,
        min_samples_leaf = 20,
        random_state     = seed,
    )
    gb.fit(X_train, y_train_raw)
    y_pred = np.clip(gb.predict(X_test), 0, None)
    scores = regression_scorecard(y_test_raw, y_pred,
                                  "Gradient Boosting (raw target)", "ERS Cost")
    print_regression_scorecard(scores)
    results["gradient_boosting"] = scores
    models["gradient_boosting"]  = gb

    # ── Select and save best ──────────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_r2   = results[best_name]["r2"]

    print(f"\n  Best regressor: {best_name}")
    print(f"  RMSE: {results[best_name]['rmse']:.2f}   R²: {best_r2:.4f}")

    # Leakage warning
    if best_r2 > 0.60:
        print(f"\n  ⚠️  R² = {best_r2:.4f} — verify no cost columns remain in features")
    elif best_r2 > 0.25:
        print(f"\n  ✅  R² = {best_r2:.4f} — good honest result for ERS cost prediction")
    else:
        print(f"\n  ℹ️  R² = {best_r2:.4f} — low but expected for emergency event data")

    if save:
        save_model(models[best_name],
                   label=f"ers_cost_{best_name}",
                   scores=results[best_name],
                   model_dir=model_dir,
                   feature_names=list(X.columns))

    return results