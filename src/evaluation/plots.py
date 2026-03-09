"""
src/evaluation/plots.py
=======================
All project visualisations in one place.

Every function contract:
    plot_*(data, ..., save_path=None) -> matplotlib.Figure
If save_path is provided: figure is saved to disk and axes are closed.
If save_path is None: figure is returned for notebook display.
This pattern makes the same functions work in both notebooks and pipelines.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# ── Style ─────────────────────────────────────────────────────────────────────
PRIMARY   = "#2E4A8C"
SECONDARY = "#E84C3D"
POSITIVE  = "#27AE60"
WARNING   = "#F39C12"
NEUTRAL   = "#7F8C8D"
DPI       = 150

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


def _save(fig: plt.Figure, path: Optional[str]) -> plt.Figure:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── EDA ───────────────────────────────────────────────────────────────────────

def plot_missing_values(
    df:        pd.DataFrame,
    top_n:     int = 25,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N columns by missing percentage."""
    from src.features.preprocessing import audit_missing
    report = audit_missing(df).head(top_n)

    fig, ax = plt.subplots(figsize=(11, max(5, len(report) * 0.35)))
    bars = ax.barh(report.index, report["missing_pct"], color=PRIMARY, edgecolor="white")
    ax.axvline(x=50, color=SECONDARY, linestyle="--", alpha=0.7, label="50% line")
    for bar, val in zip(bars, report["missing_pct"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Missing (%)", fontsize=11)
    ax.set_title(f"Top {top_n} Columns by Missing Rate", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, save_path)


def plot_product_adoption(
    df:           pd.DataFrame,
    product_names: List[str],
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """Bar chart of product adoption rates with 5% modeling threshold line."""
    present  = [p for p in product_names if p in df.columns]
    adoption = df[present].mean().sort_values()
    colors   = [PRIMARY if v >= 0.05 else "#AABDDB" for v in adoption.values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(adoption.index, adoption.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, adoption.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)
    ax.axvline(x=0.05, color=SECONDARY, linestyle="--", alpha=0.7, label="5% modeling threshold")
    ax.set_xlabel("Proportion of Households", fontsize=11)
    ax.set_title("Product Adoption — Active Households", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, save_path)


def plot_correlation_matrix(
    df:        pd.DataFrame,
    top_n:     int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Correlation heatmap for top-N highest-variance numeric features."""
    numeric  = df.select_dtypes(include=[np.number])
    top_cols = numeric.var().nlargest(top_n).index
    corr     = numeric[top_cols].corr()
    mask     = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"Feature Correlation (top {top_n} by variance)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return _save(fig, save_path)


# ── Classification ────────────────────────────────────────────────────────────

def plot_roc_curves(
    models_results: List[Tuple[str, np.ndarray, np.ndarray]],
    title:     str = "ROC Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    models_results : list of (model_name, y_true, y_prob)
    """
    colors = [PRIMARY, SECONDARY, POSITIVE, WARNING, NEUTRAL]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, y_true, y_prob) in enumerate(models_results):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name}  (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    return _save(fig, save_path)


def plot_lift_curve(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    label:     str,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, float]:
    """
    Lift curve with annotated Lift@10% value.

    Returns (figure, lift_at_10_value)
    """
    df = (
        pd.DataFrame({"y": y_true, "p": y_prob})
        .sort_values("p", ascending=False)
        .reset_index(drop=True)
    )
    n = len(df)
    df["pct_pop"]      = (df.index + 1) / n
    df["pct_captured"] = df["y"].cumsum() / max(df["y"].sum(), 1)
    df["lift"]         = df["pct_captured"] / df["pct_pop"]

    lift_at_10 = float(df[df["pct_pop"] <= 0.10]["lift"].iloc[-1]) if (df["pct_pop"] <= 0.10).any() else 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["pct_pop"], df["lift"], color=PRIMARY, lw=2.5)
    ax.axhline(y=1.0, color=NEUTRAL, linestyle="--", alpha=0.6, label="Random (1×)")
    ax.axvline(x=0.10, color=SECONDARY, linestyle=":", lw=1.8,
               label=f"Top 10%  →  Lift = {lift_at_10:.2f}×")
    ax.annotate(
        f"{lift_at_10:.2f}×",
        xy=(0.10, lift_at_10),
        xytext=(0.18, lift_at_10 + 0.4),
        arrowprops={"arrowstyle": "->", "color": SECONDARY},
        fontsize=11, color=SECONDARY, fontweight="bold",
    )
    ax.set_xlabel("Proportion of Population Contacted", fontsize=11)
    ax.set_ylabel("Lift", fontsize=11)
    ax.set_title(f"Lift Curve — {label}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    return _save(fig, save_path), lift_at_10


def plot_feature_importance(
    model:         Any,
    feature_names: List[str],
    label:         str,
    top_n:         int = 20,
    save_path:     Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{type(model).__name__} has no feature_importances_")

    imp = (
        pd.Series(model.feature_importances_, index=feature_names)
        .nlargest(top_n)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    bars = ax.barh(imp.index, imp.values, color=PRIMARY, edgecolor="white")
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {label}",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return _save(fig, save_path)


# ── Regression ────────────────────────────────────────────────────────────────

def plot_residuals(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str,
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """Three-panel residual diagnostic: scatter, histogram, Q-Q plot."""
    import scipy.stats as stats

    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.35, color=PRIMARY, s=18)
    axes[0].axhline(0, color=SECONDARY, linestyle="--", lw=1.5)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    axes[1].hist(residuals, bins=40, color=PRIMARY, edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot (Normality Check)")

    fig.suptitle(f"Residual Diagnostics — {model_name}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, save_path)


# ── Clustering ────────────────────────────────────────────────────────────────

def plot_elbow_silhouette(
    k_values:   List[int],
    inertias:   List[float],
    silhouettes: List[float],
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """Side-by-side elbow and silhouette charts for choosing k."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    best_k = k_values[int(np.argmax(silhouettes))]

    axes[0].plot(k_values, inertias, "o-", color=PRIMARY, lw=2, markersize=7)
    axes[0].set_xlabel("k", fontsize=11)
    axes[0].set_ylabel("Inertia (within-cluster SSE)", fontsize=11)
    axes[0].set_title("Elbow Method", fontsize=12, fontweight="bold")

    axes[1].plot(k_values, silhouettes, "s-", color=POSITIVE, lw=2, markersize=7)
    axes[1].axvline(x=best_k, color=SECONDARY, linestyle="--", lw=1.5,
                    label=f"Best k = {best_k}")
    axes[1].set_xlabel("k", fontsize=11)
    axes[1].set_ylabel("Silhouette Score", fontsize=11)
    axes[1].set_title("Silhouette Score by k", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)

    fig.suptitle("Optimal Number of Clusters", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, save_path)


def plot_cluster_profiles(
    df_hh:        pd.DataFrame,
    feature_cols: List[str],
    save_path:    Optional[str] = None,
) -> plt.Figure:
    """Heatmap of normalised cluster mean values for selected features."""
    pivot = df_hh.groupby("cluster")[feature_cols].mean().T
    denom = pivot.max(axis=1) - pivot.min(axis=1)
    denom[denom == 0] = 1
    pivot_norm = (pivot.sub(pivot.min(axis=1), axis=0)).div(denom, axis=0)

    fig, ax = plt.subplots(figsize=(12, max(5, len(feature_cols) * 0.5)))
    sns.heatmap(pivot_norm, annot=pivot.round(2), fmt=".2f",
                cmap="Blues", linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_title("Cluster Profiles (normalised means)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return _save(fig, save_path)