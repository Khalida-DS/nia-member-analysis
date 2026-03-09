"""
src/models/clustering.py
========================
Member segmentation via K-Means on predicted propensity scores.

Why cluster on propensity scores instead of raw features?
----------------------------------------------------------
Raw feature clustering groups members by who they ARE (demographics).
Propensity score clustering groups members by what they are LIKELY TO DO.
A cluster of "high credit-card propensity, low insurance propensity" directly
answers: lead with the credit card offer for this segment.

Pipeline
--------
1. Evaluate k=2..11: record inertia + silhouette for each k
2. Plot elbow + silhouette: choose k where silhouette > 0.30
3. Fit final K-Means
4. Profile clusters: mean features per cluster
5. Name segments: human-readable labels based on dominant signals
6. Build action table: cluster → recommended product
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig, get_config
from src.evaluation.plots import plot_cluster_profiles, plot_elbow_silhouette


def evaluate_k_range(
    X:       np.ndarray,
    k_range: List[int],
    seed:    int = 42,
) -> pd.DataFrame:
    """
    Fit K-Means for each k in k_range.
    Returns DataFrame with inertia, silhouette, calinski_harabasz per k.
    """
    rows = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k":                  k,
            "inertia":            km.inertia_,
            "silhouette":         silhouette_score(X, labels,
                                                   sample_size=min(1000, len(X))),
            "calinski_harabasz":  calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(rows).set_index("k")


def fit_kmeans(
    X:    np.ndarray,
    k:    int,
    seed: int = 42,
) -> Tuple[KMeans, np.ndarray]:
    """Fit the final K-Means model and return (model, labels)."""
    km     = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"\n  K-Means fitted: k={k}")
    print(f"  Silhouette score: {sil:.4f}  (target > 0.30)")
    print(f"  Cluster sizes:\n{pd.Series(labels).value_counts().sort_index().to_string()}")
    return km, labels


def profile_clusters(
    df_hh:         pd.DataFrame,
    labels:        np.ndarray,
    product_names: List[str],
) -> pd.DataFrame:
    """
    Build a cluster profile DataFrame: mean feature values per cluster.
    Attach cluster labels to df_hh before profiling.
    """
    df = df_hh.copy()
    df["cluster"] = labels

    profile_cols = [p for p in product_names if p in df.columns]
    demo_cols    = [
        "Income mean", "Credit Ranges mean",
        "Member Tenure Years mean", "Number of Children mean",
        "product_count", "total_ers_calls",
        "is_high_income", "is_long_term_member", "has_used_ers",
    ]
    profile_cols += [c for c in demo_cols if c in df.columns]

    profile = df.groupby("cluster")[profile_cols].mean().round(4)
    profile["cluster_size"] = df.groupby("cluster").size()
    return profile


def name_clusters(
    profile:       pd.DataFrame,
    product_names: List[str],
) -> Dict[int, str]:
    """
    Assign human-readable segment names based on dominant signals.
    These names are heuristic — domain experts should validate them.
    """
    names = {}
    for cid, row in profile.iterrows():
        prods    = [p for p in product_names if p in row.index]
        dominant = max(prods, key=lambda p: row.get(p, 0)) if prods else None
        multi    = row.get("product_count",  0)
        ers      = row.get("total_ers_calls", 0)
        income   = row.get("Income mean",     0)
        tenure   = row.get("Member Tenure Years mean", 0)

        if multi >= 2.5:
            name = "Loyal Multi-Product Members"
        elif ers > 2.0:
            name = "High ERS Utilizers"
        elif income > 100_000:
            name = "High-Income Prospects"
        elif tenure > 15:
            name = "Long-Tenure Single-Product"
        elif dominant and row.get(dominant, 0) > 0.30:
            name = f"{dominant} Aficionados"
        else:
            name = f"General Segment {cid}"

        names[int(cid)] = name
    return names


def build_action_table(
    profile:        pd.DataFrame,
    cluster_names:  dict,
    product_names:  list,
    min_propensity: float = 0.15,
) -> pd.DataFrame:
    rows = []
    for cid, row in profile.iterrows():
        best_product = None
        best_score   = 0.0
        for p in product_names:
            score = float(row.get(p, 0.0))
            if score > best_score:
                best_score   = score
                best_product = p
        if best_product is None:
            best_product = product_names[0] if product_names else "Unknown"
        rows.append({
            "cluster":             int(cid),
            "cluster_name":        cluster_names.get(int(cid), f"Segment {cid}"),
            "cluster_size":        int(row.get("cluster_size", 0)),
            "recommended_product": best_product if best_score >= min_propensity
                                   else "Nurture — no immediate offer",
            "avg_propensity":      round(float(best_score), 4),
            "avg_income":          round(float(row.get("Income mean", 0)), 0),
            "avg_tenure_years":    round(float(row.get("Member Tenure Years mean", 0)), 1),
            "avg_product_count":   round(float(row.get("product_count", 0)), 2),
            "pct_ers_users":       round(float(row.get("has_used_ers", 0)), 3),
        })
    return pd.DataFrame(rows).sort_values("avg_propensity", ascending=False)


def run_clustering(
    X_prop:        pd.DataFrame,
    df_hh:         pd.DataFrame,
    product_names: Optional[List[str]] = None,
    cfg:           Optional[ProjectConfig] = None,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Full clustering pipeline.

    Parameters
    ----------
    df_hh         : household DataFrame (must contain product propensity columns)
    product_names : list of product column names to cluster on
    cfg           : ProjectConfig

    Returns
    -------
    labels       : cluster assignment per household (np.ndarray)
    profile      : cluster profile DataFrame
    action_table : segment-to-product recommendation table
    """
    if cfg is None:
        cfg = get_config()
    if product_names is None:
        product_names = cfg.data.product_names

    seed    = cfg.random_seed
    k_range = cfg.clustering.k_range
    final_k = cfg.clustering.final_k
    fig_dir = cfg.paths.report_dir

    # X_prop is the propensity score matrix passed from the pipeline
    X_cluster = X_prop.values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Evaluate k range
    print("\n  Evaluating k range...")
    k_metrics = evaluate_k_range(X_scaled, k_range, seed=seed)
    print(k_metrics.to_string())

    plot_elbow_silhouette(
        k_values    = list(k_metrics.index),
        inertias    = k_metrics["inertia"].tolist(),
        silhouettes = k_metrics["silhouette"].tolist(),
        save_path   = f"{fig_dir}/clustering_elbow_silhouette.png",
    )

    # Fit final model
    km, labels = fit_kmeans(X_scaled, k=final_k, seed=seed)

    # Profile
    profile       = profile_clusters(df_hh, labels, product_names)
    cluster_names = name_clusters(profile, product_names)

    print("\n  Cluster Names:")
    for cid, name in cluster_names.items():
        size = int(profile.loc[cid, "cluster_size"]) if cid in profile.index else 0
        print(f"    {cid}: {name}  (n={size})")

    # Visualise
    vis_cols = [p for p in product_names if p in profile.columns][:8]
    plot_cluster_profiles(
        df_hh.assign(cluster=labels),
        feature_cols = vis_cols,
        save_path    = f"{fig_dir}/cluster_profiles.png",
    )

    # Action table
    action_table = build_action_table(profile, cluster_names, product_names)
    print("\n  Segment Action Table:")
    print(action_table.to_string(index=False))

    return labels, profile, action_table