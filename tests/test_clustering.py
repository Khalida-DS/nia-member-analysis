"""
tests/test_clustering.py
========================
Unit tests for src/models/clustering.py.
Run: pytest tests/test_clustering.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.models.clustering import (
    build_action_table,
    evaluate_k_range,
    fit_kmeans,
    profile_clusters,
)


@pytest.fixture
def blobs():
    """Well-separated 3-cluster data so silhouette is meaningful."""
    np.random.seed(42)
    c1 = np.random.randn(40, 5) + [0,  0,  0,  0,  0]
    c2 = np.random.randn(40, 5) + [8,  8,  8,  8,  8]
    c3 = np.random.randn(40, 5) + [0,  8,  0,  8,  0]
    return np.vstack([c1, c2, c3])


@pytest.fixture
def sample_df():
    """Small household DataFrame for profiling tests."""
    np.random.seed(42)
    prods = ["FSV Credit Card", "INS Client", "TRV Globalware"]
    n     = 120
    df    = pd.DataFrame({p: np.random.binomial(1, 0.2, n) for p in prods})
    df["Income mean"]              = np.random.randint(30_000, 150_000, n).astype(float)
    df["Member Tenure Years mean"] = np.random.uniform(1, 20, n)
    df["product_count"]            = df[prods].sum(axis=1)
    df["total_ers_calls"]          = np.random.poisson(0.5, n).astype(float)
    df["is_high_income"]           = 0
    df["is_long_term_member"]      = 0
    df["has_used_ers"]             = (df["total_ers_calls"] > 0).astype(int)
    return df, prods


class TestEvaluateKRange:
    def test_returns_dataframe(self, blobs):
        result = evaluate_k_range(blobs, [2, 3, 4])
        assert isinstance(result, pd.DataFrame)

    def test_index_matches_k_range(self, blobs):
        ks = [2, 3, 4, 5]
        assert list(evaluate_k_range(blobs, ks).index) == ks

    def test_has_required_columns(self, blobs):
        result = evaluate_k_range(blobs, [2, 3])
        for col in ("inertia", "silhouette", "calinski_harabasz"):
            assert col in result.columns

    def test_inertia_decreases_with_k(self, blobs):
        result   = evaluate_k_range(blobs, [2, 3, 4, 5, 6])
        inertias = result["inertia"].tolist()
        assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))

    def test_silhouette_in_valid_range(self, blobs):
        result = evaluate_k_range(blobs, [2, 3, 4])
        assert (result["silhouette"] >= -1).all()
        assert (result["silhouette"] <=  1).all()


class TestFitKMeans:
    def test_labels_length_matches_data(self, blobs):
        _, labels = fit_kmeans(blobs, k=3)
        assert len(labels) == len(blobs)

    def test_unique_labels_equal_k(self, blobs):
        _, labels = fit_kmeans(blobs, k=3)
        assert len(np.unique(labels)) == 3

    def test_labels_are_integers(self, blobs):
        _, labels = fit_kmeans(blobs, k=3)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_known_clusters_detected(self, blobs):
        """With well-separated blobs, silhouette should be > 0.5."""
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        X      = StandardScaler().fit_transform(blobs)
        _, lab = fit_kmeans(X, k=3)
        assert silhouette_score(X, lab) > 0.5


class TestProfileClusters:
    def test_one_row_per_cluster(self, sample_df):
        df, prods = sample_df
        labels    = np.array([i % 3 for i in range(len(df))])
        profile   = profile_clusters(df, labels, prods)
        assert len(profile) == 3

    def test_contains_product_cols(self, sample_df):
        df, prods = sample_df
        labels    = np.random.randint(0, 4, len(df))
        profile   = profile_clusters(df, labels, prods)
        for p in prods:
            assert p in profile.columns

    def test_cluster_sizes_sum_to_n(self, sample_df):
        df, prods = sample_df
        labels    = np.random.randint(0, 3, len(df))
        profile   = profile_clusters(df, labels, prods)
        assert profile["cluster_size"].sum() == len(df)


class TestBuildActionTable:
    def test_returns_dataframe(self, sample_df):
        df, prods = sample_df
        labels    = np.random.randint(0, 3, len(df))
        profile   = profile_clusters(df, labels, prods)
        names     = {i: f"Seg {i}" for i in profile.index}
        table     = build_action_table(profile, names, prods)
        assert isinstance(table, pd.DataFrame)

    def test_has_required_columns(self, sample_df):
        df, prods = sample_df
        labels    = np.random.randint(0, 3, len(df))
        profile   = profile_clusters(df, labels, prods)
        names     = {i: f"Seg {i}" for i in profile.index}
        table     = build_action_table(profile, names, prods)
        for col in ("cluster", "cluster_name", "recommended_product", "avg_propensity"):
            assert col in table.columns

    def test_recommended_product_in_product_list(self, sample_df):
        df, prods = sample_df
        labels    = np.random.randint(0, 3, len(df))
        profile   = profile_clusters(df, labels, prods)
        names     = {i: f"Seg {i}" for i in profile.index}
        table     = build_action_table(profile, names, prods)
        for prod in table["recommended_product"]:
            assert prod in prods