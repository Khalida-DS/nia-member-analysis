"""
tests/test_metrics.py
=====================
Unit tests for src/evaluation/metrics.py.
Run: pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest

from src.evaluation.metrics import (
    classification_scorecard,
    lift_at_k,
    regression_scorecard,
)


class TestLiftAtK:
    def test_perfect_model_gives_high_lift(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.85, 0.8, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])
        assert lift_at_k(y_true, y_prob, k=0.30) > 1.0

    def test_all_negatives_returns_zero(self):
        y_true = np.zeros(10)
        y_prob = np.random.rand(10)
        assert lift_at_k(y_true, y_prob) == 0.0

    def test_k_one_gives_lift_of_one(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 200)
        y_prob = np.random.rand(200)
        assert abs(lift_at_k(y_true, y_prob, k=1.0) - 1.0) < 1e-9

    def test_random_scores_near_one(self):
        np.random.seed(0)
        y_true = np.random.binomial(1, 0.2, 1000)
        y_prob = np.random.rand(1000)
        assert 0.5 < lift_at_k(y_true, y_prob, k=0.10) < 2.0


class TestClassificationScorecard:
    @pytest.fixture
    def sample(self):
        np.random.seed(42)
        n      = 300
        y_true = np.random.binomial(1, 0.25, n)
        y_prob = np.clip(y_true * 0.6 + np.random.rand(n) * 0.4, 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
        return y_true, y_pred, y_prob

    def test_returns_expected_keys(self, sample):
        y_true, y_pred, y_prob = sample
        s = classification_scorecard(y_true, y_pred, y_prob, "Test")
        for key in ("model", "roc_auc", "f1", "precision", "recall", "lift_at_10"):
            assert key in s

    def test_auc_between_0_and_1(self, sample):
        y_true, y_pred, y_prob = sample
        s = classification_scorecard(y_true, y_pred, y_prob, "Test")
        assert 0.0 <= s["roc_auc"] <= 1.0

    def test_perfect_classifier_auc_is_one(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        y_pred = (y_prob >= 0.5).astype(int)
        s = classification_scorecard(y_true, y_pred, y_prob, "Perfect")
        assert s["roc_auc"] == 1.0

    def test_label_stored(self, sample):
        y_true, y_pred, y_prob = sample
        s = classification_scorecard(y_true, y_pred, y_prob, "RF", label="INS Client")
        assert s["label"] == "INS Client"


class TestRegressionScorecard:
    def test_returns_expected_keys(self):
        y = np.array([10.0, 20.0, 30.0])
        s = regression_scorecard(y, y * 1.05, "Ridge")
        for key in ("model", "rmse", "mae", "r2"):
            assert key in s

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        s = regression_scorecard(y, y, "Perfect")
        assert s["rmse"] == 0.0
        assert abs(s["r2"] - 1.0) < 1e-9

    def test_rmse_nonnegative(self):
        y_true = np.random.rand(100) * 100
        y_pred = y_true + np.random.randn(100) * 5
        s = regression_scorecard(y_true, y_pred, "RF")
        assert s["rmse"] >= 0