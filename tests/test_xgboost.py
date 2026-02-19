"""Tests for M2.4 XGBoost AMR risk prediction model."""

from __future__ import annotations

import numpy as np
import pytest

from resistrack.common.constants import RANDOM_STATE
from resistrack.ml.xgboost_model import (
    DEFAULT_PARAMS,
    MIN_HPO_TRIALS,
    SMOTE_THRESHOLD,
    DataSplit,
    SMOTEHandler,
    StratifiedSplitter,
    TrainingResult,
    XGBoostAMRModel,
)


# ── Fixtures ──


def _make_dataset(
    n_samples: int = 200,
    n_features: int = 47,
    positive_rate: float = 0.3,
    n_tenants: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    n_positive = int(n_samples * positive_rate)
    y = np.zeros(n_samples, dtype=np.int32)
    y[:n_positive] = 1
    rng.shuffle(y)
    tenant_ids = np.array([f"hospital_{i % n_tenants}" for i in range(n_samples)])
    return X, y, tenant_ids


# ── StratifiedSplitter tests ──


class TestStratifiedSplitter:
    def test_split_ratios(self) -> None:
        X, y, tenants = _make_dataset(n_samples=500, n_tenants=10)
        splitter = StratifiedSplitter()
        split = splitter.split(X, y, tenants)

        total = len(split.X_train) + len(split.X_val) + len(split.X_test)
        assert total == len(X)

    def test_split_non_overlapping_tenants(self) -> None:
        X, y, tenants = _make_dataset(n_samples=500, n_tenants=10)
        splitter = StratifiedSplitter()
        split = splitter.split(X, y, tenants)

        train_set = set(split.tenant_ids_train.tolist())
        val_set = set(split.tenant_ids_val.tolist())
        assert len(train_set & val_set) == 0

    def test_split_returns_datasplit(self) -> None:
        X, y, tenants = _make_dataset()
        splitter = StratifiedSplitter()
        split = splitter.split(X, y, tenants)
        assert isinstance(split, DataSplit)

    def test_split_preserves_features(self) -> None:
        X, y, tenants = _make_dataset(n_features=47)
        splitter = StratifiedSplitter()
        split = splitter.split(X, y, tenants)
        assert split.X_train.shape[1] == 47
        assert split.X_val.shape[1] == 47
        assert split.X_test.shape[1] == 47

    def test_split_deterministic(self) -> None:
        X, y, tenants = _make_dataset()
        s1 = StratifiedSplitter().split(X, y, tenants)
        s2 = StratifiedSplitter().split(X, y, tenants)
        np.testing.assert_array_equal(s1.X_train, s2.X_train)


# ── SMOTEHandler tests ──


class TestSMOTEHandler:
    def test_needs_smote_below_threshold(self) -> None:
        handler = SMOTEHandler()
        y = np.zeros(100, dtype=np.int32)
        y[:10] = 1  # 10% positive
        assert handler.needs_smote(y) is True

    def test_needs_smote_above_threshold(self) -> None:
        handler = SMOTEHandler()
        y = np.zeros(100, dtype=np.int32)
        y[:30] = 1  # 30% positive
        assert handler.needs_smote(y) is False

    def test_needs_smote_at_threshold(self) -> None:
        handler = SMOTEHandler()
        y = np.zeros(100, dtype=np.int32)
        y[:20] = 1  # 20% = threshold
        assert handler.needs_smote(y) is False

    def test_needs_smote_empty(self) -> None:
        handler = SMOTEHandler()
        assert handler.needs_smote(np.array([])) is False

    def test_apply_increases_positive(self) -> None:
        handler = SMOTEHandler()
        rng = np.random.RandomState(RANDOM_STATE)
        X = rng.randn(100, 10).astype(np.float32)
        y = np.zeros(100, dtype=np.int32)
        y[:5] = 1  # 5% positive

        X_new, y_new = handler.apply(X, y)
        assert len(y_new) > len(y)
        assert float(np.mean(y_new == 1)) >= SMOTE_THRESHOLD * 0.8

    def test_apply_no_change_above_threshold(self) -> None:
        handler = SMOTEHandler()
        rng = np.random.RandomState(RANDOM_STATE)
        X = rng.randn(100, 10).astype(np.float32)
        y = np.zeros(100, dtype=np.int32)
        y[:30] = 1  # 30% positive

        X_new, y_new = handler.apply(X, y)
        assert len(y_new) == len(y)

    def test_apply_deterministic(self) -> None:
        handler = SMOTEHandler()
        rng = np.random.RandomState(RANDOM_STATE)
        X = rng.randn(100, 10).astype(np.float32)
        y = np.zeros(100, dtype=np.int32)
        y[:5] = 1

        X1, y1 = handler.apply(X.copy(), y.copy())
        handler2 = SMOTEHandler()
        X2, y2 = handler2.apply(X.copy(), y.copy())
        np.testing.assert_array_equal(X1, X2)


# ── XGBoostAMRModel tests ──


class TestXGBoostAMRModel:
    def test_default_params(self) -> None:
        model = XGBoostAMRModel()
        assert model.params["random_state"] == RANDOM_STATE
        assert model.params["objective"] == "binary:logistic"

    def test_custom_params(self) -> None:
        model = XGBoostAMRModel(params={"max_depth": 4})
        assert model.params["max_depth"] == 4
        assert model.params["random_state"] == RANDOM_STATE

    def test_not_fitted_initially(self) -> None:
        model = XGBoostAMRModel()
        assert model.is_fitted is False
        assert model.feature_importances is None

    def test_fit(self) -> None:
        model = XGBoostAMRModel()
        X, y, tenants = _make_dataset()
        result = model.fit(X, y, tenants)

        assert isinstance(result, TrainingResult)
        assert model.is_fitted is True
        assert result.n_trials >= MIN_HPO_TRIALS
        assert result.train_auc_roc > 0
        assert result.val_auc_roc > 0
        assert result.val_auprc > 0

    def test_fit_with_smote(self) -> None:
        model = XGBoostAMRModel()
        X, y, tenants = _make_dataset(positive_rate=0.05)
        result = model.fit(X, y, tenants)
        assert result.smote_applied is True

    def test_fit_without_smote(self) -> None:
        model = XGBoostAMRModel()
        X, y, tenants = _make_dataset(positive_rate=0.4)
        result = model.fit(X, y, tenants)
        assert result.smote_applied is False

    def test_feature_importances(self) -> None:
        model = XGBoostAMRModel()
        X, y, tenants = _make_dataset(n_features=47)
        model.fit(X, y, tenants)

        importances = model.feature_importances
        assert importances is not None
        assert importances.shape == (47,)
        assert abs(float(np.sum(importances)) - 1.0) < 1e-5

    def test_predict_proba(self) -> None:
        model = XGBoostAMRModel()
        X, y, tenants = _make_dataset()
        model.fit(X, y, tenants)

        probas = model.predict_proba(X[:10])
        assert probas.shape == (10,)
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

    def test_predict_risk_score(self) -> None:
        model = XGBoostAMRModel()
        X, y, tenants = _make_dataset()
        model.fit(X, y, tenants)

        scores = model.predict_risk_score(X[:10])
        assert scores.shape == (10,)
        assert np.all(scores >= 0)
        assert np.all(scores <= 100)

    def test_predict_before_fit_raises(self) -> None:
        model = XGBoostAMRModel()
        X, _, _ = _make_dataset()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(X)

    def test_predict_wrong_features_raises(self) -> None:
        model = XGBoostAMRModel()
        X, y, tenants = _make_dataset(n_features=47)
        model.fit(X, y, tenants)

        X_wrong = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(ValueError, match="features"):
            model.predict_proba(X_wrong)
