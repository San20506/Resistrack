"""Tests for M2.6 Ensemble Calibration & SHAP Pipeline."""
from __future__ import annotations

import time

import numpy as np
import pytest

from resistrack.common.constants import (
    ANTIBIOTIC_CLASSES,
    CONFIDENCE_THRESHOLD,
    RANDOM_STATE,
    RISK_TIER_RANGES,
    RiskTier,
)
from resistrack.common.schemas import AMRPredictionOutput
from resistrack.ml.ensemble import (
    AntibioticRiskEstimator,
    CalibrationResult,
    EnsemblePredictor,
    EnsemblePrediction,
    EnsembleTrainingResult,
    FeatureAttributor,
    MetaLearner,
    MetaLearnerState,
    PlattCalibrator,
    SubModelOutput,
    _compute_auc,
    _score_to_risk_tier,
    _sigmoid,
)


def _make_sub_model_outputs(
    n_samples: int, rng: np.random.RandomState | None = None,
) -> list[SubModelOutput]:
    """Create 3 realistic sub-model outputs for testing."""
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)
    return [
        SubModelOutput(
            model_name="xgboost",
            scores=rng.uniform(0.1, 0.9, n_samples),
        ),
        SubModelOutput(
            model_name="lstm",
            scores=rng.uniform(0.2, 0.8, n_samples),
        ),
        SubModelOutput(
            model_name="clinicalbert",
            scores=rng.uniform(0.15, 0.85, n_samples),
        ),
    ]


def _make_score_matrix(outputs: list[SubModelOutput]) -> np.ndarray:
    """Stack SubModelOutput list into (n_samples, n_models) ndarray."""
    return np.column_stack([o.scores for o in outputs])


def _make_training_data(
    n_samples: int = 200,
) -> tuple[list[SubModelOutput], np.ndarray]:
    """Create sub-model outputs + labels for training."""
    rng = np.random.RandomState(RANDOM_STATE)
    outputs = _make_sub_model_outputs(n_samples, rng)
    avg_scores = np.mean([o.scores for o in outputs], axis=0)
    labels = (avg_scores + rng.normal(0, 0.15, n_samples) > 0.5).astype(np.float64)
    return outputs, labels


class TestSigmoid:
    """Tests for _sigmoid helper."""

    def test_midpoint(self) -> None:
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self) -> None:
        assert _sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self) -> None:
        assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self) -> None:
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)

    def test_array_input(self) -> None:
        result = _sigmoid(np.array([0.0, 1.0, -1.0]))
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.5)


class TestScoreToRiskTier:
    """Tests for _score_to_risk_tier."""

    def test_low_tier(self) -> None:
        assert _score_to_risk_tier(10) == RiskTier.LOW
        assert _score_to_risk_tier(0) == RiskTier.LOW
        assert _score_to_risk_tier(24) == RiskTier.LOW

    def test_medium_tier(self) -> None:
        assert _score_to_risk_tier(25) == RiskTier.MEDIUM
        assert _score_to_risk_tier(49) == RiskTier.MEDIUM

    def test_high_tier(self) -> None:
        assert _score_to_risk_tier(50) == RiskTier.HIGH
        assert _score_to_risk_tier(74) == RiskTier.HIGH

    def test_critical_tier(self) -> None:
        assert _score_to_risk_tier(75) == RiskTier.CRITICAL
        assert _score_to_risk_tier(100) == RiskTier.CRITICAL

    def test_clamp_below_zero(self) -> None:
        assert _score_to_risk_tier(-5) == RiskTier.LOW

    def test_clamp_above_100(self) -> None:
        assert _score_to_risk_tier(150) == RiskTier.CRITICAL


class TestComputeAUC:
    """Tests for _compute_auc (trapezoidal AUC-ROC)."""

    def test_perfect_separation(self) -> None:
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], dtype=np.float64)
        auc = _compute_auc(labels, scores)
        assert auc == pytest.approx(1.0)

    def test_random_auc(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        labels = rng.randint(0, 2, 1000).astype(np.float64)
        scores = rng.uniform(0, 1, 1000)
        auc = _compute_auc(labels, scores)
        assert 0.4 <= auc <= 0.6

    def test_inverse_separation(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], dtype=np.float64)
        auc = _compute_auc(labels, scores)
        assert auc == pytest.approx(0.0)


class TestMetaLearner:
    """Tests for MetaLearner (RULE-TRAIN-06: learned weights)."""

    def test_initialization(self) -> None:
        ml = MetaLearner()
        assert ml.is_fitted is False

    def test_fit_produces_weights(self) -> None:
        outputs, labels = _make_training_data()
        ml = MetaLearner()
        scores = _make_score_matrix(outputs)
        ml.fit(scores, labels)
        assert ml.is_fitted is True
        assert ml.weights is not None
        assert len(ml.weights) == 3

    def test_weights_are_learned_not_equal(self) -> None:
        """RULE-TRAIN-06: weights must be learned, not hardcoded equal."""
        outputs, labels = _make_training_data()
        ml = MetaLearner()
        scores = _make_score_matrix(outputs)
        ml.fit(scores, labels)
        weights = ml.weights
        assert weights is not None
        assert not np.allclose(weights, weights[0])

    def test_predict_raw_unfitted_raises(self) -> None:
        ml = MetaLearner()
        with pytest.raises(RuntimeError, match="not fitted"):
            ml.predict_raw(np.array([[0.5, 0.5, 0.5]]))

    def test_predict_raw_produces_scores(self) -> None:
        outputs, labels = _make_training_data()
        ml = MetaLearner()
        scores = _make_score_matrix(outputs)
        ml.fit(scores, labels)
        raw = ml.predict_raw(scores)
        assert raw.shape == (200,)

    def test_get_state_load_state(self) -> None:
        outputs, labels = _make_training_data()
        ml = MetaLearner()
        scores = _make_score_matrix(outputs)
        ml.fit(scores, labels)
        state = ml.get_state()
        assert isinstance(state, MetaLearnerState)

        ml2 = MetaLearner()
        ml2.load_state(state)
        assert ml2.is_fitted is True
        raw1 = ml.predict_raw(scores)
        raw2 = ml2.predict_raw(scores)
        np.testing.assert_array_almost_equal(raw1, raw2)

    def test_deterministic(self) -> None:
        outputs, labels = _make_training_data()
        scores = _make_score_matrix(outputs)
        ml1 = MetaLearner()
        ml1.fit(scores, labels)
        ml2 = MetaLearner()
        ml2.fit(scores, labels)
        np.testing.assert_array_almost_equal(
            ml1.weights, ml2.weights,  # type: ignore[arg-type]
        )


class TestPlattCalibrator:
    """Tests for PlattCalibrator (Brier <= 0.15 target)."""

    def test_fit_produces_calibration(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        scores = rng.uniform(0, 1, 200)
        labels = (scores + rng.normal(0, 0.2, 200) > 0.5).astype(np.float64)
        cal = PlattCalibrator()
        result = cal.fit(scores, labels)
        assert isinstance(result, CalibrationResult)
        assert result.n_samples == 200

    def test_calibrate_range(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        scores = rng.uniform(0, 1, 200)
        labels = (scores + rng.normal(0, 0.1, 200) > 0.5).astype(np.float64)
        cal = PlattCalibrator()
        cal.fit(scores, labels)
        calibrated = cal.calibrate(scores)
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_unfitted_raises(self) -> None:
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.calibrate(np.array([0.5]))


class TestFeatureAttributor:
    """Tests for FeatureAttributor (SHAP top-5 features)."""

    def test_attribute_returns_top_k(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        features = rng.uniform(0, 1, 20)
        meta_weights = rng.uniform(-1, 1, 3)
        sub_imps = [rng.uniform(0, 1, 20) for _ in range(3)]
        fa = FeatureAttributor(top_k=5)
        attrs = fa.attribute(features, meta_weights, sub_imps)
        assert len(attrs) == 5

    def test_attribute_has_correct_fields(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        features = rng.uniform(0, 1, 20)
        meta_weights = rng.uniform(-1, 1, 3)
        sub_imps = [rng.uniform(0, 1, 20) for _ in range(3)]
        fa = FeatureAttributor(top_k=3)
        attrs = fa.attribute(features, meta_weights, sub_imps)
        for attr in attrs:
            assert attr.name != ""
            assert isinstance(attr.value, float)
            assert attr.direction in ("positive", "negative")
            assert attr.human_readable != ""

    def test_default_top_5(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        features = rng.uniform(0, 1, 20)
        meta_weights = rng.uniform(-1, 1, 3)
        sub_imps = [rng.uniform(0, 1, 20) for _ in range(3)]
        fa = FeatureAttributor()
        attrs = fa.attribute(features, meta_weights, sub_imps)
        assert len(attrs) == 5


class TestAntibioticRiskEstimator:
    """Tests for AntibioticRiskEstimator (5 antibiotic classes)."""

    def test_returns_all_classes(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        features = rng.uniform(0, 1, 20)
        est = AntibioticRiskEstimator()
        risk = est.estimate(60.0, features)
        assert risk.penicillins >= 0
        assert risk.cephalosporins >= 0
        assert risk.carbapenems >= 0
        assert risk.fluoroquinolones >= 0
        assert risk.aminoglycosides >= 0

    def test_risks_in_range(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        features = rng.uniform(0, 1, 20)
        est = AntibioticRiskEstimator()
        risk = est.estimate(50.0, features)
        for val in [
            risk.penicillins, risk.cephalosporins, risk.carbapenems,
            risk.fluoroquinolones, risk.aminoglycosides,
        ]:
            assert 0.0 <= val <= 1.0

    def test_deterministic(self) -> None:
        features = np.ones(20) * 0.5
        est = AntibioticRiskEstimator()
        r1 = est.estimate(70.0, features)
        r2 = est.estimate(70.0, features)
        assert r1.penicillins == r2.penicillins


class TestEnsemblePredictor:
    """Integration tests for EnsemblePredictor."""

    def test_construction(self) -> None:
        ep = EnsemblePredictor(model_version="1.0.0")
        assert ep.is_fitted is False

    def test_train(self) -> None:
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        result = ep.train(outputs, labels)
        assert isinstance(result, EnsembleTrainingResult)
        assert ep.is_fitted is True
        assert len(result.meta_weights) == 3
        assert result.n_train > 0
        assert result.n_val > 0

    def test_predict_batch(self) -> None:
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        ep.train(outputs, labels)

        test_outputs = _make_sub_model_outputs(10)
        scores = _make_score_matrix(test_outputs)
        features = np.random.RandomState(RANDOM_STATE).uniform(0, 1, (10, 20))
        tokens = [f"tok_{i}" for i in range(10)]
        preds = ep.predict(scores, features, tokens)
        assert len(preds) == 10
        for p in preds:
            assert isinstance(p, EnsemblePrediction)
            assert 0 <= p.risk_score <= 100
            assert p.risk_tier in list(RiskTier)
            assert 0 <= p.confidence <= 1
            assert isinstance(p.low_confidence_flag, bool)
            assert len(p.shap_features) == 5
            assert p.antibiotic_class_risk is not None

    def test_low_confidence_flag(self) -> None:
        """RULE-SAFETY-02: flag if confidence < 0.60."""
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        ep.train(outputs, labels)

        test_outputs = _make_sub_model_outputs(50)
        scores = _make_score_matrix(test_outputs)
        features = np.random.RandomState(RANDOM_STATE).uniform(0, 1, (50, 20))
        tokens = [f"tok_{i}" for i in range(50)]
        preds = ep.predict(scores, features, tokens)
        for p in preds:
            if p.confidence < CONFIDENCE_THRESHOLD:
                assert p.low_confidence_flag is True
            else:
                assert p.low_confidence_flag is False

    def test_predict_unfitted_raises(self) -> None:
        ep = EnsemblePredictor(model_version="1.0.0")
        with pytest.raises(RuntimeError):
            ep.predict(
                np.zeros((5, 3)),
                np.zeros((5, 20)),
                [f"tok_{i}" for i in range(5)],
            )

    def test_predict_single(self) -> None:
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        ep.train(outputs, labels)

        rng = np.random.RandomState(RANDOM_STATE)
        single_scores = rng.uniform(0.3, 0.7, 3)
        features = rng.uniform(0, 1, 20)
        result = ep.predict_single(single_scores, features, "tok_abc123")
        assert isinstance(result, AMRPredictionOutput)
        assert result.patient_token == "tok_abc123"
        assert 0 <= result.amr_risk_score <= 100
        assert result.risk_tier in [t.value for t in RiskTier]
        assert result.model_version == "1.0.0"

    def test_predict_single_low_confidence_flag(self) -> None:
        """Verify predict_single sets low_confidence_flag correctly."""
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        ep.train(outputs, labels)

        rng = np.random.RandomState(99)
        single_scores = rng.uniform(0.3, 0.7, 3)
        features = rng.uniform(0, 1, 20)
        result = ep.predict_single(single_scores, features, "tok_test")
        if result.confidence_score < CONFIDENCE_THRESHOLD:
            assert result.low_confidence_flag is True

    def test_risk_tier_ranges(self) -> None:
        """Verify risk score -> tier mapping matches constants."""
        for tier, (low, high) in RISK_TIER_RANGES.items():
            for score in [low, (low + high) // 2, high]:
                result = _score_to_risk_tier(score)
                assert result == tier, f"Score {score} -> {result}, expected {tier}"

    def test_serialization_roundtrip(self) -> None:
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        ep.train(outputs, labels)
        state = ep.get_state()

        ep2 = EnsemblePredictor(model_version="1.0.0")
        ep2.load_state(state)
        assert ep2.is_fitted is True

        test_outputs = _make_sub_model_outputs(5)
        scores = _make_score_matrix(test_outputs)
        features = np.random.RandomState(RANDOM_STATE).uniform(0, 1, (5, 20))
        tokens = [f"tok_{i}" for i in range(5)]
        preds1 = ep.predict(scores, features, tokens)
        preds2 = ep2.predict(scores, features, tokens)
        for p1, p2 in zip(preds1, preds2):
            assert p1.risk_score == pytest.approx(p2.risk_score)
            assert p1.risk_tier == p2.risk_tier

    def test_val_auc_reasonable(self) -> None:
        """Validation AUC should be reasonable (>0.5 for correlated data)."""
        outputs, labels = _make_training_data(n_samples=500)
        ep = EnsemblePredictor(model_version="1.0.0")
        result = ep.train(outputs, labels)
        assert result.val_auc_roc > 0.5

    def test_latency_tracking(self) -> None:
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        ep.train(outputs, labels)

        test_outputs = _make_sub_model_outputs(1)
        scores = _make_score_matrix(test_outputs)
        features = np.random.RandomState(RANDOM_STATE).uniform(0, 1, (1, 20))
        preds = ep.predict(scores, features, ["tok_0"])
        assert preds[0].latency_ms >= 0

    def test_shap_features_have_human_readable(self) -> None:
        """SHAP features must have human-readable clinical labels."""
        outputs, labels = _make_training_data()
        ep = EnsemblePredictor(model_version="1.0.0")
        ep.train(outputs, labels)

        test_outputs = _make_sub_model_outputs(1)
        scores = _make_score_matrix(test_outputs)
        features = np.random.RandomState(RANDOM_STATE).uniform(0, 1, (1, 20))
        preds = ep.predict(scores, features, ["tok_0"])
        for sf in preds[0].shap_features:
            assert sf.human_readable != ""
            assert sf.direction in ("positive", "negative")
