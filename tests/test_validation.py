"""Tests for M5.1 Clinical Validation."""
from __future__ import annotations

import numpy as np
import pytest

from resistrack.common.constants import RANDOM_STATE, RiskTier
from resistrack.validation.validator import (
    AUC_GAP_THRESHOLD,
    METRIC_THRESHOLDS,
    ClinicalValidator,
    SubgroupAnalysis,
    ValidationConfig,
    ValidationMetrics,
    ValidationReport,
    _compute_auc_roc,
    _compute_auprc,
    _compute_brier,
    _compute_fpr_at_threshold,
    _sensitivity_at_specificity,
)
from resistrack.validation.model_card import ModelCard, ModelCardGenerator


def _make_validation_data(
    n: int = 1200,
    auc_target: float = 0.85,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, 2, n).astype(np.float64)
    noise = rng.normal(0, 0.15, n)
    scores = np.clip(labels * auc_target + (1 - labels) * (1 - auc_target) + noise, 0, 1)
    return labels, scores


class TestComputeAucRoc:
    def test_perfect_separation(self) -> None:
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert _compute_auc_roc(labels, scores) == pytest.approx(1.0)

    def test_random_around_half(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        labels = rng.randint(0, 2, 1000).astype(np.float64)
        scores = rng.uniform(0, 1, 1000)
        assert 0.4 <= _compute_auc_roc(labels, scores) <= 0.6

    def test_empty_returns_half(self) -> None:
        assert _compute_auc_roc(np.array([]), np.array([])) == 0.5

    def test_all_positive_returns_half(self) -> None:
        labels = np.ones(10)
        scores = np.linspace(0, 1, 10)
        assert _compute_auc_roc(labels, scores) == 0.5


class TestComputeAuprc:
    def test_perfect_separation(self) -> None:
        labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert _compute_auprc(labels, scores) == pytest.approx(1.0)

    def test_empty_returns_zero(self) -> None:
        assert _compute_auprc(np.array([]), np.array([])) == 0.0

    def test_no_positives_returns_zero(self) -> None:
        assert _compute_auprc(np.zeros(10), np.ones(10)) == 0.0


class TestSensitivityAtSpecificity:
    def test_good_model(self) -> None:
        labels, scores = _make_validation_data(1000, auc_target=0.90)
        sens = _sensitivity_at_specificity(labels, scores, 0.80)
        assert sens > 0.5

    def test_empty_returns_zero(self) -> None:
        assert _sensitivity_at_specificity(np.array([]), np.array([]), 0.80) == 0.0


class TestFprAndBrier:
    def test_fpr_perfect(self) -> None:
        labels = np.array([1, 1, 0, 0], dtype=np.float64)
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert _compute_fpr_at_threshold(labels, scores, 0.5) == 0.0

    def test_brier_perfect(self) -> None:
        labels = np.array([1, 0, 1, 0], dtype=np.float64)
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        assert _compute_brier(labels, probs) == pytest.approx(0.0)

    def test_brier_worst(self) -> None:
        labels = np.array([1, 0, 1, 0], dtype=np.float64)
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        assert _compute_brier(labels, probs) == pytest.approx(1.0)


class TestValidationMetrics:
    def test_all_pass(self) -> None:
        m = ValidationMetrics(
            auc_roc=0.85, auprc=0.75, sensitivity_at_80_spec=0.82,
            fpr=0.15, brier=0.10, p95_latency_ms=1500.0,
        )
        assert m.all_pass is True
        assert all(m.passes_thresholds().values())

    def test_auc_fail(self) -> None:
        m = ValidationMetrics(
            auc_roc=0.75, auprc=0.75, sensitivity_at_80_spec=0.82,
            fpr=0.15, brier=0.10, p95_latency_ms=1500.0,
        )
        assert m.all_pass is False
        assert m.passes_thresholds()["auc_roc"] is False

    def test_brier_fail(self) -> None:
        m = ValidationMetrics(
            auc_roc=0.85, auprc=0.75, sensitivity_at_80_spec=0.82,
            fpr=0.15, brier=0.25, p95_latency_ms=1500.0,
        )
        assert m.passes_thresholds()["brier"] is False


class TestSubgroupAnalysis:
    def test_no_flagged(self) -> None:
        from resistrack.validation.validator import SubgroupResult
        sg = SubgroupAnalysis(subgroups=[
            SubgroupResult("age_adult", 500, 0.84, 0.01, False),
        ])
        assert sg.any_flagged is False

    def test_flagged_subgroup(self) -> None:
        from resistrack.validation.validator import SubgroupResult
        sg = SubgroupAnalysis(subgroups=[
            SubgroupResult("age_pediatric", 50, 0.70, 0.15, True),
        ])
        assert sg.any_flagged is True
        assert len(sg.flagged_subgroups) == 1


class TestClinicalValidator:
    def test_too_few_records_raises(self) -> None:
        cv = ClinicalValidator()
        labels = np.zeros(100)
        scores = np.zeros(100)
        with pytest.raises(ValueError, match="Need >= 1000"):
            cv.validate(labels, scores)

    def test_mismatched_lengths_raises(self) -> None:
        cv = ClinicalValidator(ValidationConfig(min_records=5))
        with pytest.raises(ValueError, match="same length"):
            cv.validate(np.zeros(10), np.zeros(5))

    def test_full_validation(self) -> None:
        labels, scores = _make_validation_data(1200, auc_target=0.85)
        cv = ClinicalValidator()
        report = cv.validate(labels, scores, model_version="1.0.0")
        assert isinstance(report, ValidationReport)
        assert report.n_records == 1200
        assert report.model_version == "1.0.0"
        assert 0 <= report.metrics.auc_roc <= 1
        assert 0 <= report.metrics.brier <= 1

    def test_with_subgroups(self) -> None:
        rng = np.random.RandomState(RANDOM_STATE)
        labels, scores = _make_validation_data(1200)
        ages = rng.choice([10, 30, 50, 70, 80], 1200)
        icu_flags = rng.randint(0, 2, 1200)
        cv = ClinicalValidator()
        report = cv.validate(labels, scores, ages=ages, icu_flags=icu_flags)
        assert len(report.subgroup_analysis.subgroups) > 0

    def test_with_latencies(self) -> None:
        labels, scores = _make_validation_data(1200)
        latencies = np.random.RandomState(RANDOM_STATE).uniform(50, 500, 1200)
        cv = ClinicalValidator()
        report = cv.validate(labels, scores, latencies_ms=latencies)
        assert report.metrics.p95_latency_ms > 0

    def test_bootstrap_ci(self) -> None:
        labels, scores = _make_validation_data(1200)
        cv = ClinicalValidator()
        report = cv.validate(labels, scores)
        assert report.ci_lower <= report.ci_upper
        assert report.ci_lower > 0
        assert report.ci_upper <= 1

    def test_custom_config(self) -> None:
        cfg = ValidationConfig(min_records=50, n_bootstrap=10)
        labels, scores = _make_validation_data(100)
        cv = ClinicalValidator(cfg)
        report = cv.validate(labels, scores)
        assert report.n_records == 100


class TestModelCardGenerator:
    def _make_report(self) -> ValidationReport:
        labels, scores = _make_validation_data(1200)
        cv = ClinicalValidator()
        return cv.validate(labels, scores, model_version="2.0.0")

    def test_generate_card(self) -> None:
        report = self._make_report()
        gen = ModelCardGenerator()
        card = gen.generate(report)
        assert isinstance(card, ModelCard)
        assert card.model_version == "2.0.0"
        assert card.model_name == "ResisTrack AMR Ensemble"
        assert "n_records" in card.validation_summary
        assert len(card.metrics) == 6

    def test_to_markdown(self) -> None:
        report = self._make_report()
        gen = ModelCardGenerator()
        card = gen.generate(report)
        md = gen.to_markdown(card)
        assert "# Model Card" in md
        assert "auc_roc" in md
        assert "Performance Metrics" in md

    def test_custom_model_name(self) -> None:
        report = self._make_report()
        gen = ModelCardGenerator(model_name="Custom Model")
        card = gen.generate(report)
        assert card.model_name == "Custom Model"

    def test_limitations_present(self) -> None:
        report = self._make_report()
        gen = ModelCardGenerator()
        card = gen.generate(report)
        assert len(card.limitations) > 0
        assert len(card.ethical_considerations) > 0
