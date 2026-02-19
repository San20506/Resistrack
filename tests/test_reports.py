"""Tests for M4.5 Stewardship Report Generator."""

import json
from typing import Any

import pytest

from resistrack.reports.generator import ReportConfig, ReportGenerator, ReportOutput


def _make_prediction(**overrides: Any) -> dict[str, Any]:
    """Create a sample prediction dict."""
    base: dict[str, Any] = {
        "patient_token": "PT_TEST001",
        "amr_risk_score": 55,
        "risk_tier": "HIGH",
        "confidence_score": 0.82,
        "low_confidence_flag": False,
        "data_completeness_score": 0.90,
        "data_quality_flag": True,
        "antibiotic_class_risk": {
            "penicillins": 0.7,
            "cephalosporins": 0.5,
            "carbapenems": 0.3,
            "fluoroquinolones": 0.4,
            "aminoglycosides": 0.2,
        },
        "recommended_action": "Review coverage",
        "model_version": "v1.0.0",
    }
    base.update(overrides)
    return base


@pytest.fixture
def generator() -> ReportGenerator:
    return ReportGenerator()


@pytest.fixture
def sample_predictions() -> list[dict[str, Any]]:
    return [
        _make_prediction(patient_token="PT_001", amr_risk_score=20, risk_tier="LOW"),
        _make_prediction(patient_token="PT_002", amr_risk_score=35, risk_tier="MEDIUM"),
        _make_prediction(patient_token="PT_003", amr_risk_score=55, risk_tier="HIGH"),
        _make_prediction(patient_token="PT_004", amr_risk_score=80, risk_tier="CRITICAL"),
        _make_prediction(patient_token="PT_005", amr_risk_score=15, risk_tier="LOW", low_confidence_flag=True),
    ]


# ── ReportConfig tests ──

class TestReportConfig:
    def test_defaults(self) -> None:
        cfg = ReportConfig()
        assert cfg.report_period_days == 7
        assert cfg.include_patient_details is False
        assert "csv" in cfg.output_formats

    def test_frozen(self) -> None:
        cfg = ReportConfig()
        with pytest.raises(AttributeError):
            cfg.report_period_days = 14  # type: ignore[misc]


# ── ReportGenerator tests ──

class TestReportGenerator:
    def test_generate_returns_output(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        assert isinstance(result, ReportOutput)
        assert result.report_id.startswith("RPT-")

    def test_summary_total_patients(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        assert result.summary["total_patients"] == 5

    def test_risk_distribution(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        dist = result.summary["risk_distribution"]
        assert dist["LOW"] == 2
        assert dist["MEDIUM"] == 1
        assert dist["HIGH"] == 1
        assert dist["CRITICAL"] == 1

    def test_avg_risk_score(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        expected = (20 + 35 + 55 + 80 + 15) / 5
        assert result.summary["avg_risk_score"] == expected

    def test_low_confidence_count(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        assert result.summary["low_confidence_count"] == 1

    def test_high_risk_rate(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        assert result.summary["high_risk_rate"] == 0.4  # 2 out of 5

    def test_antibiotic_risks_computed(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        abx = result.summary["top_antibiotic_risks"]
        assert "penicillins" in abx
        assert "carbapenems" in abx
        assert all(isinstance(v, float) for v in abx.values())

    def test_csv_content_generated(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        assert "Total Patients" in result.csv_content
        assert "Risk Tier" in result.csv_content

    def test_json_content_valid(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        parsed = json.loads(result.json_content)
        assert "report_id" in parsed
        assert "summary" in parsed

    def test_empty_predictions(self, generator: ReportGenerator) -> None:
        result = generator.generate([], "2026-02-10", "2026-02-17")
        assert result.summary["total_patients"] == 0
        assert result.summary["avg_risk_score"] == 0.0

    def test_recommendations_for_high_risk(self, generator: ReportGenerator) -> None:
        preds = [_make_prediction(risk_tier="CRITICAL", amr_risk_score=90) for _ in range(5)]
        result = generator.generate(preds, "2026-02-10", "2026-02-17")
        recs = result.summary["recommendations"]
        assert any("HIGH/CRITICAL" in r for r in recs)

    def test_recommendations_for_low_confidence(self, generator: ReportGenerator) -> None:
        preds = [_make_prediction(low_confidence_flag=True) for _ in range(5)]
        result = generator.generate(preds, "2026-02-10", "2026-02-17")
        recs = result.summary["recommendations"]
        assert any("low confidence" in r.lower() or "DATA QUALITY" in r for r in recs)

    def test_recommendations_for_normal(self, generator: ReportGenerator) -> None:
        preds = [
            _make_prediction(
                risk_tier="LOW",
                amr_risk_score=10,
                antibiotic_class_risk={
                    "penicillins": 0.1,
                    "cephalosporins": 0.1,
                    "carbapenems": 0.1,
                    "fluoroquinolones": 0.1,
                    "aminoglycosides": 0.1,
                },
            )
        ]
        result = generator.generate(preds, "2026-02-10", "2026-02-17")
        recs = result.summary["recommendations"]
        assert any("routine monitoring" in r.lower() for r in recs)

    def test_period_dates_in_output(
        self, generator: ReportGenerator, sample_predictions: list[dict[str, Any]]
    ) -> None:
        result = generator.generate(sample_predictions, "2026-02-10", "2026-02-17")
        assert result.period_start == "2026-02-10"
        assert result.period_end == "2026-02-17"
