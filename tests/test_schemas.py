"""Tests for ResisTrack Pydantic schemas."""

import pytest
from pydantic import ValidationError

from resistrack.common.constants import RiskTier
from resistrack.common.schemas import (
    AMRPredictionOutput,
    AntibioticClassRisk,
    SHAPFeature,
)


def _make_prediction(**overrides: object) -> dict:
    """Create a valid prediction dict with optional overrides."""
    base: dict = {
        "patient_token": "PT_ABC123",
        "amr_risk_score": 72,
        "risk_tier": RiskTier.HIGH,
        "confidence_score": 0.85,
        "data_completeness_score": 0.95,
        "data_quality_flag": True,
        "antibiotic_class_risk": {
            "penicillins": 0.8,
            "cephalosporins": 0.6,
            "carbapenems": 0.3,
            "fluoroquinolones": 0.5,
            "aminoglycosides": 0.2,
        },
        "shap_top_features": [
            {
                "name": "prior_resistance",
                "value": 0.42,
                "direction": "positive",
                "human_readable": "Previous resistant culture increases risk",
            }
        ],
        "recommended_action": "Consider broadening empiric coverage",
        "model_version": "v1.0.0",
    }
    base.update(overrides)
    return base


class TestSHAPFeature:
    def test_valid_creation(self) -> None:
        feat = SHAPFeature(
            name="prior_resistance",
            value=0.42,
            direction="positive",
            human_readable="Increases risk",
        )
        assert feat.name == "prior_resistance"
        assert feat.direction == "positive"

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValidationError):
            SHAPFeature(
                name="test", value=0.1, direction="invalid", human_readable="x"
            )


class TestAntibioticClassRisk:
    def test_valid_creation(self) -> None:
        risk = AntibioticClassRisk(
            penicillins=0.8,
            cephalosporins=0.6,
            carbapenems=0.3,
            fluoroquinolones=0.5,
            aminoglycosides=0.2,
        )
        assert risk.penicillins == 0.8

    def test_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            AntibioticClassRisk(
                penicillins=1.5,
                cephalosporins=0.6,
                carbapenems=0.3,
                fluoroquinolones=0.5,
                aminoglycosides=0.2,
            )


class TestAMRPredictionOutput:
    def test_valid_creation(self) -> None:
        pred = AMRPredictionOutput(**_make_prediction())
        assert pred.amr_risk_score == 72
        assert pred.risk_tier == RiskTier.HIGH

    def test_low_confidence_flag_auto_set_true(self) -> None:
        pred = AMRPredictionOutput(**_make_prediction(confidence_score=0.45))
        assert pred.low_confidence_flag is True

    def test_low_confidence_flag_auto_set_false(self) -> None:
        pred = AMRPredictionOutput(**_make_prediction(confidence_score=0.85))
        assert pred.low_confidence_flag is False

    def test_confidence_at_threshold(self) -> None:
        pred = AMRPredictionOutput(**_make_prediction(confidence_score=0.60))
        assert pred.low_confidence_flag is False

    def test_confidence_just_below_threshold(self) -> None:
        pred = AMRPredictionOutput(**_make_prediction(confidence_score=0.59))
        assert pred.low_confidence_flag is True

    def test_risk_score_too_high(self) -> None:
        with pytest.raises(ValidationError):
            AMRPredictionOutput(**_make_prediction(amr_risk_score=101))

    def test_risk_score_too_low(self) -> None:
        with pytest.raises(ValidationError):
            AMRPredictionOutput(**_make_prediction(amr_risk_score=-1))

    def test_risk_score_boundary_zero(self) -> None:
        pred = AMRPredictionOutput(**_make_prediction(amr_risk_score=0))
        assert pred.amr_risk_score == 0

    def test_risk_score_boundary_hundred(self) -> None:
        pred = AMRPredictionOutput(**_make_prediction(amr_risk_score=100))
        assert pred.amr_risk_score == 100

    def test_confidence_score_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            AMRPredictionOutput(**_make_prediction(confidence_score=1.5))
