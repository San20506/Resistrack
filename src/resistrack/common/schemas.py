"""Pydantic v2 schemas for ResisTrack AMR prediction output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from resistrack.common.constants import CONFIDENCE_THRESHOLD, RiskTier


class SHAPFeature(BaseModel):
    """A single SHAP feature explanation."""

    name: str
    value: float
    direction: str = Field(pattern=r"^(positive|negative)$")
    human_readable: str


class AntibioticClassRisk(BaseModel):
    """Risk scores per antibiotic class."""

    penicillins: float = Field(ge=0.0, le=1.0)
    cephalosporins: float = Field(ge=0.0, le=1.0)
    carbapenems: float = Field(ge=0.0, le=1.0)
    fluoroquinolones: float = Field(ge=0.0, le=1.0)
    aminoglycosides: float = Field(ge=0.0, le=1.0)


class AMRPredictionOutput(BaseModel):
    """Complete AMR risk prediction output schema."""

    patient_token: str
    amr_risk_score: int = Field(ge=0, le=100)
    risk_tier: RiskTier
    confidence_score: float = Field(ge=0.0, le=1.0)
    low_confidence_flag: bool = False
    data_completeness_score: float = Field(ge=0.0, le=1.0)
    data_quality_flag: bool
    antibiotic_class_risk: AntibioticClassRisk
    shap_top_features: list[SHAPFeature]
    recommended_action: str
    model_version: str

    @model_validator(mode="before")
    @classmethod
    def set_low_confidence_flag(cls, data: Any) -> Any:
        """Auto-set low_confidence_flag when confidence_score < threshold."""
        if isinstance(data, dict):
            confidence = data.get("confidence_score")
            if confidence is not None:
                data["low_confidence_flag"] = float(confidence) < CONFIDENCE_THRESHOLD
        return data


__all__ = ["SHAPFeature", "AntibioticClassRisk", "AMRPredictionOutput"]
