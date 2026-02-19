"""Data quality assessment for feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

from resistrack.features.extractor import (
    ALL_FEATURE_NAMES,
    EXPECTED_FEATURE_COUNT,
    ExtractedFeatures,
)


@dataclass
class QualityReport:
    """Data quality assessment report."""

    completeness_score: float
    missing_features: list[str]
    zero_value_features: list[str]
    data_quality_flag: bool
    feature_count: int
    expected_count: int

    @property
    def is_acceptable(self) -> bool:
        """Check if data quality meets minimum threshold (>=0.6)."""
        return self.completeness_score >= 0.6


class DataQualityChecker:
    """Assess data quality and completeness of extracted features."""

    def __init__(self, min_completeness: float = 0.6) -> None:
        self.min_completeness = min_completeness

    def assess(self, features: ExtractedFeatures) -> QualityReport:
        """Assess quality of extracted features."""
        flat = features.to_flat_dict()

        missing: list[str] = [
            name for name in ALL_FEATURE_NAMES if name not in flat
        ]
        zero_values: list[str] = [
            name for name, val in flat.items() if val == 0.0
        ]

        present_count = len(flat) - len(missing)
        completeness = present_count / EXPECTED_FEATURE_COUNT if EXPECTED_FEATURE_COUNT > 0 else 0.0

        non_zero_count = len(flat) - len(zero_values)
        data_quality_flag = non_zero_count >= (len(flat) * 0.3)

        return QualityReport(
            completeness_score=round(completeness, 4),
            missing_features=missing,
            zero_value_features=zero_values,
            data_quality_flag=data_quality_flag,
            feature_count=len(flat),
            expected_count=EXPECTED_FEATURE_COUNT,
        )
