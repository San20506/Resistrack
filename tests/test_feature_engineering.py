"""Tests for M2.1 Feature Engineering pipeline."""

import pytest

from resistrack.features.extractor import (
    ALL_FEATURE_NAMES,
    EXPECTED_FEATURE_COUNT,
    ExtractedFeatures,
    FeatureConfig,
    FeatureExtractor,
    LAB_FEATURE_NAMES,
    MED_FEATURE_NAMES,
    CLINICAL_FEATURE_NAMES,
    HOSP_FEATURE_NAMES,
    VITAL_FEATURE_NAMES,
)
from resistrack.features.quality import DataQualityChecker, QualityReport


def _make_fhir_bundle(**kwargs: object) -> dict:
    """Create a minimal FHIR bundle for testing."""
    entries = []

    entries.append({
        "resource": {
            "resourceType": "Patient",
            "identifier": [{"value": "PT_TEST_001"}],
            "birthDate": "1960-05-15",
        }
    })

    entries.append({
        "resource": {
            "resourceType": "Encounter",
            "class": {"code": "IMP"},
            "period": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-05T00:00:00Z"},
        }
    })

    entries.append({
        "resource": {
            "resourceType": "Observation",
            "code": {"coding": [{"code": "6690-2", "display": "WBC"}]},
            "valueQuantity": {"value": 15.2, "unit": "10^3/uL"},
            "effectiveDateTime": "2026-01-04T10:00:00Z",
            "category": [{"coding": [{"code": "laboratory"}]}],
        }
    })

    entries.append({
        "resource": {
            "resourceType": "Observation",
            "code": {"coding": [{"code": "1988-5", "display": "CRP"}]},
            "valueQuantity": {"value": 85.0, "unit": "mg/L"},
            "effectiveDateTime": "2026-01-04T10:00:00Z",
            "category": [{"coding": [{"code": "laboratory"}]}],
        }
    })

    entries.append({
        "resource": {
            "resourceType": "Observation",
            "code": {"coding": [{"code": "8310-5", "display": "Temperature"}]},
            "valueQuantity": {"value": 38.9, "unit": "C"},
            "effectiveDateTime": "2026-01-04T10:00:00Z",
            "category": [{"coding": [{"code": "vital-signs"}]}],
        }
    })

    entries.append({
        "resource": {
            "resourceType": "Observation",
            "code": {"coding": [{"code": "8867-4", "display": "Heart Rate"}]},
            "valueQuantity": {"value": 110.0, "unit": "bpm"},
            "effectiveDateTime": "2026-01-04T10:00:00Z",
            "category": [{"coding": [{"code": "vital-signs"}]}],
        }
    })

    entries.append({
        "resource": {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {
                "coding": [{"code": "RX001", "display": "Piperacillin-tazobactam"}]
            },
            "status": "active",
        }
    })

    entries.append({
        "resource": {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {
                "coding": [{"code": "RX002", "display": "Meropenem"}]
            },
            "status": "active",
        }
    })

    entries.append({
        "resource": {
            "resourceType": "Condition",
            "code": {"coding": [{"code": "E11", "display": "Type 2 Diabetes"}]},
        }
    })

    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


class TestFeatureConfig:
    def test_default_config(self) -> None:
        config = FeatureConfig()
        assert config.lab_lookback_hours == 72
        assert config.med_lookback_hours == 168
        assert config.vital_lookback_hours == 24
        assert config.random_state == 42

    def test_custom_config(self) -> None:
        config = FeatureConfig(lab_lookback_hours=48, random_state=123)
        assert config.lab_lookback_hours == 48
        assert config.random_state == 123


class TestFeatureNames:
    def test_total_feature_count(self) -> None:
        assert len(ALL_FEATURE_NAMES) == EXPECTED_FEATURE_COUNT
        assert EXPECTED_FEATURE_COUNT == 47

    def test_lab_feature_count(self) -> None:
        assert len(LAB_FEATURE_NAMES) == 12

    def test_med_feature_count(self) -> None:
        assert len(MED_FEATURE_NAMES) == 10

    def test_clinical_feature_count(self) -> None:
        assert len(CLINICAL_FEATURE_NAMES) == 10

    def test_hospitalization_feature_count(self) -> None:
        assert len(HOSP_FEATURE_NAMES) == 8

    def test_vital_feature_count(self) -> None:
        assert len(VITAL_FEATURE_NAMES) == 7

    def test_no_duplicate_names(self) -> None:
        assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES))


class TestExtractedFeatures:
    def test_to_flat_dict(self) -> None:
        features = ExtractedFeatures(
            patient_token="PT_001",
            lab_features={"wbc_latest": 10.0},
            med_features={"abx_count_7d": 2.0},
        )
        flat = features.to_flat_dict()
        assert flat["wbc_latest"] == 10.0
        assert flat["abx_count_7d"] == 2.0

    def test_feature_count(self) -> None:
        features = ExtractedFeatures(
            patient_token="PT_001",
            lab_features={"a": 1.0, "b": 2.0},
            vital_features={"c": 3.0},
        )
        assert features.feature_count == 3

    def test_extraction_timestamp_auto(self) -> None:
        features = ExtractedFeatures(patient_token="PT_001")
        assert features.extraction_timestamp != ""


class TestFeatureExtractor:
    def test_extract_returns_features(self) -> None:
        extractor = FeatureExtractor()
        bundle = _make_fhir_bundle()
        result = extractor.extract(bundle)
        assert isinstance(result, ExtractedFeatures)
        assert result.patient_token == "PT_TEST_001"

    def test_extract_lab_features(self) -> None:
        extractor = FeatureExtractor()
        bundle = _make_fhir_bundle()
        result = extractor.extract(bundle)
        assert result.lab_features["wbc_latest"] == 15.2
        assert result.lab_features["crp_latest"] == 85.0

    def test_extract_vital_features(self) -> None:
        extractor = FeatureExtractor()
        bundle = _make_fhir_bundle()
        result = extractor.extract(bundle)
        assert result.vital_features["temperature_latest"] == 38.9
        assert result.vital_features["heart_rate_latest"] == 110.0

    def test_extract_med_features(self) -> None:
        extractor = FeatureExtractor()
        bundle = _make_fhir_bundle()
        result = extractor.extract(bundle)
        assert result.med_features["abx_count_7d"] == 2.0
        assert result.med_features["carbapenem_exposure"] == 1.0
        assert result.med_features["combination_therapy_flag"] == 1.0

    def test_extract_clinical_features(self) -> None:
        extractor = FeatureExtractor()
        bundle = _make_fhir_bundle()
        result = extractor.extract(bundle)
        assert result.clinical_features["diabetes_flag"] == 1.0
        assert result.clinical_features["age"] > 0

    def test_extract_hospitalization_features(self) -> None:
        extractor = FeatureExtractor()
        bundle = _make_fhir_bundle()
        result = extractor.extract(bundle)
        assert result.hospitalization_features["los_days"] == 4.0
        assert result.hospitalization_features["ward_type_encoded"] == 1.0

    def test_extract_total_feature_count(self) -> None:
        extractor = FeatureExtractor()
        bundle = _make_fhir_bundle()
        result = extractor.extract(bundle)
        assert result.feature_count == EXPECTED_FEATURE_COUNT

    def test_extract_empty_bundle(self) -> None:
        extractor = FeatureExtractor()
        result = extractor.extract({"entry": []})
        assert result.patient_token == "UNKNOWN"
        assert result.feature_count == EXPECTED_FEATURE_COUNT

    def test_custom_config(self) -> None:
        config = FeatureConfig(lab_lookback_hours=48)
        extractor = FeatureExtractor(config=config)
        assert extractor.config.lab_lookback_hours == 48


class TestDataQualityChecker:
    def test_quality_report_from_good_data(self) -> None:
        extractor = FeatureExtractor()
        checker = DataQualityChecker()
        bundle = _make_fhir_bundle()
        features = extractor.extract(bundle)
        report = checker.assess(features)
        assert isinstance(report, QualityReport)
        assert report.completeness_score == 1.0
        assert report.feature_count == EXPECTED_FEATURE_COUNT

    def test_quality_report_from_empty_data(self) -> None:
        extractor = FeatureExtractor()
        checker = DataQualityChecker()
        features = extractor.extract({"entry": []})
        report = checker.assess(features)
        assert report.completeness_score == 1.0

    def test_data_quality_flag_with_good_data(self) -> None:
        extractor = FeatureExtractor()
        checker = DataQualityChecker()
        bundle = _make_fhir_bundle()
        features = extractor.extract(bundle)
        report = checker.assess(features)
        assert isinstance(report.data_quality_flag, bool)

    def test_is_acceptable(self) -> None:
        report = QualityReport(
            completeness_score=0.8,
            missing_features=[],
            zero_value_features=[],
            data_quality_flag=True,
            feature_count=47,
            expected_count=47,
        )
        assert report.is_acceptable is True

    def test_is_not_acceptable(self) -> None:
        report = QualityReport(
            completeness_score=0.4,
            missing_features=["a", "b"],
            zero_value_features=[],
            data_quality_flag=False,
            feature_count=20,
            expected_count=47,
        )
        assert report.is_acceptable is False
