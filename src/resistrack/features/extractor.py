"""Feature extraction pipeline for AMR risk prediction.

Extracts 47 tabular features from FHIR resources organized into 5 feature groups:
- Lab Trends (12 features)
- Medication History (10 features)
- Clinical Context (10 features)
- Hospitalization (8 features)
- Vitals (7 features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from resistrack.common.constants import RANDOM_STATE


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    lab_lookback_hours: int = 72
    med_lookback_hours: int = 168
    vital_lookback_hours: int = 24
    random_state: int = RANDOM_STATE


@dataclass
class ExtractedFeatures:
    """Container for extracted feature vectors."""

    patient_token: str
    lab_features: dict[str, float] = field(default_factory=dict)
    med_features: dict[str, float] = field(default_factory=dict)
    clinical_features: dict[str, float] = field(default_factory=dict)
    hospitalization_features: dict[str, float] = field(default_factory=dict)
    vital_features: dict[str, float] = field(default_factory=dict)
    extraction_timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.extraction_timestamp:
            self.extraction_timestamp = datetime.utcnow().isoformat()

    def to_flat_dict(self) -> dict[str, float]:
        """Flatten all feature groups into a single dictionary."""
        result: dict[str, float] = {}
        result.update(self.lab_features)
        result.update(self.med_features)
        result.update(self.clinical_features)
        result.update(self.hospitalization_features)
        result.update(self.vital_features)
        return result

    @property
    def feature_count(self) -> int:
        """Total number of extracted features."""
        return len(self.to_flat_dict())


# Lab feature names (12)
LAB_FEATURE_NAMES: list[str] = [
    "wbc_latest",
    "wbc_trend_slope",
    "wbc_max_72h",
    "crp_latest",
    "crp_trend_slope",
    "procalcitonin_latest",
    "procalcitonin_elevated",
    "lactate_latest",
    "lactate_elevated",
    "blood_culture_positive",
    "culture_count_30d",
    "prior_resistance_count",
]

# Medication feature names (10)
MED_FEATURE_NAMES: list[str] = [
    "abx_count_7d",
    "abx_duration_current",
    "abx_class_count",
    "broad_spectrum_flag",
    "escalation_count",
    "deescalation_count",
    "abx_changes_48h",
    "carbapenem_exposure",
    "fluoroquinolone_exposure",
    "combination_therapy_flag",
]

# Clinical context feature names (10)
CLINICAL_FEATURE_NAMES: list[str] = [
    "age",
    "charlson_comorbidity_index",
    "immunosuppressed_flag",
    "diabetes_flag",
    "renal_impairment_flag",
    "surgical_procedure_flag",
    "icu_admission_flag",
    "ventilator_flag",
    "central_line_flag",
    "urinary_catheter_flag",
]

# Hospitalization feature names (8)
HOSP_FEATURE_NAMES: list[str] = [
    "los_days",
    "los_icu_days",
    "transfer_count",
    "readmission_30d_flag",
    "admission_source_ed",
    "admission_source_transfer",
    "ward_type_encoded",
    "bed_occupancy_rate",
]

# Vitals feature names (7)
VITAL_FEATURE_NAMES: list[str] = [
    "temperature_latest",
    "temperature_max_24h",
    "heart_rate_latest",
    "systolic_bp_latest",
    "respiratory_rate_latest",
    "spo2_latest",
    "sofa_score",
]

ALL_FEATURE_NAMES: list[str] = (
    LAB_FEATURE_NAMES
    + MED_FEATURE_NAMES
    + CLINICAL_FEATURE_NAMES
    + HOSP_FEATURE_NAMES
    + VITAL_FEATURE_NAMES
)

EXPECTED_FEATURE_COUNT = 47


class FeatureExtractor:
    """Extracts AMR risk features from FHIR resource bundles."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def extract(self, fhir_bundle: dict[str, Any]) -> ExtractedFeatures:
        """Extract all features from a FHIR resource bundle."""
        patient_token = self._get_patient_token(fhir_bundle)
        now = datetime.utcnow()

        return ExtractedFeatures(
            patient_token=patient_token,
            lab_features=self._extract_lab_features(fhir_bundle, now),
            med_features=self._extract_med_features(fhir_bundle, now),
            clinical_features=self._extract_clinical_features(fhir_bundle),
            hospitalization_features=self._extract_hospitalization_features(fhir_bundle),
            vital_features=self._extract_vital_features(fhir_bundle, now),
        )

    def _get_patient_token(self, bundle: dict[str, Any]) -> str:
        """Extract tokenized patient identifier."""
        entries = bundle.get("entry", [])
        for entry in entries:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                identifiers = resource.get("identifier", [])
                if identifiers:
                    return str(identifiers[0].get("value", "UNKNOWN"))
        return "UNKNOWN"

    def _extract_lab_features(
        self, bundle: dict[str, Any], now: datetime
    ) -> dict[str, float]:
        """Extract laboratory result features."""
        features: dict[str, float] = {name: 0.0 for name in LAB_FEATURE_NAMES}
        cutoff = now - timedelta(hours=self.config.lab_lookback_hours)

        observations = self._get_resources(bundle, "Observation")
        wbc_values: list[float] = []

        for obs in observations:
            code = self._get_observation_code(obs)
            value = self._get_observation_value(obs)
            effective = obs.get("effectiveDateTime", "")

            if value is None:
                continue

            if code in ("6690-2", "WBC"):
                wbc_values.append(value)
                features["wbc_latest"] = value
            elif code in ("1988-5", "CRP"):
                features["crp_latest"] = value
            elif code in ("33959-8", "PCT"):
                features["procalcitonin_latest"] = value
                features["procalcitonin_elevated"] = 1.0 if value > 0.5 else 0.0
            elif code in ("2524-7", "LACTATE"):
                features["lactate_latest"] = value
                features["lactate_elevated"] = 1.0 if value > 2.0 else 0.0
            elif code in ("600-7", "BLOOD_CULTURE"):
                if value > 0:
                    features["blood_culture_positive"] = 1.0

        if wbc_values:
            features["wbc_max_72h"] = max(wbc_values)
            if len(wbc_values) >= 2:
                features["wbc_trend_slope"] = wbc_values[-1] - wbc_values[0]

        return features

    def _extract_med_features(
        self, bundle: dict[str, Any], now: datetime
    ) -> dict[str, float]:
        """Extract medication/antibiotic history features."""
        features: dict[str, float] = {name: 0.0 for name in MED_FEATURE_NAMES}

        med_requests = self._get_resources(bundle, "MedicationRequest")
        abx_classes: set[str] = set()

        for med in med_requests:
            coding = (
                med.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
                if med.get("medicationCodeableConcept", {}).get("coding")
                else {}
            )
            display = coding.get("display", "").lower()

            if self._is_antibiotic(display):
                features["abx_count_7d"] += 1
                abx_class = self._classify_antibiotic(display)
                abx_classes.add(abx_class)

                if abx_class == "carbapenem":
                    features["carbapenem_exposure"] = 1.0
                elif abx_class == "fluoroquinolone":
                    features["fluoroquinolone_exposure"] = 1.0

        features["abx_class_count"] = float(len(abx_classes))
        features["broad_spectrum_flag"] = 1.0 if len(abx_classes) >= 3 else 0.0
        features["combination_therapy_flag"] = (
            1.0 if features["abx_count_7d"] >= 2 else 0.0
        )

        return features

    def _extract_clinical_features(
        self, bundle: dict[str, Any]
    ) -> dict[str, float]:
        """Extract clinical context features."""
        features: dict[str, float] = {name: 0.0 for name in CLINICAL_FEATURE_NAMES}

        patients = self._get_resources(bundle, "Patient")
        if patients:
            patient = patients[0]
            birth_date = patient.get("birthDate", "")
            if birth_date:
                try:
                    birth = datetime.strptime(birth_date, "%Y-%m-%d")
                    features["age"] = float((datetime.utcnow() - birth).days // 365)
                except ValueError:
                    pass

        conditions = self._get_resources(bundle, "Condition")
        for condition in conditions:
            code = condition.get("code", {}).get("coding", [{}])
            code_value = code[0].get("code", "") if code else ""

            if code_value in ("E10", "E11", "250"):
                features["diabetes_flag"] = 1.0
            elif code_value in ("N17", "N18", "N19"):
                features["renal_impairment_flag"] = 1.0

        return features

    def _extract_hospitalization_features(
        self, bundle: dict[str, Any]
    ) -> dict[str, float]:
        """Extract hospitalization-related features."""
        features: dict[str, float] = {name: 0.0 for name in HOSP_FEATURE_NAMES}

        encounters = self._get_resources(bundle, "Encounter")
        if encounters:
            encounter = encounters[0]
            period = encounter.get("period", {})
            start = period.get("start", "")
            end = period.get("end", "")

            if start:
                try:
                    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    end_dt = (
                        datetime.fromisoformat(end.replace("Z", "+00:00"))
                        if end
                        else datetime.utcnow()
                    )
                    los = (end_dt.replace(tzinfo=None) - start_dt.replace(tzinfo=None)).days
                    features["los_days"] = float(max(los, 0))
                except ValueError:
                    pass

            enc_class = encounter.get("class", {}).get("code", "")
            if enc_class == "IMP":
                features["ward_type_encoded"] = 1.0
            elif enc_class == "EMER":
                features["admission_source_ed"] = 1.0

        return features

    def _extract_vital_features(
        self, bundle: dict[str, Any], now: datetime
    ) -> dict[str, float]:
        """Extract vital signs features."""
        features: dict[str, float] = {name: 0.0 for name in VITAL_FEATURE_NAMES}

        observations = self._get_resources(bundle, "Observation")
        temps: list[float] = []

        for obs in observations:
            code = self._get_observation_code(obs)
            value = self._get_observation_value(obs)
            category = obs.get("category", [{}])
            cat_code = (
                category[0].get("coding", [{}])[0].get("code", "")
                if category and category[0].get("coding")
                else ""
            )

            if value is None:
                continue

            if cat_code == "vital-signs" or code in (
                "8310-5", "8867-4", "8480-6", "9279-1", "2708-6",
            ):
                if code in ("8310-5", "TEMP"):
                    features["temperature_latest"] = value
                    temps.append(value)
                elif code in ("8867-4", "HR"):
                    features["heart_rate_latest"] = value
                elif code in ("8480-6", "SBP"):
                    features["systolic_bp_latest"] = value
                elif code in ("9279-1", "RR"):
                    features["respiratory_rate_latest"] = value
                elif code in ("2708-6", "SPO2"):
                    features["spo2_latest"] = value

        if temps:
            features["temperature_max_24h"] = max(temps)

        return features

    def _get_resources(
        self, bundle: dict[str, Any], resource_type: str
    ) -> list[dict[str, Any]]:
        """Extract resources of a given type from a FHIR bundle."""
        entries = bundle.get("entry", [])
        return [
            entry["resource"]
            for entry in entries
            if entry.get("resource", {}).get("resourceType") == resource_type
        ]

    def _get_observation_code(self, obs: dict[str, Any]) -> str:
        """Get the LOINC or local code from an Observation."""
        coding = obs.get("code", {}).get("coding", [])
        return coding[0].get("code", "") if coding else ""

    def _get_observation_value(self, obs: dict[str, Any]) -> float | None:
        """Get the numeric value from an Observation."""
        vq = obs.get("valueQuantity", {})
        if "value" in vq:
            return float(vq["value"])
        return None

    def _is_antibiotic(self, display: str) -> bool:
        """Check if a medication display name is an antibiotic."""
        abx_keywords = [
            "cillin", "cephalexin", "cefazolin", "ceftriaxone", "cefepime",
            "meropenem", "imipenem", "ertapenem", "azithromycin", "ciprofloxacin",
            "levofloxacin", "vancomycin", "gentamicin", "amikacin", "tobramycin",
            "metronidazole", "doxycycline", "trimethoprim", "linezolid",
            "piperacillin", "ampicillin", "oxacillin",
        ]
        return any(kw in display for kw in abx_keywords)

    def _classify_antibiotic(self, display: str) -> str:
        """Classify antibiotic into class."""
        if any(kw in display for kw in ["cillin", "ampicillin", "oxacillin", "piperacillin"]):
            return "penicillin"
        elif any(kw in display for kw in ["cef", "cephalexin"]):
            return "cephalosporin"
        elif any(kw in display for kw in ["meropenem", "imipenem", "ertapenem"]):
            return "carbapenem"
        elif any(kw in display for kw in ["ciprofloxacin", "levofloxacin"]):
            return "fluoroquinolone"
        elif any(kw in display for kw in ["gentamicin", "amikacin", "tobramycin"]):
            return "aminoglycoside"
        return "other"
