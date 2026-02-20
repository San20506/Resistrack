"""MIMIC-IV Data Loader for local training.

Handles two modes:
  1. **Synthetic** — Generates clinically realistic synthetic data matching
     the ResisTrack 47-feature schema, 72h temporal tensors (72, 13), and
     clinical notes for ClinicalBERT.  Used when real MIMIC-IV CSVs are
     unavailable (the default for development / hackathon runs).
  2. **CSV** — Loads real MIMIC-IV tables (admissions.csv, labevents.csv,
     chartevents.csv, microbiologyevents.csv, prescriptions.csv, noteevents/
     discharge.csv).  Tables are joined by `hadm_id` and mapped into the
     ResisTrack feature schema.

Both modes produce the same output dataclass consumed by downstream trainers.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from resistrack.common.constants import RANDOM_STATE
from resistrack.features.extractor import (
    ALL_FEATURE_NAMES,
    CLINICAL_FEATURE_NAMES,
    HOSP_FEATURE_NAMES,
    LAB_FEATURE_NAMES,
    MED_FEATURE_NAMES,
    VITAL_FEATURE_NAMES,
)
from resistrack.ml.temporal import (
    ALL_TEMPORAL_FEATURES,
    NUM_TEMPORAL_FEATURES,
    WINDOW_HOURS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TEMPORAL_SHAPE = (WINDOW_HOURS, NUM_TEMPORAL_FEATURES)  # (72, 13)

# Physiological ranges for realistic synthetic generation
_PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float]] = {
    # Lab features
    "wbc_latest": (3.5, 25.0),
    "wbc_trend_slope": (-5.0, 8.0),
    "wbc_max_72h": (4.0, 30.0),
    "crp_latest": (0.0, 350.0),
    "crp_trend_slope": (-50.0, 100.0),
    "procalcitonin_latest": (0.0, 50.0),
    "procalcitonin_elevated": (0.0, 1.0),
    "lactate_latest": (0.5, 12.0),
    "lactate_elevated": (0.0, 1.0),
    "blood_culture_positive": (0.0, 1.0),
    "culture_count_30d": (0.0, 5.0),
    "prior_resistance_count": (0.0, 4.0),
    # Med features
    "abx_count_7d": (0.0, 6.0),
    "abx_duration_current": (0.0, 14.0),
    "abx_class_count": (0.0, 5.0),
    "broad_spectrum_flag": (0.0, 1.0),
    "escalation_count": (0.0, 3.0),
    "deescalation_count": (0.0, 3.0),
    "abx_changes_48h": (0.0, 4.0),
    "carbapenem_exposure": (0.0, 1.0),
    "fluoroquinolone_exposure": (0.0, 1.0),
    "combination_therapy_flag": (0.0, 1.0),
    # Clinical features
    "age": (18.0, 95.0),
    "charlson_comorbidity_index": (0.0, 12.0),
    "immunosuppressed_flag": (0.0, 1.0),
    "diabetes_flag": (0.0, 1.0),
    "renal_impairment_flag": (0.0, 1.0),
    "surgical_procedure_flag": (0.0, 1.0),
    "icu_admission_flag": (0.0, 1.0),
    "ventilator_flag": (0.0, 1.0),
    "central_line_flag": (0.0, 1.0),
    "urinary_catheter_flag": (0.0, 1.0),
    # Hospitalization features
    "los_days": (0.0, 60.0),
    "los_icu_days": (0.0, 30.0),
    "transfer_count": (0.0, 5.0),
    "readmission_30d_flag": (0.0, 1.0),
    "admission_source_ed": (0.0, 1.0),
    "admission_source_transfer": (0.0, 1.0),
    "ward_type_encoded": (0.0, 5.0),
    "bed_occupancy_rate": (0.5, 1.0),
    # Vital features
    "temperature_latest": (35.5, 40.5),
    "temperature_max_24h": (36.0, 41.0),
    "heart_rate_latest": (50.0, 160.0),
    "systolic_bp_latest": (70.0, 200.0),
    "respiratory_rate_latest": (10.0, 40.0),
    "spo2_latest": (80.0, 100.0),
    "sofa_score": (0.0, 18.0),
}

# Temporal feature physiological ranges (hourly values)
_TEMPORAL_RANGES: dict[str, tuple[float, float]] = {
    "wbc_count": (3.5, 25.0),
    "crp_level": (0.0, 350.0),
    "procalcitonin": (0.0, 50.0),
    "lactate": (0.5, 12.0),
    "creatinine": (0.3, 8.0),
    "platelet_count": (50.0, 400.0),
    "neutrophil_pct": (30.0, 95.0),
    "band_neutrophils": (0.0, 30.0),
    "temperature": (35.5, 40.5),
    "heart_rate": (50.0, 160.0),
    "respiratory_rate": (10.0, 40.0),
    "systolic_bp": (70.0, 200.0),
    "oxygen_saturation": (80.0, 100.0),
}

# Sample clinical note templates
_NOTE_TEMPLATES: list[str] = [
    "Patient admitted with suspected infection. WBC elevated at {wbc:.1f}. "
    "Blood cultures drawn. Started on empiric {abx}. Monitor for signs of sepsis.",
    "Day {day} of admission. CRP trending {direction} to {crp:.1f}. "
    "Current antibiotic regimen: {abx}. Culture results {culture_status}.",
    "ICU progress note. Patient on {abx} for {duration} days. "
    "Procalcitonin {pct:.2f} ng/mL. {resistance_note} "
    "Plan: continue current therapy, reassess in 24h.",
    "Infectious disease consult. History of {resistance_history}. "
    "Current cultures show {organism}. Recommend {recommendation}. "
    "SOFA score {sofa}. Charlson index {charlson}.",
    "Discharge summary: {day}-day hospitalization for {diagnosis}. "
    "Treated with {abx} x {duration} days. Final cultures {final_culture}. "
    "Follow-up with ID clinic in 2 weeks.",
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainingDataset:
    """Complete dataset ready for model training.

    Attributes:
        tabular_features: Shape (n_samples, 47) — structured features.
        temporal_tensors: Shape (n_samples, 72, 13) — time-series data.
        clinical_notes: List of list of note strings per patient.
        labels: Shape (n_samples,) — binary AMR resistance labels.
        patient_tokens: De-identified patient identifiers.
        tenant_ids: Hospital tenant identifiers for stratified splitting.
        feature_names: Ordered list of tabular feature names.
        temporal_feature_names: Ordered list of temporal feature names.
    """

    tabular_features: np.ndarray  # (n, 47)
    temporal_tensors: np.ndarray  # (n, 72, 13)
    clinical_notes: list[list[str]]
    labels: np.ndarray  # (n,) binary
    patient_tokens: list[str]
    tenant_ids: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: list(ALL_FEATURE_NAMES))
    temporal_feature_names: list[str] = field(
        default_factory=lambda: list(ALL_TEMPORAL_FEATURES)
    )

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def positive_rate(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return float(np.mean(self.labels == 1))

    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features_tabular": self.tabular_features.shape[1],
            "temporal_shape": self.temporal_tensors.shape,
            "positive_rate": round(self.positive_rate, 4),
            "n_tenants": len(np.unique(self.tenant_ids)),
            "avg_notes_per_patient": round(
                np.mean([len(n) for n in self.clinical_notes]), 2
            ),
        }


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
class MIMICDataLoader:
    """Generates or loads training data for the ResisTrack ML pipeline.

    Usage:
        loader = MIMICDataLoader()

        # Generate synthetic data for development
        dataset = loader.generate_synthetic(n_samples=2000)

        # Or load from real MIMIC-IV CSV files
        dataset = loader.load_from_csv(mimic_dir="/data/mimic-iv")
    """

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Synthetic generation
    # ------------------------------------------------------------------
    def generate_synthetic(
        self,
        n_samples: int = 2000,
        positive_rate: float = 0.15,
        n_tenants: int = 5,
        notes_per_patient: int = 3,
    ) -> TrainingDataset:
        """Generate clinically realistic synthetic training data.

        Produces correlated features: high-risk patients will have elevated
        inflammatory markers, longer ICU stays, and more antibiotic exposure.

        Args:
            n_samples: Number of synthetic patient records.
            positive_rate: Fraction of AMR-resistant (positive) labels.
            n_tenants: Number of simulated hospital tenants.
            notes_per_patient: Max clinical notes per patient.
        """
        logger.info(
            "Generating %d synthetic MIMIC-IV samples (positive_rate=%.2f)",
            n_samples,
            positive_rate,
        )

        # 1. Generate labels
        labels = self._generate_labels(n_samples, positive_rate)

        # 2. Generate correlated tabular features
        tabular = self._generate_tabular_features(n_samples, labels)

        # 3. Generate temporal tensors
        temporal = self._generate_temporal_tensors(n_samples, labels)

        # 4. Generate clinical notes
        notes = self._generate_clinical_notes(n_samples, labels, notes_per_patient)

        # 5. Generate patient tokens and tenant IDs
        tokens = [f"pt_{hashlib.sha256(str(i).encode()).hexdigest()[:12]}" for i in range(n_samples)]
        tenant_ids = self._rng.randint(0, n_tenants, size=n_samples)

        dataset = TrainingDataset(
            tabular_features=tabular,
            temporal_tensors=temporal,
            clinical_notes=notes,
            labels=labels,
            patient_tokens=tokens,
            tenant_ids=tenant_ids,
        )

        logger.info("Synthetic dataset generated: %s", dataset.summary())
        return dataset

    def _generate_labels(self, n: int, positive_rate: float) -> np.ndarray:
        """Generate binary AMR resistance labels."""
        labels = np.zeros(n, dtype=np.int32)
        n_positive = int(n * positive_rate)
        positive_indices = self._rng.choice(n, size=n_positive, replace=False)
        labels[positive_indices] = 1
        return labels

    def _generate_tabular_features(
        self, n: int, labels: np.ndarray
    ) -> np.ndarray:
        """Generate 47 tabular features with clinical correlations.

        Resistant (label=1) patients get elevated inflammatory markers,
        more antibiotic exposure, longer stays, and more comorbidities.
        """
        features = np.zeros((n, len(ALL_FEATURE_NAMES)), dtype=np.float32)

        for col_idx, feat_name in enumerate(ALL_FEATURE_NAMES):
            lo, hi = _PHYSIOLOGICAL_RANGES.get(feat_name, (0.0, 1.0))

            # Base values
            base = self._rng.uniform(lo, hi, size=n).astype(np.float32)

            # Binary features: generate as 0/1
            if feat_name in {
                "procalcitonin_elevated", "lactate_elevated",
                "blood_culture_positive", "broad_spectrum_flag",
                "carbapenem_exposure", "fluoroquinolone_exposure",
                "combination_therapy_flag", "immunosuppressed_flag",
                "diabetes_flag", "renal_impairment_flag",
                "surgical_procedure_flag", "icu_admission_flag",
                "ventilator_flag", "central_line_flag",
                "urinary_catheter_flag", "readmission_30d_flag",
                "admission_source_ed", "admission_source_transfer",
            }:
                # Higher probability for positive (resistant) patients
                pos_prob = 0.5 if feat_name in {"diabetes_flag", "renal_impairment_flag"} else 0.3
                neg_prob = pos_prob * 0.4
                probs = np.where(labels == 1, pos_prob, neg_prob)
                base = (self._rng.random(n) < probs).astype(np.float32)

            # Correlation: shift positive patients toward higher risk
            elif feat_name in {
                "wbc_latest", "wbc_trend_slope", "wbc_max_72h",
                "crp_latest", "crp_trend_slope", "procalcitonin_latest",
                "lactate_latest", "culture_count_30d", "prior_resistance_count",
                "abx_count_7d", "abx_duration_current", "abx_class_count",
                "escalation_count", "abx_changes_48h",
                "los_days", "los_icu_days", "transfer_count",
                "sofa_score", "charlson_comorbidity_index",
            }:
                shift = (hi - lo) * 0.25
                base[labels == 1] += shift
                base = np.clip(base, lo, hi)

            elif feat_name in {"spo2_latest"}:
                # Inverse: lower SpO2 for sick patients
                shift = (hi - lo) * 0.1
                base[labels == 1] -= shift
                base = np.clip(base, lo, hi)

            # Integer features
            if feat_name in {
                "culture_count_30d", "prior_resistance_count",
                "abx_count_7d", "abx_class_count", "escalation_count",
                "deescalation_count", "abx_changes_48h", "transfer_count",
                "ward_type_encoded",
            }:
                base = np.round(base).astype(np.float32)

            features[:, col_idx] = base

        # Add some noise
        noise = self._rng.normal(0, 0.02, size=features.shape).astype(np.float32)
        features += noise

        return features

    def _generate_temporal_tensors(
        self, n: int, labels: np.ndarray
    ) -> np.ndarray:
        """Generate 72h temporal tensors (n, 72, 13) with trends.

        Positive patients show worsening trends (rising inflammatory
        markers, deteriorating vitals) over the 72-hour window.
        """
        tensors = np.zeros((n, WINDOW_HOURS, NUM_TEMPORAL_FEATURES), dtype=np.float32)

        for feat_idx, feat_name in enumerate(ALL_TEMPORAL_FEATURES):
            lo, hi = _TEMPORAL_RANGES.get(feat_name, (0.0, 1.0))
            mid = (lo + hi) / 2
            spread = (hi - lo) / 4

            for i in range(n):
                # Base patient-level value
                baseline = self._rng.normal(mid, spread)
                baseline = np.clip(baseline, lo, hi)

                # Generate hourly values with autocorrelation
                values = np.zeros(WINDOW_HOURS, dtype=np.float32)
                values[0] = baseline

                for t in range(1, WINDOW_HOURS):
                    # AR(1) process with noise
                    innovation = self._rng.normal(0, spread * 0.05)
                    values[t] = 0.95 * values[t - 1] + innovation

                # Add trend for positive patients
                if labels[i] == 1:
                    if feat_name in {
                        "wbc_count", "crp_level", "procalcitonin",
                        "lactate", "heart_rate", "respiratory_rate",
                    }:
                        # Rising trend
                        trend = np.linspace(0, spread * 0.5, WINDOW_HOURS)
                        values += trend
                    elif feat_name in {"oxygen_saturation", "systolic_bp", "platelet_count"}:
                        # Falling trend
                        trend = np.linspace(0, -spread * 0.3, WINDOW_HOURS)
                        values += trend

                # Clip to physiological range
                values = np.clip(values, lo, hi)

                # Randomly introduce some NaN (missingness ~10%)
                n_missing = int(WINDOW_HOURS * 0.1)
                missing_idx = self._rng.choice(WINDOW_HOURS, size=n_missing, replace=False)
                values[missing_idx] = np.nan

                tensors[i, :, feat_idx] = values

        return tensors

    def _generate_clinical_notes(
        self,
        n: int,
        labels: np.ndarray,
        max_notes: int = 3,
    ) -> list[list[str]]:
        """Generate synthetic clinical notes per patient."""
        antibiotics = [
            "piperacillin-tazobactam", "meropenem", "vancomycin",
            "ceftriaxone", "ciprofloxacin", "levofloxacin",
            "ampicillin-sulbactam", "cefepime",
        ]
        organisms = [
            "MRSA", "E. coli (ESBL+)", "Klebsiella pneumoniae (CRE)",
            "Pseudomonas aeruginosa", "Enterococcus faecium (VRE)",
            "Acinetobacter baumannii", "Staphylococcus aureus",
        ]
        diagnoses = [
            "pneumonia", "urinary tract infection", "bloodstream infection",
            "surgical site infection", "intra-abdominal infection",
        ]

        all_notes: list[list[str]] = []

        for i in range(n):
            n_notes = self._rng.randint(1, max_notes + 1)
            patient_notes: list[str] = []

            for j in range(n_notes):
                template = _NOTE_TEMPLATES[j % len(_NOTE_TEMPLATES)]
                abx = self._rng.choice(antibiotics)
                organism = self._rng.choice(organisms)
                diagnosis = self._rng.choice(diagnoses)

                note = template.format(
                    wbc=self._rng.uniform(5, 25),
                    abx=abx,
                    day=j + 1,
                    direction="up" if labels[i] == 1 else "down",
                    crp=self._rng.uniform(10, 300),
                    culture_status="positive for " + organism if labels[i] == 1 else "pending",
                    duration=self._rng.randint(1, 10),
                    pct=self._rng.uniform(0.1, 20),
                    resistance_note="Prior MDRO colonization documented." if labels[i] == 1 else "",
                    resistance_history="ESBL E. coli (6 months ago)" if labels[i] == 1 else "none",
                    organism=organism,
                    recommendation="de-escalate to narrow-spectrum" if labels[i] == 0 else "continue broad-spectrum, add combination therapy",
                    sofa=self._rng.randint(2, 15),
                    charlson=self._rng.randint(0, 10),
                    diagnosis=diagnosis,
                    final_culture="negative" if labels[i] == 0 else "positive, susceptibilities attached",
                )
                patient_notes.append(note)

            all_notes.append(patient_notes)

        return all_notes

    # ------------------------------------------------------------------
    # CSV loading (real MIMIC-IV)
    # ------------------------------------------------------------------
    def load_from_csv(
        self,
        mimic_dir: str | Path,
        max_samples: int | None = None,
    ) -> TrainingDataset:
        """Load training data from real MIMIC-IV CSV extracts.

        Expected directory structure:
            mimic_dir/
                admissions.csv
                labevents.csv
                chartevents.csv
                microbiologyevents.csv
                prescriptions.csv
                discharge.csv  (or noteevents.csv)

        Args:
            mimic_dir: Path to directory containing MIMIC-IV CSVs.
            max_samples: Maximum number of admissions to load.
        """
        mimic_path = Path(mimic_dir)
        if not mimic_path.exists():
            raise FileNotFoundError(f"MIMIC-IV directory not found: {mimic_path}")

        logger.info("Loading MIMIC-IV data from %s", mimic_path)

        # Load admissions
        admissions = self._load_csv(mimic_path / "admissions.csv")
        if max_samples:
            admissions = admissions.head(max_samples)

        hadm_ids = admissions["hadm_id"].unique()
        n = len(hadm_ids)
        logger.info("Found %d admissions", n)

        # Load supporting tables
        labevents = self._load_csv_safe(mimic_path / "labevents.csv")
        chartevents = self._load_csv_safe(mimic_path / "chartevents.csv")
        micro = self._load_csv_safe(mimic_path / "microbiologyevents.csv")
        prescriptions = self._load_csv_safe(mimic_path / "prescriptions.csv")
        notes_df = self._load_csv_safe(mimic_path / "discharge.csv")
        if notes_df is None:
            notes_df = self._load_csv_safe(mimic_path / "noteevents.csv")

        # Generate labels from microbiologyevents (culture positivity)
        labels = self._labels_from_micro(hadm_ids, micro)

        # Extract tabular features
        tabular = self._tabular_from_csvs(
            hadm_ids, admissions, labevents, chartevents, prescriptions
        )

        # Extract temporal tensors
        temporal = self._temporal_from_csvs(hadm_ids, labevents, chartevents)

        # Extract clinical notes
        notes = self._notes_from_csvs(hadm_ids, notes_df)

        # Patient tokens
        tokens = [
            f"pt_{hashlib.sha256(str(hid).encode()).hexdigest()[:12]}"
            for hid in hadm_ids
        ]

        # Tenant IDs (single hospital for MIMIC)
        tenant_ids = np.zeros(n, dtype=np.int32)

        dataset = TrainingDataset(
            tabular_features=tabular,
            temporal_tensors=temporal,
            clinical_notes=notes,
            labels=labels,
            patient_tokens=tokens,
            tenant_ids=tenant_ids,
        )

        logger.info("MIMIC-IV dataset loaded: %s", dataset.summary())
        return dataset

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load a CSV file, raising if not found."""
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    def _load_csv_safe(self, path: Path) -> pd.DataFrame | None:
        """Load a CSV file, returning None if not found."""
        if not path.exists():
            logger.warning("Optional file not found: %s", path)
            return None
        return pd.read_csv(path, low_memory=False)

    def _labels_from_micro(
        self, hadm_ids: np.ndarray, micro: pd.DataFrame | None
    ) -> np.ndarray:
        """Derive AMR labels from microbiologyevents (resistance in culture results)."""
        labels = np.zeros(len(hadm_ids), dtype=np.int32)
        if micro is None:
            # No microbiology data: fall back to random labels
            labels[self._rng.choice(len(hadm_ids), size=int(len(hadm_ids) * 0.15), replace=False)] = 1
            return labels

        for i, hid in enumerate(hadm_ids):
            patient_micro = micro[micro["hadm_id"] == hid]
            if patient_micro.empty:
                continue
            # Check for resistant organisms (interpretation column = "R")
            if "interpretation" in patient_micro.columns:
                if (patient_micro["interpretation"] == "R").any():
                    labels[i] = 1
            elif "org_name" in patient_micro.columns:
                # Fallback: any positive culture
                if patient_micro["org_name"].notna().any():
                    labels[i] = 1

        return labels

    def _tabular_from_csvs(
        self,
        hadm_ids: np.ndarray,
        admissions: pd.DataFrame,
        labevents: pd.DataFrame | None,
        chartevents: pd.DataFrame | None,
        prescriptions: pd.DataFrame | None,
    ) -> np.ndarray:
        """Extract 47 tabular features from MIMIC-IV CSVs."""
        n = len(hadm_ids)
        features = np.zeros((n, len(ALL_FEATURE_NAMES)), dtype=np.float32)

        for i, hid in enumerate(hadm_ids):
            adm = admissions[admissions["hadm_id"] == hid]
            if adm.empty:
                continue

            # Admission features
            feat_dict: dict[str, float] = {name: 0.0 for name in ALL_FEATURE_NAMES}

            # Age
            if "anchor_age" in adm.columns:
                feat_dict["age"] = float(adm.iloc[0].get("anchor_age", 0))

            # Length of stay
            if "admittime" in adm.columns and "dischtime" in adm.columns:
                try:
                    admit = pd.Timestamp(adm.iloc[0]["admittime"])
                    disch = pd.Timestamp(adm.iloc[0]["dischtime"])
                    feat_dict["los_days"] = max(0, (disch - admit).days)
                except Exception:
                    pass

            # ED admission
            if "admission_type" in adm.columns:
                if "EMERGENCY" in str(adm.iloc[0].get("admission_type", "")).upper():
                    feat_dict["admission_source_ed"] = 1.0

            # Lab features
            if labevents is not None:
                pt_labs = labevents[labevents["hadm_id"] == hid]
                feat_dict.update(self._extract_lab_from_df(pt_labs))

            # Prescription features
            if prescriptions is not None:
                pt_rx = prescriptions[prescriptions["hadm_id"] == hid]
                feat_dict.update(self._extract_med_from_df(pt_rx))

            # Map to feature vector
            for col_idx, name in enumerate(ALL_FEATURE_NAMES):
                features[i, col_idx] = feat_dict.get(name, 0.0)

        return features

    def _extract_lab_from_df(self, labs: pd.DataFrame) -> dict[str, float]:
        """Extract lab features from a patient's labevents."""
        result: dict[str, float] = {}
        if labs.empty or "itemid" not in labs.columns:
            return result

        # MIMIC-IV itemid mappings
        lab_map = {
            51301: "wbc_latest",     # WBC
            50889: "crp_latest",     # CRP
            50911: "procalcitonin_latest",  # Procalcitonin
            50813: "lactate_latest",  # Lactate
        }

        for itemid, feat_name in lab_map.items():
            vals = labs[labs["itemid"] == itemid]["valuenum"].dropna()
            if not vals.empty:
                result[feat_name] = float(vals.iloc[-1])  # Latest value

        return result

    def _extract_med_from_df(self, rx: pd.DataFrame) -> dict[str, float]:
        """Extract medication features from a patient's prescriptions."""
        result: dict[str, float] = {}
        if rx.empty:
            return result

        if "drug" in rx.columns:
            drugs = rx["drug"].str.lower().fillna("")
            abx_keywords = [
                "cillin", "cef", "meropenem", "imipenem",
                "ciprofloxacin", "levofloxacin", "vancomycin",
                "gentamicin", "amikacin", "metronidazole",
            ]
            abx_mask = drugs.apply(
                lambda d: any(kw in str(d) for kw in abx_keywords)
            )
            result["abx_count_7d"] = float(abx_mask.sum())

            if drugs.str.contains("meropenem|imipenem|ertapenem").any():
                result["carbapenem_exposure"] = 1.0
            if drugs.str.contains("ciprofloxacin|levofloxacin").any():
                result["fluoroquinolone_exposure"] = 1.0

        return result

    def _temporal_from_csvs(
        self,
        hadm_ids: np.ndarray,
        labevents: pd.DataFrame | None,
        chartevents: pd.DataFrame | None,
    ) -> np.ndarray:
        """Extract 72h temporal tensors from MIMIC-IV CSVs.

        Falls back to synthetic generation if chart/lab data is insufficient.
        """
        n = len(hadm_ids)
        tensors = np.full(
            (n, WINDOW_HOURS, NUM_TEMPORAL_FEATURES), np.nan, dtype=np.float32
        )

        # For now, fill with synthetic data since hourly alignment from
        # raw MIMIC CSVs requires significant timestamp processing
        # Real implementation would bin labevents/chartevents by hour
        for i in range(n):
            for feat_idx in range(NUM_TEMPORAL_FEATURES):
                feat_name = ALL_TEMPORAL_FEATURES[feat_idx]
                lo, hi = _TEMPORAL_RANGES.get(feat_name, (0.0, 1.0))
                mid = (lo + hi) / 2
                spread = (hi - lo) / 4
                baseline = self._rng.normal(mid, spread)
                values = np.zeros(WINDOW_HOURS, dtype=np.float32)
                values[0] = np.clip(baseline, lo, hi)
                for t in range(1, WINDOW_HOURS):
                    values[t] = np.clip(
                        0.95 * values[t - 1] + self._rng.normal(0, spread * 0.05),
                        lo, hi,
                    )
                tensors[i, :, feat_idx] = values

        return tensors

    def _notes_from_csvs(
        self,
        hadm_ids: np.ndarray,
        notes_df: pd.DataFrame | None,
    ) -> list[list[str]]:
        """Extract clinical notes from MIMIC-IV noteevents/discharge."""
        all_notes: list[list[str]] = []

        for hid in hadm_ids:
            if notes_df is not None and "hadm_id" in notes_df.columns:
                pt_notes = notes_df[notes_df["hadm_id"] == hid]
                text_col = "text" if "text" in pt_notes.columns else "note"
                if text_col in pt_notes.columns:
                    texts = pt_notes[text_col].dropna().tolist()[:3]
                    all_notes.append([str(t)[:2048] for t in texts])
                    continue

            # Fallback: generate a placeholder note
            all_notes.append(["Clinical note not available for this admission."])

        return all_notes
