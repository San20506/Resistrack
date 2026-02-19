"""M2.6 — Ensemble Combiner, Platt Calibration & SHAP Explanation.

Combines XGBoost, LSTM and ClinicalBERT sub-model predictions via a
learned meta-learner (logistic regression on validation-set outputs).
Weights are **never hardcoded** (RULE-TRAIN-06).

After combining, Platt scaling (sigmoid calibration) maps the raw
ensemble score to a calibrated probability (target Brier ≤ 0.15).

SHAP-style feature attribution uses a linear-approximation method to
produce the top-5 clinical feature explanations for each prediction.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from resistrack.common.constants import (
    ANTIBIOTIC_CLASSES,
    CONFIDENCE_THRESHOLD,
    RANDOM_STATE,
    RISK_TIER_RANGES,
    RiskTier,
)
from resistrack.common.schemas import (
    AMRPredictionOutput,
    AntibioticClassRisk,
    SHAPFeature,
)

# ---------------------------------------------------------------------------
# Clinical feature name mapping (index → human-readable)
# ---------------------------------------------------------------------------
CLINICAL_FEATURE_NAMES: dict[int, tuple[str, str]] = {
    # lab features (0-7)
    0: ("wbc_count", "White Blood Cell Count"),
    1: ("crp_level", "C-Reactive Protein Level"),
    2: ("procalcitonin", "Procalcitonin Level"),
    3: ("lactate", "Blood Lactate Level"),
    4: ("creatinine", "Serum Creatinine"),
    5: ("platelet_count", "Platelet Count"),
    6: ("neutrophil_pct", "Neutrophil Percentage"),
    7: ("albumin", "Serum Albumin Level"),
    # vital features (8-12)
    8: ("temperature", "Body Temperature"),
    9: ("heart_rate", "Heart Rate"),
    10: ("resp_rate", "Respiratory Rate"),
    11: ("systolic_bp", "Systolic Blood Pressure"),
    12: ("o2_saturation", "Oxygen Saturation"),
    # derived / NLP features (13+)
    13: ("prior_abx_exposure", "Prior Antibiotic Exposure (days)"),
    14: ("icu_stay_hours", "ICU Length of Stay (hours)"),
    15: ("culture_positive", "Positive Culture Result"),
    16: ("note_amr_signal", "Clinical Notes AMR Signal (NLP)"),
    17: ("age_years", "Patient Age"),
    18: ("comorbidity_index", "Comorbidity Burden Index"),
    19: ("prior_resistance", "Prior Resistance History"),
}


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubModelOutput:
    """Raw output vector from a single sub-model."""

    model_name: str
    scores: NDArray[np.floating[Any]]  # shape (n_samples,)


@dataclass
class MetaLearnerState:
    """Serialisable state of the trained meta-learner."""

    weights: NDArray[np.floating[Any]]  # shape (n_models,)
    bias: float = 0.0
    platt_a: float = -1.0
    platt_b: float = 0.0
    is_fitted: bool = False


@dataclass(frozen=True)
class CalibrationResult:
    """Result of Platt scaling calibration."""

    platt_a: float
    platt_b: float
    brier_score: float
    n_samples: int


@dataclass(frozen=True)
class EnsemblePrediction:
    """Single-patient ensemble prediction with explanation."""

    raw_score: float
    calibrated_probability: float
    risk_score: int  # 0-100
    risk_tier: RiskTier
    confidence: float
    low_confidence_flag: bool
    shap_features: list[SHAPFeature]
    antibiotic_class_risk: AntibioticClassRisk
    latency_ms: float


@dataclass(frozen=True)
class EnsembleTrainingResult:
    """Result from meta-learner training."""

    meta_weights: NDArray[np.floating[Any]]
    meta_bias: float
    calibration: CalibrationResult
    val_auc_roc: float
    n_train: int
    n_val: int


# ---------------------------------------------------------------------------
# Helper: sigmoid
# ---------------------------------------------------------------------------
def _sigmoid(x: NDArray[np.floating[Any]] | float) -> NDArray[np.floating[Any]] | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=np.float64)
    result = np.where(
        x_arr >= 0,
        1.0 / (1.0 + np.exp(-x_arr)),
        np.exp(x_arr) / (1.0 + np.exp(x_arr)),
    )
    if np.ndim(x) == 0:
        return float(result)
    return result  # type: ignore[return-value]


def _score_to_risk_tier(score: int) -> RiskTier:
    """Map integer risk score 0-100 to a RiskTier."""
    score = max(0, min(score, 100))
    for tier, (lo, hi) in RISK_TIER_RANGES.items():
        if lo <= score <= hi:
            return tier
    return RiskTier.CRITICAL  # pragma: no cover


def _compute_auc(
    labels: NDArray[np.floating[Any]], scores: NDArray[np.floating[Any]],
) -> float:
    """Simple AUC-ROC via trapezoidal rule."""
    pos = labels == 1
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal integration
    auc = 0.0
    for j in range(1, len(fpr_list)):
        auc += (fpr_list[j] - fpr_list[j - 1]) * (tpr_list[j] + tpr_list[j - 1]) / 2
    return float(np.clip(auc, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Meta-Learner (learned weights via logistic regression)
# ---------------------------------------------------------------------------
class MetaLearner:
    """Logistic-regression meta-learner over sub-model outputs.

    Weights are learned from validation-set predictions, never hardcoded
    (RULE-TRAIN-06).
    """

    def __init__(self, *, learning_rate: float = 0.05, n_iterations: int = 500) -> None:
        self._lr = learning_rate
        self._n_iter = n_iterations
        self._rng = np.random.RandomState(RANDOM_STATE)
        self._state = MetaLearnerState(weights=np.array([]), bias=0.0)

    @property
    def is_fitted(self) -> bool:
        return self._state.is_fitted

    @property
    def weights(self) -> NDArray[np.floating[Any]]:
        if not self._state.is_fitted:
            raise RuntimeError("MetaLearner not fitted")
        return self._state.weights.copy()

    @property
    def bias(self) -> float:
        if not self._state.is_fitted:
            raise RuntimeError("MetaLearner not fitted")
        return self._state.bias

    def fit(
        self,
        sub_model_scores: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]],
    ) -> None:
        """Train via gradient descent on binary cross-entropy.

        Parameters
        ----------
        sub_model_scores : ndarray of shape (n_samples, n_models)
        labels : ndarray of shape (n_samples,) values in {0, 1}
        """
        n_samples, n_models = sub_model_scores.shape
        if n_samples != labels.shape[0]:
            raise ValueError("Score/label length mismatch")
        if n_samples == 0:
            raise ValueError("Cannot fit on empty data")

        w = self._rng.randn(n_models).astype(np.float64) * 0.01
        b = 0.0

        for _ in range(self._n_iter):
            logits = sub_model_scores @ w + b
            preds = _sigmoid(logits)
            err = preds - labels  # type: ignore[operator]
            grad_w = (sub_model_scores.T @ err) / n_samples
            grad_b = float(np.mean(err))
            w -= self._lr * grad_w
            b -= self._lr * grad_b

        self._state = MetaLearnerState(weights=w, bias=b, is_fitted=True)

    def predict_raw(
        self,
        sub_model_scores: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Return raw logits (before Platt scaling)."""
        if not self._state.is_fitted:
            raise RuntimeError("MetaLearner not fitted")
        return sub_model_scores @ self._state.weights + self._state.bias

    def get_state(self) -> MetaLearnerState:
        return MetaLearnerState(
            weights=self._state.weights.copy(),
            bias=self._state.bias,
            platt_a=self._state.platt_a,
            platt_b=self._state.platt_b,
            is_fitted=self._state.is_fitted,
        )

    def load_state(self, state: MetaLearnerState) -> None:
        self._state = MetaLearnerState(
            weights=state.weights.copy(),
            bias=state.bias,
            platt_a=state.platt_a,
            platt_b=state.platt_b,
            is_fitted=state.is_fitted,
        )


# ---------------------------------------------------------------------------
# Platt Scaling (calibration)
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """Platt sigmoid calibration: P(y=1|f) = 1/(1+exp(A*f+B))."""

    def __init__(self, *, n_iterations: int = 200, learning_rate: float = 0.01) -> None:
        self._n_iter = n_iterations
        self._lr = learning_rate
        self._a: float = 1.0
        self._b: float = 0.0
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def params(self) -> tuple[float, float]:
        return (self._a, self._b)

    def fit(
        self,
        raw_scores: NDArray[np.floating[Any]],
        labels: NDArray[np.floating[Any]],
    ) -> CalibrationResult:
        """Fit Platt A, B via gradient descent on log-loss."""
        n = len(raw_scores)
        if n == 0:
            raise ValueError("Cannot calibrate on empty data")

        a, b = 1.0, 0.0
        for _ in range(self._n_iter):
            logits = a * raw_scores + b
            preds = _sigmoid(logits)
            err = preds - labels  # type: ignore[operator]
            grad_a = float(np.mean(err * raw_scores))
            grad_b = float(np.mean(err))
            a -= self._lr * grad_a
            b -= self._lr * grad_b

        self._a = a
        self._b = b
        self._is_fitted = True

        # Compute Brier score
        cal_probs = self.calibrate(raw_scores)
        brier = float(np.mean((cal_probs - labels) ** 2))

        return CalibrationResult(
            platt_a=a, platt_b=b, brier_score=brier, n_samples=n,
        )

    def calibrate(
        self, raw_scores: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Apply fitted Platt scaling."""
        if not self._is_fitted:
            raise RuntimeError("PlattCalibrator not fitted")
        return _sigmoid(self._a * raw_scores + self._b)  # type: ignore[return-value]

    def load_params(self, a: float, b: float) -> None:
        self._a = a
        self._b = b
        self._is_fitted = True


# ---------------------------------------------------------------------------
# SHAP-like Feature Attributor (linear approximation)
# ---------------------------------------------------------------------------
class FeatureAttributor:
    """Approximate SHAP via linear decomposition.

    Uses the meta-learner weights to propagate importance back through
    sub-model feature importances to produce patient-level attributions.
    """

    def __init__(
        self,
        feature_names: dict[int, tuple[str, str]] | None = None,
        top_k: int = 5,
    ) -> None:
        self._feature_names = feature_names or CLINICAL_FEATURE_NAMES
        self._top_k = top_k

    def attribute(
        self,
        feature_vector: NDArray[np.floating[Any]],
        meta_weights: NDArray[np.floating[Any]],
        sub_importances: list[NDArray[np.floating[Any]]],
    ) -> list[SHAPFeature]:
        """Compute top-k SHAP-like attributions for a single patient.

        Parameters
        ----------
        feature_vector : shape (n_features,) — patient's raw feature values
        meta_weights : shape (n_models,) — learned ensemble weights
        sub_importances : list of shape (n_features,) per model
        """
        n_features = len(feature_vector)
        # Weighted combination of sub-model importances
        combined = np.zeros(n_features, dtype=np.float64)
        abs_weights = np.abs(meta_weights)
        weight_sum = abs_weights.sum()
        if weight_sum > 0:
            normed = abs_weights / weight_sum
        else:
            normed = np.ones_like(abs_weights) / len(abs_weights)

        for i, imp in enumerate(sub_importances):
            if i < len(normed):
                padded = np.zeros(n_features, dtype=np.float64)
                padded[: len(imp)] = imp[: n_features]
                combined += normed[i] * padded * feature_vector

        # Get top-k by absolute value
        top_indices = np.argsort(np.abs(combined))[::-1][: self._top_k]

        result: list[SHAPFeature] = []
        for idx in top_indices:
            idx_int = int(idx)
            name_code, human = self._feature_names.get(
                idx_int, (f"feature_{idx_int}", f"Feature {idx_int}")
            )
            shap_val = float(combined[idx_int])
            direction = "positive" if shap_val >= 0 else "negative"
            result.append(
                SHAPFeature(
                    name=name_code,
                    value=round(shap_val, 6),
                    direction=direction,
                    human_readable=human,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Antibiotic Risk Estimator
# ---------------------------------------------------------------------------
class AntibioticRiskEstimator:
    """Derive per-class antibiotic resistance probability.

    Uses a deterministic hash of feature vector + class name to produce
    stable per-class risk values, modulated by the overall AMR risk.
    """

    def estimate(
        self,
        overall_risk_score: float,
        feature_vector: NDArray[np.floating[Any]],
    ) -> AntibioticClassRisk:
        """Produce antibiotic class-level risk scores (0-1)."""
        rng = np.random.RandomState(RANDOM_STATE)
        base = overall_risk_score / 100.0
        risks: dict[str, float] = {}
        for cls in ANTIBIOTIC_CLASSES:
            # Deterministic per-class modulation
            seed_bytes = hashlib.sha256(
                f"{cls}:{feature_vector.tobytes().hex()[:32]}".encode()
            ).digest()[:4]
            cls_seed = int.from_bytes(seed_bytes, "big") % 10000
            rng_cls = np.random.RandomState(cls_seed)
            modulation = rng_cls.uniform(0.7, 1.3)
            risk_val = float(np.clip(base * modulation, 0.0, 1.0))
            risks[cls] = round(risk_val, 4)
        return AntibioticClassRisk(**risks)


# ---------------------------------------------------------------------------
# Ensemble Predictor (main orchestrator)
# ---------------------------------------------------------------------------
class EnsemblePredictor:
    """End-to-end ensemble prediction pipeline.

    1. Combine sub-model scores via learned MetaLearner
    2. Calibrate via PlattCalibrator
    3. Map to risk score/tier
    4. Generate SHAP explanations
    5. Estimate antibiotic class risks
    6. Return AMRPredictionOutput
    """

    def __init__(
        self,
        *,
        model_version: str = "1.0.0",
        top_k_shap: int = 5,
    ) -> None:
        self._meta = MetaLearner()
        self._calibrator = PlattCalibrator()
        self._attributor = FeatureAttributor(top_k=top_k_shap)
        self._abx_estimator = AntibioticRiskEstimator()
        self._model_version = model_version
        self._sub_importances: list[NDArray[np.floating[Any]]] = []

    @property
    def is_fitted(self) -> bool:
        return self._meta.is_fitted and self._calibrator.is_fitted

    @property
    def model_version(self) -> str:
        return self._model_version

    def train(
        self,
        sub_model_outputs: list[SubModelOutput],
        labels: NDArray[np.floating[Any]],
        feature_importances: list[NDArray[np.floating[Any]]] | None = None,
        val_fraction: float = 0.3,
    ) -> EnsembleTrainingResult:
        """Train meta-learner and calibrator on sub-model predictions.

        Parameters
        ----------
        sub_model_outputs : list of SubModelOutput, one per sub-model
        labels : ndarray shape (n_samples,) in {0, 1}
        feature_importances : optional per-model feature importance vectors
        val_fraction : holdout fraction for calibration
        """
        n_models = len(sub_model_outputs)
        n_samples = len(labels)
        if n_models == 0:
            raise ValueError("No sub-model outputs provided")
        if n_samples == 0:
            raise ValueError("No samples provided")

        # Stack scores: (n_samples, n_models)
        scores = np.column_stack([o.scores for o in sub_model_outputs])

        # Split train/val for calibration
        rng = np.random.RandomState(RANDOM_STATE)
        indices = rng.permutation(n_samples)
        n_val = max(1, int(n_samples * val_fraction))
        n_train = n_samples - n_val
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        # 1. Fit meta-learner on train split
        self._meta.fit(scores[train_idx], labels[train_idx])

        # 2. Calibrate on val split
        raw_val = self._meta.predict_raw(scores[val_idx])
        cal_result = self._calibrator.fit(raw_val, labels[val_idx])

        # Store feature importances for SHAP
        if feature_importances is not None:
            self._sub_importances = [imp.copy() for imp in feature_importances]
        else:
            self._sub_importances = [np.ones(20) / 20 for _ in range(n_models)]

        # Compute val AUC-ROC (simple trapezoid)
        val_probs = self._calibrator.calibrate(raw_val)
        val_auc = _compute_auc(labels[val_idx], val_probs)

        # Sync Platt params back to meta state
        state = self._meta.get_state()
        updated = MetaLearnerState(
            weights=state.weights,
            bias=state.bias,
            platt_a=self._calibrator.params[0],
            platt_b=self._calibrator.params[1],
            is_fitted=True,
        )
        self._meta.load_state(updated)

        return EnsembleTrainingResult(
            meta_weights=self._meta.weights,
            meta_bias=self._meta.bias,
            calibration=cal_result,
            val_auc_roc=val_auc,
            n_train=n_train,
            n_val=n_val,
        )

    def predict(
        self,
        sub_model_scores: NDArray[np.floating[Any]],
        feature_vectors: NDArray[np.floating[Any]],
        patient_tokens: Sequence[str],
        *,
        data_completeness_scores: Sequence[float] | None = None,
    ) -> list[EnsemblePrediction]:
        """Run full ensemble prediction for a batch of patients.

        Parameters
        ----------
        sub_model_scores : shape (n_patients, n_models)
        feature_vectors : shape (n_patients, n_features)
        patient_tokens : de-identified patient identifiers
        data_completeness_scores : optional per-patient completeness (0-1)
        """
        if not self.is_fitted:
            raise RuntimeError("EnsemblePredictor not fitted")

        n_patients = sub_model_scores.shape[0]
        if feature_vectors.shape[0] != n_patients:
            raise ValueError("Score/feature count mismatch")

        raw = self._meta.predict_raw(sub_model_scores)
        calibrated = self._calibrator.calibrate(raw)
        meta_w = self._meta.weights

        predictions: list[EnsemblePrediction] = []
        for i in range(n_patients):
            t_start = time.monotonic()

            prob = float(calibrated[i])
            risk_score = int(np.clip(np.round(prob * 100), 0, 100))
            risk_tier = _score_to_risk_tier(risk_score)

            # Confidence from calibration certainty (distance from 0.5)
            confidence = round(float(2.0 * abs(prob - 0.5)), 4)
            confidence = min(confidence, 1.0)
            low_conf = confidence < CONFIDENCE_THRESHOLD

            # SHAP features
            fv = feature_vectors[i]
            shap_features = self._attributor.attribute(
                fv, meta_w, self._sub_importances,
            )

            # Antibiotic risk
            abx_risk = self._abx_estimator.estimate(risk_score, fv)

            latency_ms = (time.monotonic() - t_start) * 1000.0

            predictions.append(
                EnsemblePrediction(
                    raw_score=float(raw[i]),
                    calibrated_probability=prob,
                    risk_score=risk_score,
                    risk_tier=risk_tier,
                    confidence=confidence,
                    low_confidence_flag=low_conf,
                    shap_features=shap_features,
                    antibiotic_class_risk=abx_risk,
                    latency_ms=latency_ms,
                )
            )

        return predictions

    def predict_single(
        self,
        sub_model_scores: NDArray[np.floating[Any]],
        feature_vector: NDArray[np.floating[Any]],
        patient_token: str,
        *,
        data_completeness: float = 1.0,
    ) -> AMRPredictionOutput:
        """Convenience method: predict for one patient → AMRPredictionOutput."""
        preds = self.predict(
            sub_model_scores.reshape(1, -1),
            feature_vector.reshape(1, -1),
            [patient_token],
            data_completeness_scores=[data_completeness],
        )
        p = preds[0]

        # Determine recommended action
        if p.risk_tier == RiskTier.CRITICAL:
            action = "IMMEDIATE: Initiate broad-spectrum empiric therapy and order cultures"
        elif p.risk_tier == RiskTier.HIGH:
            action = "URGENT: Review antibiogram and consider targeted therapy adjustment"
        elif p.risk_tier == RiskTier.MEDIUM:
            action = "MONITOR: Continue current therapy with close monitoring"
        else:
            action = "ROUTINE: Standard antimicrobial stewardship protocols"

        data_quality = data_completeness >= 0.70

        return AMRPredictionOutput(
            patient_token=patient_token,
            amr_risk_score=p.risk_score,
            risk_tier=p.risk_tier.value,
            confidence_score=p.confidence,
            low_confidence_flag=p.low_confidence_flag,
            data_completeness_score=data_completeness,
            data_quality_flag=data_quality,
            antibiotic_class_risk=p.antibiotic_class_risk,
            shap_top_features=p.shap_features,
            recommended_action=action,
            model_version=self._model_version,
        )

    def get_state(self) -> dict[str, Any]:
        """Serialisable snapshot for deployment."""
        meta_state = self._meta.get_state()
        return {
            "meta_weights": meta_state.weights.tolist(),
            "meta_bias": meta_state.bias,
            "platt_a": meta_state.platt_a,
            "platt_b": meta_state.platt_b,
            "model_version": self._model_version,
            "sub_importances": [imp.tolist() for imp in self._sub_importances],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore from serialised snapshot."""
        meta_state = MetaLearnerState(
            weights=np.array(state["meta_weights"], dtype=np.float64),
            bias=float(state["meta_bias"]),
            platt_a=float(state["platt_a"]),
            platt_b=float(state["platt_b"]),
            is_fitted=True,
        )
        self._meta.load_state(meta_state)
        self._calibrator.load_params(meta_state.platt_a, meta_state.platt_b)
        self._model_version = state.get("model_version", self._model_version)
        if "sub_importances" in state:
            self._sub_importances = [
                np.array(imp, dtype=np.float64) for imp in state["sub_importances"]
            ]

