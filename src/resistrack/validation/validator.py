"""Clinical validation framework for AMR prediction model.

Validates ensemble model on held-out records with 6 performance metrics,
subgroup fairness analysis, and latency benchmarks per M5.1 spec.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Sequence

import numpy as np

from resistrack.common.constants import RANDOM_STATE, RiskTier


# ── Thresholds (from Agent Rules) ──────────────────────────────────────────

METRIC_THRESHOLDS = {
    "auc_roc": 0.82,
    "auprc": 0.70,
    "sensitivity_at_80_spec": 0.80,
    "fpr": 0.20,
    "brier": 0.15,
    "p95_latency_ms": 2000.0,
}

AUC_GAP_THRESHOLD = 0.10  # Flag subgroup if AUC gap >= 10%

SUBGROUP_AGE_BINS = [
    ("pediatric", 0, 18),
    ("adult", 18, 65),
    ("geriatric", 65, 200),
]


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    """Configuration for clinical validation run."""

    min_records: int = 1000
    specificity_target: float = 0.80
    n_bootstrap: int = 100
    random_state: int = RANDOM_STATE


@dataclasses.dataclass(frozen=True)
class ValidationMetrics:
    """Six required performance metrics."""

    auc_roc: float
    auprc: float
    sensitivity_at_80_spec: float
    fpr: float
    brier: float
    p95_latency_ms: float

    def passes_thresholds(self) -> dict[str, bool]:
        """Check each metric against its threshold."""
        return {
            "auc_roc": self.auc_roc >= METRIC_THRESHOLDS["auc_roc"],
            "auprc": self.auprc >= METRIC_THRESHOLDS["auprc"],
            "sensitivity_at_80_spec": (
                self.sensitivity_at_80_spec
                >= METRIC_THRESHOLDS["sensitivity_at_80_spec"]
            ),
            "fpr": self.fpr <= METRIC_THRESHOLDS["fpr"],
            "brier": self.brier <= METRIC_THRESHOLDS["brier"],
            "p95_latency_ms": (
                self.p95_latency_ms <= METRIC_THRESHOLDS["p95_latency_ms"]
            ),
        }

    @property
    def all_pass(self) -> bool:
        return all(self.passes_thresholds().values())


@dataclasses.dataclass(frozen=True)
class SubgroupResult:
    """Validation result for a single subgroup."""

    name: str
    n_samples: int
    auc_roc: float
    auc_gap: float  # gap vs overall
    flagged: bool  # True if gap >= 10%


@dataclasses.dataclass(frozen=True)
class SubgroupAnalysis:
    """Aggregated subgroup fairness analysis."""

    subgroups: list[SubgroupResult]

    @property
    def any_flagged(self) -> bool:
        return any(s.flagged for s in self.subgroups)

    @property
    def flagged_subgroups(self) -> list[SubgroupResult]:
        return [s for s in self.subgroups if s.flagged]


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """Complete clinical validation report."""

    metrics: ValidationMetrics
    subgroup_analysis: SubgroupAnalysis
    n_records: int
    model_version: str
    overall_pass: bool
    ci_lower: float  # 95% CI lower bound for AUC-ROC
    ci_upper: float  # 95% CI upper bound for AUC-ROC


# ── Helper Functions ───────────────────────────────────────────────────────

def _compute_auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC via trapezoidal integration."""
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos

    tpr_points: list[float] = [0.0]
    fpr_points: list[float] = [0.0]
    tp = 0.0
    fp = 0.0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_pos)
        fpr_points.append(fp / n_neg)

    auc = 0.0
    for i in range(1, len(fpr_points)):
        auc += (fpr_points[i] - fpr_points[i - 1]) * (
            tpr_points[i] + tpr_points[i - 1]
        ) / 2.0
    return float(auc)


def _compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under precision-recall curve."""
    if len(labels) == 0 or labels.sum() == 0:
        return 0.0

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tp = 0.0
    fp = 0.0
    precisions: list[float] = [1.0]
    recalls: list[float] = [0.0]
    n_pos = sorted_labels.sum()

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_pos
        precisions.append(precision)
        recalls.append(recall)

    auprc = 0.0
    for i in range(1, len(recalls)):
        auprc += (recalls[i] - recalls[i - 1]) * precisions[i]
    return float(auprc)


def _sensitivity_at_specificity(
    labels: np.ndarray,
    scores: np.ndarray,
    target_specificity: float,
) -> float:
    """Find sensitivity at a given specificity target."""
    if len(labels) == 0:
        return 0.0

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    thresholds = np.sort(np.unique(scores))[::-1]
    best_sensitivity = 0.0

    for thresh in thresholds:
        preds = (scores >= thresh).astype(float)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        spec = tn / n_neg if n_neg > 0 else 0.0
        sens = tp / n_pos if n_pos > 0 else 0.0

        if spec >= target_specificity:
            best_sensitivity = max(best_sensitivity, sens)

    return float(best_sensitivity)


def _compute_fpr_at_threshold(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5,
) -> float:
    """False positive rate at default threshold."""
    n_neg = (labels == 0).sum()
    if n_neg == 0:
        return 0.0
    preds = (scores >= threshold).astype(float)
    fp = ((preds == 1) & (labels == 0)).sum()
    return float(fp / n_neg)


def _compute_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    """Brier score (mean squared error of predicted probabilities)."""
    return float(np.mean((probs - labels) ** 2))


# ── Main Validator ─────────────────────────────────────────────────────────

class ClinicalValidator:
    """Validates AMR prediction model on held-out clinical data.

    Parameters
    ----------
    config : ValidationConfig
        Validation configuration with thresholds and settings.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self._config = config or ValidationConfig()

    @property
    def config(self) -> ValidationConfig:
        return self._config

    def validate(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        ages: np.ndarray | None = None,
        icu_flags: np.ndarray | None = None,
        latencies_ms: np.ndarray | None = None,
        model_version: str = "1.0.0",
    ) -> ValidationReport:
        """Run full clinical validation.

        Parameters
        ----------
        labels : ndarray
            Ground truth binary labels (0/1), shape (n_records,).
        scores : ndarray
            Predicted probabilities, shape (n_records,).
        ages : ndarray, optional
            Patient ages for subgroup analysis.
        icu_flags : ndarray, optional
            ICU admission flags (0/1) for subgroup analysis.
        latencies_ms : ndarray, optional
            Per-prediction latency measurements in milliseconds.
        model_version : str
            Model version string for the report.

        Returns
        -------
        ValidationReport
            Complete validation report with metrics, subgroups, and CI.

        Raises
        ------
        ValueError
            If record count < min_records or array shapes mismatch.
        """
        n = len(labels)
        if n < self._config.min_records:
            msg = (
                f"Need >= {self._config.min_records} records, got {n}"
            )
            raise ValueError(msg)

        if len(scores) != n:
            raise ValueError("labels and scores must have same length")

        metrics = self._compute_metrics(labels, scores, latencies_ms)
        subgroups = self._analyze_subgroups(labels, scores, ages, icu_flags)
        ci_lower, ci_upper = self._bootstrap_ci(labels, scores)
        overall_pass = metrics.all_pass and not subgroups.any_flagged

        return ValidationReport(
            metrics=metrics,
            subgroup_analysis=subgroups,
            n_records=n,
            model_version=model_version,
            overall_pass=overall_pass,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def _compute_metrics(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        latencies_ms: np.ndarray | None,
    ) -> ValidationMetrics:
        """Compute the 6 required performance metrics."""
        auc_roc = _compute_auc_roc(labels, scores)
        auprc = _compute_auprc(labels, scores)
        sens = _sensitivity_at_specificity(
            labels, scores, self._config.specificity_target,
        )
        fpr = _compute_fpr_at_threshold(labels, scores)
        brier = _compute_brier(labels, scores)

        if latencies_ms is not None and len(latencies_ms) > 0:
            p95 = float(np.percentile(latencies_ms, 95))
        else:
            p95 = 0.0

        return ValidationMetrics(
            auc_roc=auc_roc,
            auprc=auprc,
            sensitivity_at_80_spec=sens,
            fpr=fpr,
            brier=brier,
            p95_latency_ms=p95,
        )

    def _analyze_subgroups(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        ages: np.ndarray | None,
        icu_flags: np.ndarray | None,
    ) -> SubgroupAnalysis:
        """Perform subgroup fairness analysis."""
        overall_auc = _compute_auc_roc(labels, scores)
        results: list[SubgroupResult] = []

        if ages is not None:
            for name, low, high in SUBGROUP_AGE_BINS:
                mask = (ages >= low) & (ages < high)
                if mask.sum() >= 10:
                    sub_auc = _compute_auc_roc(labels[mask], scores[mask])
                    gap = abs(overall_auc - sub_auc)
                    results.append(SubgroupResult(
                        name=f"age_{name}",
                        n_samples=int(mask.sum()),
                        auc_roc=sub_auc,
                        auc_gap=gap,
                        flagged=gap >= AUC_GAP_THRESHOLD,
                    ))

        if icu_flags is not None:
            for flag_val, name in [(0, "non_icu"), (1, "icu")]:
                mask = icu_flags == flag_val
                if mask.sum() >= 10:
                    sub_auc = _compute_auc_roc(labels[mask], scores[mask])
                    gap = abs(overall_auc - sub_auc)
                    results.append(SubgroupResult(
                        name=name,
                        n_samples=int(mask.sum()),
                        auc_roc=sub_auc,
                        auc_gap=gap,
                        flagged=gap >= AUC_GAP_THRESHOLD,
                    ))

        return SubgroupAnalysis(subgroups=results)

    def _bootstrap_ci(
        self, labels: np.ndarray, scores: np.ndarray,
    ) -> tuple[float, float]:
        """95% bootstrap confidence interval for AUC-ROC."""
        rng = np.random.RandomState(self._config.random_state)
        n = len(labels)
        aucs: list[float] = []

        for _ in range(self._config.n_bootstrap):
            idx = rng.randint(0, n, n)
            boot_labels = labels[idx]
            boot_scores = scores[idx]
            if boot_labels.sum() > 0 and boot_labels.sum() < len(boot_labels):
                aucs.append(_compute_auc_roc(boot_labels, boot_scores))

        if len(aucs) < 2:
            base_auc = _compute_auc_roc(labels, scores)
            return base_auc, base_auc

        return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
