"""PSI-based drift monitoring for feature distributions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from resistrack.common.constants import RANDOM_STATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PSI_THRESHOLD = 0.20
_MONITORING_WINDOW_DAYS = 30
_MIN_SAMPLES = 100
_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftConfig:
    psi_threshold: float = _PSI_THRESHOLD
    monitoring_window_days: int = _MONITORING_WINDOW_DAYS
    min_samples: int = _MIN_SAMPLES


@dataclass(frozen=True)
class DriftResult:
    feature_name: str
    psi_value: float
    is_drifted: bool
    baseline_distribution: np.ndarray
    current_distribution: np.ndarray
    timestamp: float

    class Config:
        arbitrary_types_allowed = True


@dataclass(frozen=True)
class DriftReport:
    results: list[DriftResult]
    overall_drift_detected: bool
    emergency_retrain_triggered: bool
    report_timestamp: float


# ---------------------------------------------------------------------------
# PSI Drift Monitor
# ---------------------------------------------------------------------------

class PSIDriftMonitor:

    def __init__(self, config: DriftConfig | None = None) -> None:
        self._config = config or DriftConfig()
        self._reports: list[DriftReport] = []
        self._rng = np.random.RandomState(RANDOM_STATE)

    @staticmethod
    def compute_psi(baseline: np.ndarray, current: np.ndarray,
                    bins: int = 10) -> float:
        """Population Stability Index between two distributions.

        PSI = Σ (p_i - q_i) * ln(p_i / q_i)
        where p = current proportions, q = baseline proportions.
        """
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        breakpoints = np.linspace(min_val, max_val, bins + 1)

        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]

        baseline_pct = baseline_counts / max(len(baseline), 1) + _EPSILON
        current_pct = current_counts / max(len(current), 1) + _EPSILON

        psi = np.sum(
            (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
        )
        return float(psi)

    def monitor_features(
        self,
        baseline_data: dict[str, np.ndarray],
        current_data: dict[str, np.ndarray],
    ) -> DriftReport:
        now = time.time()
        results: list[DriftResult] = []

        for feature_name in baseline_data:
            if feature_name not in current_data:
                continue

            baseline = baseline_data[feature_name]
            current = current_data[feature_name]

            if len(baseline) < self._config.min_samples:
                continue
            if len(current) < self._config.min_samples:
                continue

            psi_value = self.compute_psi(baseline, current)
            is_drifted = psi_value > self._config.psi_threshold

            results.append(DriftResult(
                feature_name=feature_name,
                psi_value=psi_value,
                is_drifted=is_drifted,
                baseline_distribution=baseline,
                current_distribution=current,
                timestamp=now,
            ))

        overall_drift = any(r.is_drifted for r in results)
        emergency = overall_drift

        report = DriftReport(
            results=results,
            overall_drift_detected=overall_drift,
            emergency_retrain_triggered=emergency,
            report_timestamp=now,
        )
        self._reports.append(report)
        return report

    def should_trigger_retrain(self, report: DriftReport) -> bool:
        return report.emergency_retrain_triggered

    def get_monitoring_summary(self) -> dict:
        return {
            "total_reports": len(self._reports),
            "drift_detected_count": sum(
                1 for r in self._reports if r.overall_drift_detected
            ),
            "emergency_retrain_count": sum(
                1 for r in self._reports if r.emergency_retrain_triggered
            ),
            "config": {
                "psi_threshold": self._config.psi_threshold,
                "monitoring_window_days": self._config.monitoring_window_days,
                "min_samples": self._config.min_samples,
            },
        }
