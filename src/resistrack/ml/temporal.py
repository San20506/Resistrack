"""M2.2 Temporal Feature Extractor.

Extracts 72-hour rolling window temporal features from lab values and vital signs.
Output tensor shape: (batch_size, 72, 13) — 8 lab values + 5 vital signs per hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from resistrack.common.constants import RANDOM_STATE

WINDOW_HOURS: Final[int] = 72
NUM_LAB_FEATURES: Final[int] = 8
NUM_VITAL_FEATURES: Final[int] = 5
NUM_TEMPORAL_FEATURES: Final[int] = NUM_LAB_FEATURES + NUM_VITAL_FEATURES  # 13

LAB_FEATURE_NAMES: Final[list[str]] = [
    "wbc_count",
    "crp_level",
    "procalcitonin",
    "lactate",
    "creatinine",
    "platelet_count",
    "neutrophil_pct",
    "band_neutrophils",
]

VITAL_FEATURE_NAMES: Final[list[str]] = [
    "temperature",
    "heart_rate",
    "respiratory_rate",
    "systolic_bp",
    "oxygen_saturation",
]

ALL_TEMPORAL_FEATURES: Final[list[str]] = LAB_FEATURE_NAMES + VITAL_FEATURE_NAMES

MISSING_THRESHOLD: Final[float] = 0.30


@dataclass
class CohortStats:
    """Z-score normalization statistics per cohort."""

    means: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(NUM_TEMPORAL_FEATURES, dtype=np.float64)
    )
    stds: NDArray[np.float64] = field(
        default_factory=lambda: np.ones(NUM_TEMPORAL_FEATURES, dtype=np.float64)
    )


@dataclass
class TemporalExtractionResult:
    """Result of temporal feature extraction for a single patient."""

    tensor: NDArray[np.float64]
    missing_mask: NDArray[np.bool_]
    completeness_score: float
    feature_names: list[str] = field(default_factory=lambda: list(ALL_TEMPORAL_FEATURES))


class TemporalFeatureExtractor:
    """Extracts 72h rolling window temporal features from clinical time-series data.

    Produces a tensor of shape (72, 13) per patient:
    - 8 lab values: WBC, CRP, procalcitonin, lactate, creatinine, platelets, neutrophil%, bands
    - 5 vital signs: temperature, heart rate, respiratory rate, systolic BP, SpO2

    Missing values are forward-filled then zero-filled.
    Z-score normalization uses cohort-specific statistics.
    """

    def __init__(self, cohort_stats: CohortStats | None = None) -> None:
        self._cohort_stats = cohort_stats or CohortStats()
        self._rng = np.random.default_rng(RANDOM_STATE)

    def extract(
        self, hourly_data: NDArray[np.float64] | None
    ) -> TemporalExtractionResult:
        """Extract temporal features from hourly clinical data.

        Args:
            hourly_data: Array of shape (hours, 13) with NaN for missing values.
                         If None, returns an all-zero tensor with 0.0 completeness.

        Returns:
            TemporalExtractionResult with normalized tensor and completeness score.
        """
        if hourly_data is None:
            tensor = np.zeros((WINDOW_HOURS, NUM_TEMPORAL_FEATURES), dtype=np.float64)
            mask = np.ones((WINDOW_HOURS, NUM_TEMPORAL_FEATURES), dtype=np.bool_)
            return TemporalExtractionResult(
                tensor=tensor, missing_mask=mask, completeness_score=0.0
            )

        padded = self._pad_or_truncate(hourly_data)
        missing_mask = np.isnan(padded)
        completeness = 1.0 - float(np.mean(missing_mask))

        filled = self._forward_fill(padded)
        filled = np.nan_to_num(filled, nan=0.0)

        normalized = self._zscore_normalize(filled)

        return TemporalExtractionResult(
            tensor=normalized,
            missing_mask=missing_mask,
            completeness_score=round(completeness, 4),
        )

    def extract_batch(
        self, batch_data: list[NDArray[np.float64] | None]
    ) -> list[TemporalExtractionResult]:
        """Extract temporal features for a batch of patients."""
        return [self.extract(data) for data in batch_data]

    def extract_batch_tensor(
        self, batch_data: list[NDArray[np.float64] | None]
    ) -> NDArray[np.float64]:
        """Extract and stack into a single (batch_size, 72, 13) tensor."""
        results = self.extract_batch(batch_data)
        return np.stack([r.tensor for r in results], axis=0)

    def compute_cohort_stats(
        self, all_data: list[NDArray[np.float64]]
    ) -> CohortStats:
        """Compute cohort-level mean/std for Z-score normalization."""
        valid_arrays = [d for d in all_data if d is not None and d.size > 0]
        if not valid_arrays:
            return CohortStats()

        combined = np.concatenate(valid_arrays, axis=0)
        means = np.nanmean(combined, axis=0)
        stds = np.nanstd(combined, axis=0)
        stds[stds < 1e-8] = 1.0

        stats = CohortStats(means=means, stds=stds)
        self._cohort_stats = stats
        return stats

    def _pad_or_truncate(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Ensure data has exactly WINDOW_HOURS rows."""
        rows = data.shape[0]
        cols = min(data.shape[1], NUM_TEMPORAL_FEATURES) if data.ndim > 1 else NUM_TEMPORAL_FEATURES

        result = np.full(
            (WINDOW_HOURS, NUM_TEMPORAL_FEATURES), np.nan, dtype=np.float64
        )

        if data.ndim == 1:
            return result

        take_rows = min(rows, WINDOW_HOURS)
        start_row = max(0, rows - WINDOW_HOURS)
        result[-take_rows:, :cols] = data[start_row : start_row + take_rows, :cols]
        return result

    def _forward_fill(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward-fill NaN values along the time axis."""
        result = data.copy()
        for col in range(result.shape[1]):
            last_valid = np.nan
            for row in range(result.shape[0]):
                if np.isnan(result[row, col]):
                    if not np.isnan(last_valid):
                        result[row, col] = last_valid
                else:
                    last_valid = result[row, col]
        return result

    def _zscore_normalize(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Z-score normalize using cohort statistics."""
        return (data - self._cohort_stats.means) / self._cohort_stats.stds

    @staticmethod
    def has_sufficient_data(completeness_score: float) -> bool:
        """Check if data completeness is above the minimum threshold."""
        return completeness_score >= (1.0 - MISSING_THRESHOLD)
