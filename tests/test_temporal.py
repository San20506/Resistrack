"""Tests for M2.2 Temporal Feature Extractor."""

import numpy as np
import pytest

from resistrack.ml.temporal import (
    ALL_TEMPORAL_FEATURES,
    MISSING_THRESHOLD,
    NUM_TEMPORAL_FEATURES,
    WINDOW_HOURS,
    CohortStats,
    TemporalExtractionResult,
    TemporalFeatureExtractor,
)


@pytest.fixture
def extractor() -> TemporalFeatureExtractor:
    return TemporalFeatureExtractor()


@pytest.fixture
def sample_data() -> np.ndarray:
    """72 hours x 13 features, no missing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((WINDOW_HOURS, NUM_TEMPORAL_FEATURES))


@pytest.fixture
def sparse_data() -> np.ndarray:
    """72 hours x 13 features, 50% missing."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((WINDOW_HOURS, NUM_TEMPORAL_FEATURES))
    mask = rng.random((WINDOW_HOURS, NUM_TEMPORAL_FEATURES)) < 0.5
    data[mask] = np.nan
    return data


class TestTemporalExtractionResult:
    def test_output_shape(self, extractor: TemporalFeatureExtractor, sample_data: np.ndarray) -> None:
        result = extractor.extract(sample_data)
        assert result.tensor.shape == (WINDOW_HOURS, NUM_TEMPORAL_FEATURES)

    def test_missing_mask_shape(self, extractor: TemporalFeatureExtractor, sample_data: np.ndarray) -> None:
        result = extractor.extract(sample_data)
        assert result.missing_mask.shape == (WINDOW_HOURS, NUM_TEMPORAL_FEATURES)

    def test_feature_names(self, extractor: TemporalFeatureExtractor, sample_data: np.ndarray) -> None:
        result = extractor.extract(sample_data)
        assert result.feature_names == list(ALL_TEMPORAL_FEATURES)
        assert len(result.feature_names) == NUM_TEMPORAL_FEATURES


class TestNoneInput:
    def test_none_returns_zeros(self, extractor: TemporalFeatureExtractor) -> None:
        result = extractor.extract(None)
        assert result.tensor.shape == (WINDOW_HOURS, NUM_TEMPORAL_FEATURES)
        np.testing.assert_array_equal(result.tensor, 0.0)

    def test_none_completeness_zero(self, extractor: TemporalFeatureExtractor) -> None:
        result = extractor.extract(None)
        assert result.completeness_score == 0.0

    def test_none_all_missing(self, extractor: TemporalFeatureExtractor) -> None:
        result = extractor.extract(None)
        assert np.all(result.missing_mask)


class TestCompleteness:
    def test_full_data_completeness(self, extractor: TemporalFeatureExtractor, sample_data: np.ndarray) -> None:
        result = extractor.extract(sample_data)
        assert result.completeness_score == 1.0

    def test_sparse_data_completeness(self, extractor: TemporalFeatureExtractor, sparse_data: np.ndarray) -> None:
        result = extractor.extract(sparse_data)
        assert 0.0 < result.completeness_score < 1.0

    def test_missing_mask_matches_nans(self, extractor: TemporalFeatureExtractor, sparse_data: np.ndarray) -> None:
        result = extractor.extract(sparse_data)
        expected_missing_count = np.isnan(sparse_data).sum()
        actual_missing_count = result.missing_mask.sum()
        assert actual_missing_count == expected_missing_count


class TestForwardFill:
    def test_forward_fill_works(self, extractor: TemporalFeatureExtractor) -> None:
        data = np.full((WINDOW_HOURS, NUM_TEMPORAL_FEATURES), np.nan, dtype=np.float64)
        data[0, 0] = 5.0
        data[0, 1] = 3.0
        result = extractor.extract(data)
        # After forward fill and z-score, the values should be consistent
        assert not np.isnan(result.tensor).any()


class TestPadding:
    def test_short_data_padded(self, extractor: TemporalFeatureExtractor) -> None:
        short_data = np.ones((10, NUM_TEMPORAL_FEATURES), dtype=np.float64)
        result = extractor.extract(short_data)
        assert result.tensor.shape == (WINDOW_HOURS, NUM_TEMPORAL_FEATURES)

    def test_long_data_truncated(self, extractor: TemporalFeatureExtractor) -> None:
        long_data = np.ones((100, NUM_TEMPORAL_FEATURES), dtype=np.float64)
        result = extractor.extract(long_data)
        assert result.tensor.shape == (WINDOW_HOURS, NUM_TEMPORAL_FEATURES)


class TestNormalization:
    def test_zscore_with_custom_stats(self) -> None:
        stats = CohortStats(
            means=np.full(NUM_TEMPORAL_FEATURES, 10.0, dtype=np.float64),
            stds=np.full(NUM_TEMPORAL_FEATURES, 2.0, dtype=np.float64),
        )
        ext = TemporalFeatureExtractor(cohort_stats=stats)
        data = np.full((WINDOW_HOURS, NUM_TEMPORAL_FEATURES), 10.0, dtype=np.float64)
        result = ext.extract(data)
        np.testing.assert_allclose(result.tensor, 0.0, atol=1e-10)

    def test_compute_cohort_stats(self, extractor: TemporalFeatureExtractor) -> None:
        data_list = [
            np.ones((WINDOW_HOURS, NUM_TEMPORAL_FEATURES)) * 5.0,
            np.ones((WINDOW_HOURS, NUM_TEMPORAL_FEATURES)) * 15.0,
        ]
        stats = extractor.compute_cohort_stats(data_list)
        np.testing.assert_allclose(stats.means, 10.0, atol=1e-10)
        np.testing.assert_allclose(stats.stds, 5.0, atol=1e-10)

    def test_cohort_stats_empty(self, extractor: TemporalFeatureExtractor) -> None:
        stats = extractor.compute_cohort_stats([])
        np.testing.assert_array_equal(stats.means, 0.0)
        np.testing.assert_array_equal(stats.stds, 1.0)


class TestBatch:
    def test_batch_extraction(self, extractor: TemporalFeatureExtractor, sample_data: np.ndarray) -> None:
        batch = [sample_data, None, sample_data]
        results = extractor.extract_batch(batch)
        assert len(results) == 3
        assert results[1].completeness_score == 0.0

    def test_batch_tensor_shape(self, extractor: TemporalFeatureExtractor, sample_data: np.ndarray) -> None:
        batch = [sample_data, sample_data, sample_data]
        tensor = extractor.extract_batch_tensor(batch)
        assert tensor.shape == (3, WINDOW_HOURS, NUM_TEMPORAL_FEATURES)


class TestSufficientData:
    def test_sufficient(self) -> None:
        assert TemporalFeatureExtractor.has_sufficient_data(0.8)

    def test_insufficient(self) -> None:
        assert not TemporalFeatureExtractor.has_sufficient_data(0.5)

    def test_threshold_boundary(self) -> None:
        threshold = 1.0 - MISSING_THRESHOLD
        assert TemporalFeatureExtractor.has_sufficient_data(threshold)
        assert not TemporalFeatureExtractor.has_sufficient_data(threshold - 0.01)
