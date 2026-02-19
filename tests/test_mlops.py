"""Tests for M5.2 – MLOps retraining pipeline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from resistrack.common.constants import RANDOM_STATE
from resistrack.mlops.deployment import (
    BlueGreenDeployer,
    CanaryConfig,
    DeploymentState,
    ModelVersion,
    RollbackEntry,
)
from resistrack.mlops.drift import (
    DriftConfig,
    DriftReport,
    DriftResult,
    PSIDriftMonitor,
)
from resistrack.mlops.pipeline import (
    PipelineConfig,
    PipelineState,
    RetrainResult,
    RetrainingPipeline,
)


# --- Helpers ---

_RNG = np.random.RandomState(RANDOM_STATE)


def _stable_data(n: int = 500) -> np.ndarray:
    return _RNG.normal(0, 1, n)


def _drifted_data(n: int = 500) -> np.ndarray:
    return _RNG.normal(5, 2, n)


# --- PSI Drift Tests ---

def test_psi_identical_distributions():
    rng = np.random.RandomState(42)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(0, 1, 1000)
    psi = PSIDriftMonitor.compute_psi(baseline, current)
    assert psi < 0.10  # Nearly identical


def test_psi_drifted_distributions():
    rng = np.random.RandomState(42)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(5, 2, 1000)
    psi = PSIDriftMonitor.compute_psi(baseline, current)
    assert psi > 0.20  # Clearly drifted


def test_psi_non_negative():
    rng = np.random.RandomState(42)
    baseline = rng.uniform(0, 1, 500)
    current = rng.uniform(0.5, 1.5, 500)
    psi = PSIDriftMonitor.compute_psi(baseline, current)
    assert psi >= 0.0


def test_monitor_features_no_drift():
    rng = np.random.RandomState(42)
    monitor = PSIDriftMonitor(DriftConfig(psi_threshold=0.20, min_samples=50))
    baseline = {"feat1": rng.normal(0, 1, 500), "feat2": rng.normal(0, 1, 500)}
    current = {"feat1": rng.normal(0, 1, 500), "feat2": rng.normal(0, 1, 500)}
    report = monitor.monitor_features(baseline, current)
    assert isinstance(report, DriftReport)
    assert not report.overall_drift_detected
    assert not report.emergency_retrain_triggered


def test_monitor_features_with_drift():
    rng = np.random.RandomState(42)
    monitor = PSIDriftMonitor(DriftConfig(psi_threshold=0.20, min_samples=50))
    baseline = {"feat1": rng.normal(0, 1, 500)}
    current = {"feat1": rng.normal(5, 2, 500)}
    report = monitor.monitor_features(baseline, current)
    assert report.overall_drift_detected
    assert report.emergency_retrain_triggered


def test_should_trigger_retrain():
    rng = np.random.RandomState(42)
    monitor = PSIDriftMonitor(DriftConfig(psi_threshold=0.20, min_samples=50))
    baseline = {"feat1": rng.normal(0, 1, 500)}
    current = {"feat1": rng.normal(5, 2, 500)}
    report = monitor.monitor_features(baseline, current)
    assert monitor.should_trigger_retrain(report) is True


def test_monitoring_summary():
    rng = np.random.RandomState(42)
    monitor = PSIDriftMonitor(DriftConfig(psi_threshold=0.20, min_samples=50))
    baseline = {"feat1": rng.normal(0, 1, 500)}
    current = {"feat1": rng.normal(0, 1, 500)}
    monitor.monitor_features(baseline, current)
    summary = monitor.get_monitoring_summary()
    assert summary["total_reports"] == 1
    assert "config" in summary


def test_drift_config_defaults():
    cfg = DriftConfig()
    assert cfg.psi_threshold == 0.20
    assert cfg.monitoring_window_days == 30
    assert cfg.min_samples == 100


def test_min_samples_skip():
    monitor = PSIDriftMonitor(DriftConfig(min_samples=1000))
    baseline = {"feat1": np.array([1.0] * 50)}
    current = {"feat1": np.array([1.0] * 50)}
    report = monitor.monitor_features(baseline, current)
    assert len(report.results) == 0


# --- Model Version Tests ---

def test_version_str():
    v = ModelVersion(major=1, minor=2, patch=3)
    assert str(v) == "1.2.3"


def test_version_bump_major():
    v = ModelVersion(major=1, minor=2, patch=3)
    v2 = v.bump_major()
    assert str(v2) == "2.0.0"


def test_version_bump_minor():
    v = ModelVersion(major=1, minor=2, patch=3)
    v2 = v.bump_minor()
    assert str(v2) == "1.3.0"


def test_version_bump_patch():
    v = ModelVersion(major=1, minor=2, patch=3)
    v2 = v.bump_patch()
    assert str(v2) == "1.2.4"


# --- Blue/Green Deployer Tests ---

def test_deploy_canary():
    deployer = BlueGreenDeployer()
    v2 = ModelVersion(major=1, minor=1, patch=0, created_at=time.time())
    state = deployer.deploy_canary(v2)
    assert state.is_canary_active is True
    assert state.canary_version is not None
    assert str(state.canary_version) == "1.1.0"


def test_evaluate_canary_within_tolerance():
    deployer = BlueGreenDeployer(CanaryConfig(auc_tolerance=0.02))
    result = deployer.evaluate_canary(current_auc=0.85, canary_auc=0.84)
    assert result is True


def test_evaluate_canary_outside_tolerance():
    deployer = BlueGreenDeployer(CanaryConfig(auc_tolerance=0.02))
    result = deployer.evaluate_canary(current_auc=0.85, canary_auc=0.80)
    assert result is False


def test_promote_canary():
    deployer = BlueGreenDeployer()
    v2 = ModelVersion(major=1, minor=1, patch=0, created_at=time.time())
    deployer.deploy_canary(v2)
    promoted = deployer.promote_canary()
    assert str(promoted) == "1.1.0"
    state = deployer.get_deployment_state()
    assert state.canary_version is None
    assert not state.is_canary_active


def test_promote_no_canary_raises():
    deployer = BlueGreenDeployer()
    with pytest.raises(ValueError, match="No canary"):
        deployer.promote_canary()


def test_rollback():
    deployer = BlueGreenDeployer()
    v2 = ModelVersion(major=1, minor=1, patch=0, created_at=time.time())
    deployer.deploy_canary(v2)
    deployer.promote_canary()
    rolled_back = deployer.rollback()
    assert str(rolled_back) == "1.0.0"


def test_rollback_no_history_raises():
    deployer = BlueGreenDeployer()
    with pytest.raises(ValueError, match="No versions"):
        deployer.rollback()


def test_rollback_max_two_versions():
    deployer = BlueGreenDeployer()
    for i in range(4):
        v = ModelVersion(major=1, minor=i + 1, patch=0, created_at=time.time())
        deployer.deploy_canary(v)
        deployer.promote_canary()
    history = deployer.get_rollback_history()
    assert len(history) <= 2


def test_deployment_state():
    deployer = BlueGreenDeployer()
    state = deployer.get_deployment_state()
    assert isinstance(state, DeploymentState)
    assert state.current_version is not None
    assert not state.is_canary_active


# --- Pipeline Tests ---

def test_pipeline_force_retrain():
    pipeline = RetrainingPipeline()
    result = pipeline.force_retrain(reason="manual")
    assert isinstance(result, RetrainResult)
    assert result.triggered_by == "manual"
    assert result.training_duration_s >= 0


def test_pipeline_drift_triggered_retrain():
    rng = np.random.RandomState(42)
    pipeline = RetrainingPipeline(PipelineConfig(
        drift_config=DriftConfig(psi_threshold=0.20, min_samples=50),
    ))
    baseline = {"feat1": rng.normal(0, 1, 500)}
    current = {"feat1": rng.normal(5, 2, 500)}
    result = pipeline.check_and_retrain(current, baseline)
    assert result is not None
    assert result.triggered_by == "drift"


def test_pipeline_no_drift_no_retrain():
    rng = np.random.RandomState(42)
    pipeline = RetrainingPipeline(PipelineConfig(
        drift_config=DriftConfig(psi_threshold=0.20, min_samples=50),
    ))
    baseline = {"feat1": rng.normal(0, 1, 500)}
    current = {"feat1": rng.normal(0, 1, 500)}
    result = pipeline.check_and_retrain(current, baseline)
    # No drift, and schedule not yet reached
    assert result is None


def test_pipeline_state():
    pipeline = RetrainingPipeline()
    state = pipeline.get_pipeline_state()
    assert isinstance(state, PipelineState)
    assert state.last_retrain is None
    assert not state.is_retraining


def test_pipeline_schedule():
    pipeline = RetrainingPipeline()
    next_time = pipeline.schedule_retrain()
    assert next_time > time.time()


def test_pipeline_version_after_retrain():
    pipeline = RetrainingPipeline()
    result = pipeline.force_retrain(reason="test")
    assert result.new_version.minor > 0
