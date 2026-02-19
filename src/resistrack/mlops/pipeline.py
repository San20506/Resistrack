"""Retraining pipeline orchestrating drift monitoring and blue/green deployment."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE
from resistrack.mlops.deployment import (
    BlueGreenDeployer,
    CanaryConfig,
    DeploymentState,
    ModelVersion,
)
from resistrack.mlops.drift import DriftConfig, DriftReport, PSIDriftMonitor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RETRAIN_SCHEDULE_DAYS = 30
_SECONDS_PER_DAY = 86_400


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    retrain_schedule_days: int = _RETRAIN_SCHEDULE_DAYS
    drift_config: DriftConfig = field(default_factory=DriftConfig)
    canary_config: CanaryConfig = field(default_factory=CanaryConfig)
    auto_promote: bool = True


@dataclass(frozen=True)
class RetrainResult:
    new_version: ModelVersion
    metrics: dict[str, float]
    training_duration_s: float
    triggered_by: str  # "scheduled" | "drift" | "manual"


@dataclass(frozen=True)
class PipelineState:
    last_retrain: float | None
    next_scheduled_retrain: float
    current_version: ModelVersion
    is_retraining: bool
    drift_reports: list[DriftReport]


# ---------------------------------------------------------------------------
# Retraining Pipeline
# ---------------------------------------------------------------------------

class RetrainingPipeline:

    def __init__(self, config: PipelineConfig | None = None,
                 initial_version: ModelVersion | None = None) -> None:
        self._config = config or PipelineConfig()
        self._drift_monitor = PSIDriftMonitor(self._config.drift_config)
        self._deployer = BlueGreenDeployer(
            config=self._config.canary_config,
            initial_version=initial_version,
        )
        self._last_retrain: float | None = None
        self._next_scheduled = time.time() + (
            self._config.retrain_schedule_days * _SECONDS_PER_DAY
        )
        self._is_retraining = False
        self._drift_reports: list[DriftReport] = []
        self._rng = np.random.RandomState(RANDOM_STATE)

    def _simulate_training(self) -> tuple[dict[str, Any], dict[str, float]]:
        """Simulate model training returning (model_state, metrics)."""
        duration = self._rng.uniform(60, 300)
        model_state = {
            "weights": self._rng.uniform(0, 1, 10).tolist(),
            "trained_at": time.time(),
        }
        metrics = {
            "auc": 0.85 + self._rng.uniform(-0.03, 0.05),
            "f1": 0.78 + self._rng.uniform(-0.03, 0.05),
            "precision": 0.80 + self._rng.uniform(-0.03, 0.05),
            "recall": 0.76 + self._rng.uniform(-0.03, 0.05),
        }
        return model_state, metrics

    def _execute_retrain(self, triggered_by: str) -> RetrainResult:
        self._is_retraining = True
        start = time.time()

        model_state, metrics = self._simulate_training()
        current = self._deployer.get_deployment_state().current_version
        new_version = current.bump_minor()
        new_version = ModelVersion(
            major=new_version.major,
            minor=new_version.minor,
            patch=new_version.patch,
            created_at=time.time(),
            model_state=model_state,
            metrics=metrics,
        )

        self._deployer.deploy_canary(new_version, model_state)

        if self._config.auto_promote:
            current_auc = current.metrics.get("auc", 0.85)
            if self._deployer.evaluate_canary(current_auc, metrics["auc"]):
                self._deployer.promote_canary()

        training_duration = time.time() - start
        self._last_retrain = time.time()
        self._next_scheduled = time.time() + (
            self._config.retrain_schedule_days * _SECONDS_PER_DAY
        )
        self._is_retraining = False

        return RetrainResult(
            new_version=new_version,
            metrics=metrics,
            training_duration_s=training_duration,
            triggered_by=triggered_by,
        )

    def check_and_retrain(
        self,
        current_data: dict[str, np.ndarray],
        baseline_data: dict[str, np.ndarray],
    ) -> RetrainResult | None:
        report = self._drift_monitor.monitor_features(
            baseline_data, current_data,
        )
        self._drift_reports.append(report)

        if self._drift_monitor.should_trigger_retrain(report):
            return self._execute_retrain(triggered_by="drift")

        if time.time() >= self._next_scheduled:
            return self._execute_retrain(triggered_by="scheduled")

        return None

    def schedule_retrain(self) -> float:
        self._next_scheduled = time.time() + (
            self._config.retrain_schedule_days * _SECONDS_PER_DAY
        )
        return self._next_scheduled

    def force_retrain(self, reason: str = "manual") -> RetrainResult:
        return self._execute_retrain(triggered_by=reason)

    def get_pipeline_state(self) -> PipelineState:
        deployment = self._deployer.get_deployment_state()
        return PipelineState(
            last_retrain=self._last_retrain,
            next_scheduled_retrain=self._next_scheduled,
            current_version=deployment.current_version,
            is_retraining=self._is_retraining,
            drift_reports=self._drift_reports,
        )

    @property
    def deployer(self) -> BlueGreenDeployer:
        return self._deployer

    @property
    def drift_monitor(self) -> PSIDriftMonitor:
        return self._drift_monitor
