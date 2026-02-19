"""Blue/green deployment with canary testing, rollback, and semantic versioning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANARY_TRAFFIC_PCT = 0.10
_PROMOTION_WINDOW_HOURS = 72
_AUC_TOLERANCE = 0.02
_MAX_ROLLBACK_VERSIONS = 2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelVersion:
    major: int
    minor: int
    patch: int
    created_at: float = 0.0
    model_state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_major(self) -> ModelVersion:
        return ModelVersion(
            major=self.major + 1, minor=0, patch=0,
            created_at=time.time(), model_state=self.model_state,
            metrics=self.metrics,
        )

    def bump_minor(self) -> ModelVersion:
        return ModelVersion(
            major=self.major, minor=self.minor + 1, patch=0,
            created_at=time.time(), model_state=self.model_state,
            metrics=self.metrics,
        )

    def bump_patch(self) -> ModelVersion:
        return ModelVersion(
            major=self.major, minor=self.minor, patch=self.patch + 1,
            created_at=time.time(), model_state=self.model_state,
            metrics=self.metrics,
        )


@dataclass(frozen=True)
class CanaryConfig:
    canary_traffic_pct: float = _CANARY_TRAFFIC_PCT
    promotion_window_hours: int = _PROMOTION_WINDOW_HOURS
    auc_tolerance: float = _AUC_TOLERANCE


@dataclass(frozen=True)
class DeploymentState:
    current_version: ModelVersion
    canary_version: ModelVersion | None = None
    canary_start_time: float | None = None
    canary_metrics: dict[str, float] = field(default_factory=dict)
    is_canary_active: bool = False


@dataclass(frozen=True)
class RollbackEntry:
    version: ModelVersion
    model_state: dict[str, Any] = field(default_factory=dict)
    rolled_back_at: float | None = None


# ---------------------------------------------------------------------------
# Blue/Green Deployer
# ---------------------------------------------------------------------------

class BlueGreenDeployer:

    def __init__(self, config: CanaryConfig | None = None,
                 initial_version: ModelVersion | None = None) -> None:
        self._config = config or CanaryConfig()
        self._current_version = initial_version or ModelVersion(
            major=1, minor=0, patch=0, created_at=time.time(),
        )
        self._canary_version: ModelVersion | None = None
        self._canary_start_time: float | None = None
        self._canary_metrics: dict[str, float] = {}
        self._rollback_history: list[RollbackEntry] = []
        self._rng = np.random.RandomState(RANDOM_STATE)

    def deploy_canary(self, new_version: ModelVersion,
                      model_state: dict[str, Any] | None = None,
                      ) -> DeploymentState:
        if model_state:
            new_version = ModelVersion(
                major=new_version.major,
                minor=new_version.minor,
                patch=new_version.patch,
                created_at=new_version.created_at or time.time(),
                model_state=model_state,
                metrics=new_version.metrics,
            )
        self._canary_version = new_version
        self._canary_start_time = time.time()
        self._canary_metrics = {}
        return self.get_deployment_state()

    def evaluate_canary(self, current_auc: float,
                        canary_auc: float) -> bool:
        """True if canary AUC is within tolerance of current."""
        self._canary_metrics = {
            "current_auc": current_auc,
            "canary_auc": canary_auc,
            "auc_diff": abs(current_auc - canary_auc),
        }
        return abs(current_auc - canary_auc) <= self._config.auc_tolerance

    def promote_canary(self) -> ModelVersion:
        if self._canary_version is None:
            raise ValueError("No canary deployment active")

        self._rollback_history.append(RollbackEntry(
            version=self._current_version,
            model_state=self._current_version.model_state,
        ))
        if len(self._rollback_history) > _MAX_ROLLBACK_VERSIONS:
            self._rollback_history = self._rollback_history[
                -_MAX_ROLLBACK_VERSIONS:
            ]

        self._current_version = self._canary_version
        self._canary_version = None
        self._canary_start_time = None
        self._canary_metrics = {}
        return self._current_version

    def rollback(self, target_version: ModelVersion | None = None,
                 ) -> ModelVersion:
        if not self._rollback_history:
            raise ValueError("No versions available for rollback")

        if target_version is not None:
            for entry in reversed(self._rollback_history):
                if str(entry.version) == str(target_version):
                    self._current_version = entry.version
                    entry_with_ts = RollbackEntry(
                        version=entry.version,
                        model_state=entry.model_state,
                        rolled_back_at=time.time(),
                    )
                    self._rollback_history.remove(entry)
                    self._rollback_history.append(entry_with_ts)
                    self._canary_version = None
                    self._canary_start_time = None
                    return self._current_version
            raise ValueError(
                f"Version {target_version} not found in rollback history"
            )

        entry = self._rollback_history.pop()
        self._current_version = entry.version
        self._canary_version = None
        self._canary_start_time = None
        return self._current_version

    def get_deployment_state(self) -> DeploymentState:
        return DeploymentState(
            current_version=self._current_version,
            canary_version=self._canary_version,
            canary_start_time=self._canary_start_time,
            canary_metrics=self._canary_metrics,
            is_canary_active=self._canary_version is not None,
        )

    def get_rollback_history(self) -> list[RollbackEntry]:
        return list(self._rollback_history)
