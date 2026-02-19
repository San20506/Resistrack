"""CI/CD pipeline configuration and quality gates.

Implements CodePipeline-style stages with mypy --strict
enforcement, test coverage gates, and deployment gates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE


# --- Enums ---


class StageStatus(StrEnum):
    """Pipeline stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


# --- Data Classes ---


@dataclass(frozen=True)
class PipelineConfig:
    """CI/CD pipeline configuration."""

    min_coverage_pct: float = 80.0
    mypy_strict: bool = True
    require_all_tests_pass: bool = True
    deployment_gate_enabled: bool = True
    max_build_time_seconds: float = 600.0
    container_registry: str = "resistrack-ecr"
    secrets_manager_arn: str = "arn:aws:secretsmanager:us-east-1:*:secret:resistrack/*"


@dataclass(frozen=True)
class BuildResult:
    """Result of a build stage execution."""

    stage_name: str
    status: StageStatus
    duration_seconds: float
    output: str
    exit_code: int
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GateResult:
    """Result of a quality/deployment gate check."""

    gate_name: str
    passed: bool
    metric_value: float
    threshold: float
    message: str


@dataclass(frozen=True)
class PipelineRun:
    """Complete pipeline run result."""

    run_id: str
    stages: list[BuildResult]
    gates: list[GateResult]
    overall_status: StageStatus
    total_duration_seconds: float
    timestamp: float = field(default_factory=time.time)


# --- Quality Gate ---


class QualityGate:
    """Enforces code quality standards."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def check_coverage(self, coverage_pct: float) -> GateResult:
        """Check test coverage meets minimum threshold."""
        passed = coverage_pct >= self._config.min_coverage_pct
        return GateResult(
            gate_name="test_coverage",
            passed=passed,
            metric_value=coverage_pct,
            threshold=self._config.min_coverage_pct,
            message=(
                f"Coverage {coverage_pct:.1f}% "
                f"{'meets' if passed else 'below'} "
                f"minimum {self._config.min_coverage_pct:.1f}%"
            ),
        )

    def check_mypy(self, error_count: int) -> GateResult:
        """Check mypy --strict passes with zero errors."""
        passed = error_count == 0
        return GateResult(
            gate_name="mypy_strict",
            passed=passed,
            metric_value=float(error_count),
            threshold=0.0,
            message=(
                f"mypy --strict: {error_count} error(s) "
                f"{'(clean)' if passed else '(FAILED)'}"
            ),
        )

    def check_tests(self, total: int, passed: int, failed: int) -> GateResult:
        """Check all tests pass."""
        all_passed = failed == 0 and passed > 0
        return GateResult(
            gate_name="test_suite",
            passed=all_passed,
            metric_value=float(passed),
            threshold=float(total),
            message=(
                f"Tests: {passed}/{total} passed, {failed} failed"
            ),
        )

    def check_all(
        self,
        coverage_pct: float,
        mypy_errors: int,
        tests_total: int,
        tests_passed: int,
        tests_failed: int,
    ) -> list[GateResult]:
        """Run all quality gate checks."""
        return [
            self.check_coverage(coverage_pct),
            self.check_mypy(mypy_errors),
            self.check_tests(tests_total, tests_passed, tests_failed),
        ]


# --- Deployment Gate ---


class DeploymentGate:
    """Controls deployment based on regression test results."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._rng = np.random.RandomState(RANDOM_STATE)

    def check_regression(
        self,
        baseline_metrics: dict[str, float],
        current_metrics: dict[str, float],
        tolerance: float = 0.02,
    ) -> GateResult:
        """Block deployment if any metric regresses beyond tolerance."""
        regressions: list[str] = []
        for metric, baseline_val in baseline_metrics.items():
            current_val = current_metrics.get(metric, 0.0)
            if current_val < baseline_val - tolerance:
                regressions.append(
                    f"{metric}: {current_val:.4f} < {baseline_val:.4f} - {tolerance}"
                )

        passed = len(regressions) == 0
        worst_regression = 0.0
        if not passed:
            for metric in baseline_metrics:
                current = current_metrics.get(metric, 0.0)
                diff = baseline_metrics[metric] - current
                worst_regression = max(worst_regression, diff)

        return GateResult(
            gate_name="regression_test",
            passed=passed,
            metric_value=worst_regression,
            threshold=tolerance,
            message=(
                "No regressions detected"
                if passed
                else f"Regressions: {'; '.join(regressions)}"
            ),
        )

    def check_container_build(self, image_built: bool, image_scanned: bool) -> GateResult:
        """Verify container image was built and scanned."""
        passed = image_built and image_scanned
        return GateResult(
            gate_name="container_build",
            passed=passed,
            metric_value=1.0 if passed else 0.0,
            threshold=1.0,
            message=(
                "Container built and scanned"
                if passed
                else f"Container: built={image_built}, scanned={image_scanned}"
            ),
        )

    def check_secrets(self, secrets_valid: bool) -> GateResult:
        """Verify secrets are accessible from Secrets Manager."""
        return GateResult(
            gate_name="secrets_validation",
            passed=secrets_valid,
            metric_value=1.0 if secrets_valid else 0.0,
            threshold=1.0,
            message=(
                "Secrets Manager access verified"
                if secrets_valid
                else "Secrets Manager access FAILED"
            ),
        )


# --- Build Stage ---


class BuildStage:
    """Simulates a CodeBuild build stage."""

    def __init__(self, stage_name: str, config: PipelineConfig | None = None) -> None:
        self._stage_name = stage_name
        self._config = config or PipelineConfig()
        self._rng = np.random.RandomState(RANDOM_STATE)

    @property
    def stage_name(self) -> str:
        return self._stage_name

    def execute(self, command: str, should_succeed: bool = True) -> BuildResult:
        """Execute a build stage (simulated)."""
        start = time.time()
        duration = self._rng.uniform(0.5, 5.0)
        status = StageStatus.PASSED if should_succeed else StageStatus.FAILED
        exit_code = 0 if should_succeed else 1

        return BuildResult(
            stage_name=self._stage_name,
            status=status,
            duration_seconds=duration,
            output=f"[{self._stage_name}] {command}: {'OK' if should_succeed else 'FAILED'}",
            exit_code=exit_code,
        )


# --- CI/CD Pipeline ---


class CICDPipeline:
    """Full CI/CD pipeline orchestrator."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._quality_gate = QualityGate(self._config)
        self._deployment_gate = DeploymentGate(self._config)
        self._runs: list[PipelineRun] = []
        self._run_counter = 0

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def quality_gate(self) -> QualityGate:
        return self._quality_gate

    @property
    def deployment_gate(self) -> DeploymentGate:
        return self._deployment_gate

    def run_pipeline(
        self,
        coverage_pct: float = 85.0,
        mypy_errors: int = 0,
        tests_total: int = 100,
        tests_passed: int = 100,
        tests_failed: int = 0,
        baseline_metrics: dict[str, float] | None = None,
        current_metrics: dict[str, float] | None = None,
    ) -> PipelineRun:
        """Execute full pipeline with all stages and gates."""
        start = time.time()
        self._run_counter += 1
        run_id = f"run-{self._run_counter}"

        stages: list[BuildResult] = []
        gates: list[GateResult] = []

        # Stage 1: Lint / mypy
        lint_stage = BuildStage("lint", self._config)
        stages.append(lint_stage.execute("mypy --strict src/", should_succeed=mypy_errors == 0))

        # Stage 2: Test
        test_stage = BuildStage("test", self._config)
        stages.append(test_stage.execute("pytest tests/ -v", should_succeed=tests_failed == 0))

        # Stage 3: Build container
        build_stage = BuildStage("build", self._config)
        stages.append(build_stage.execute("docker build -t resistrack .", should_succeed=True))

        # Quality gates
        gates.extend(
            self._quality_gate.check_all(
                coverage_pct, mypy_errors, tests_total, tests_passed, tests_failed
            )
        )

        # Deployment gate (if enabled)
        if self._config.deployment_gate_enabled and baseline_metrics and current_metrics:
            gates.append(
                self._deployment_gate.check_regression(baseline_metrics, current_metrics)
            )

        # Overall status
        all_stages_passed = all(s.status == StageStatus.PASSED for s in stages)
        all_gates_passed = all(g.passed for g in gates)
        overall = StageStatus.PASSED if (all_stages_passed and all_gates_passed) else StageStatus.FAILED

        run = PipelineRun(
            run_id=run_id,
            stages=stages,
            gates=gates,
            overall_status=overall,
            total_duration_seconds=time.time() - start,
        )
        self._runs.append(run)
        return run

    def get_run_history(self) -> list[PipelineRun]:
        """Return all pipeline run results."""
        return list(self._runs)

    def get_stats(self) -> dict[str, Any]:
        """Pipeline execution statistics."""
        passed = sum(1 for r in self._runs if r.overall_status == StageStatus.PASSED)
        failed = sum(1 for r in self._runs if r.overall_status == StageStatus.FAILED)
        return {
            "total_runs": len(self._runs),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / max(len(self._runs), 1),
        }
