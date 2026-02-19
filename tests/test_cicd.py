"""Tests for M5.3 – CI/CD pipeline."""

from __future__ import annotations

import pytest

from resistrack.cicd.pipeline import (
    BuildResult,
    BuildStage,
    CICDPipeline,
    DeploymentGate,
    GateResult,
    PipelineConfig,
    PipelineRun,
    QualityGate,
    StageStatus,
)


# --- StageStatus Tests ---


def test_stage_status_enum():
    assert StageStatus.PASSED == "passed"
    assert StageStatus.FAILED == "failed"
    assert StageStatus.PENDING == "pending"


# --- QualityGate Tests ---


def test_quality_gate_mypy_pass():
    gate = QualityGate()
    result = gate.check_mypy(error_count=0)
    assert result.passed is True
    assert result.gate_name == "mypy_strict"


def test_quality_gate_mypy_fail():
    gate = QualityGate()
    result = gate.check_mypy(error_count=5)
    assert result.passed is False


def test_quality_gate_coverage_pass():
    gate = QualityGate()
    result = gate.check_coverage(coverage_pct=85.0)
    assert result.passed is True
    assert result.gate_name == "test_coverage"


def test_quality_gate_coverage_fail():
    gate = QualityGate()
    result = gate.check_coverage(coverage_pct=70.0)
    assert result.passed is False


def test_quality_gate_coverage_custom_threshold():
    cfg = PipelineConfig(min_coverage_pct=90.0)
    gate = QualityGate(cfg)
    result = gate.check_coverage(coverage_pct=85.0)
    assert result.passed is False


def test_quality_gate_tests_pass():
    gate = QualityGate()
    result = gate.check_tests(total=100, passed=100, failed=0)
    assert result.passed is True
    assert result.gate_name == "test_suite"


def test_quality_gate_tests_fail():
    gate = QualityGate()
    result = gate.check_tests(total=100, passed=95, failed=5)
    assert result.passed is False


def test_quality_gate_check_all():
    gate = QualityGate()
    results = gate.check_all(
        coverage_pct=85.0, mypy_errors=0,
        tests_total=100, tests_passed=100, tests_failed=0,
    )
    assert len(results) == 3
    assert all(r.passed for r in results)


# --- BuildStage Tests ---


def test_build_stage_success():
    stage = BuildStage("lint")
    result = stage.execute("mypy --strict src/", should_succeed=True)
    assert isinstance(result, BuildResult)
    assert result.status == StageStatus.PASSED
    assert result.exit_code == 0


def test_build_stage_failure():
    stage = BuildStage("lint")
    result = stage.execute("mypy --strict src/", should_succeed=False)
    assert result.status == StageStatus.FAILED
    assert result.exit_code == 1


# --- DeploymentGate Tests ---


def test_deployment_gate_pass():
    gate = DeploymentGate()
    result = gate.check_regression(
        baseline_metrics={"auc": 0.85, "precision": 0.80},
        current_metrics={"auc": 0.84, "precision": 0.79},
        tolerance=0.02,
    )
    assert result.passed is True
    assert result.gate_name == "regression_test"


def test_deployment_gate_fail():
    gate = DeploymentGate()
    result = gate.check_regression(
        baseline_metrics={"auc": 0.85, "precision": 0.80},
        current_metrics={"auc": 0.70, "precision": 0.60},
        tolerance=0.02,
    )
    assert result.passed is False


def test_deployment_gate_tolerance():
    gate = DeploymentGate()
    result = gate.check_regression(
        baseline_metrics={"auc": 0.85},
        current_metrics={"auc": 0.80},
        tolerance=0.05,
    )
    assert result.passed is True


def test_deployment_gate_container():
    gate = DeploymentGate()
    result = gate.check_container_build(image_built=True, image_scanned=True)
    assert result.passed is True


def test_deployment_gate_secrets():
    gate = DeploymentGate()
    result = gate.check_secrets(secrets_valid=True)
    assert result.passed is True


# --- CICDPipeline Tests ---


def test_pipeline_creation():
    pipeline = CICDPipeline()
    assert pipeline.config.min_coverage_pct == 80.0
    assert pipeline.config.mypy_strict is True


def test_pipeline_custom_config():
    cfg = PipelineConfig(min_coverage_pct=90.0, deployment_gate_enabled=False)
    pipeline = CICDPipeline(cfg)
    assert pipeline.config.min_coverage_pct == 90.0


def test_pipeline_run_all_pass():
    pipeline = CICDPipeline()
    run = pipeline.run_pipeline(
        coverage_pct=85.0, mypy_errors=0,
        tests_total=100, tests_passed=100, tests_failed=0,
    )
    assert isinstance(run, PipelineRun)
    assert run.overall_status == StageStatus.PASSED
    assert len(run.stages) == 3
    assert len(run.gates) == 3


def test_pipeline_run_mypy_fail():
    pipeline = CICDPipeline()
    run = pipeline.run_pipeline(
        coverage_pct=85.0, mypy_errors=5,
        tests_total=100, tests_passed=100, tests_failed=0,
    )
    assert run.overall_status == StageStatus.FAILED


def test_pipeline_run_coverage_fail():
    pipeline = CICDPipeline()
    run = pipeline.run_pipeline(
        coverage_pct=70.0, mypy_errors=0,
        tests_total=100, tests_passed=100, tests_failed=0,
    )
    assert run.overall_status == StageStatus.FAILED


def test_pipeline_run_history():
    pipeline = CICDPipeline()
    pipeline.run_pipeline(coverage_pct=85.0, mypy_errors=0,
                          tests_total=100, tests_passed=100, tests_failed=0)
    pipeline.run_pipeline(coverage_pct=85.0, mypy_errors=0,
                          tests_total=100, tests_passed=100, tests_failed=0)
    history = pipeline.get_run_history()
    assert len(history) == 2


def test_pipeline_deployment_gate_blocks():
    pipeline = CICDPipeline(PipelineConfig(deployment_gate_enabled=True))
    run = pipeline.run_pipeline(
        coverage_pct=85.0, mypy_errors=0,
        tests_total=100, tests_passed=100, tests_failed=0,
        baseline_metrics={"auc": 0.85},
        current_metrics={"auc": 0.70},
    )
    assert run.overall_status == StageStatus.FAILED


def test_pipeline_deployment_gate_allows():
    pipeline = CICDPipeline(PipelineConfig(deployment_gate_enabled=True))
    run = pipeline.run_pipeline(
        coverage_pct=85.0, mypy_errors=0,
        tests_total=100, tests_passed=100, tests_failed=0,
        baseline_metrics={"auc": 0.85},
        current_metrics={"auc": 0.84},
    )
    assert run.overall_status == StageStatus.PASSED


def test_pipeline_stats():
    pipeline = CICDPipeline()
    pipeline.run_pipeline(coverage_pct=85.0, mypy_errors=0,
                          tests_total=100, tests_passed=100, tests_failed=0)
    stats = pipeline.get_stats()
    assert stats["total_runs"] == 1
    assert stats["passed"] == 1
