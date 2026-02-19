"""Model card generator for FDA PCCP compliance.

Generates structured model card documentation from validation results
as required by M5.1 spec and ResisTrack Agent Rules.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from resistrack.validation.validator import (
    METRIC_THRESHOLDS,
    ValidationReport,
)


@dataclasses.dataclass(frozen=True)
class ModelCard:
    """Structured model card for regulatory documentation."""

    model_name: str
    model_version: str
    generated_at: str
    intended_use: str
    validation_summary: dict[str, object]
    metrics: dict[str, float]
    metric_thresholds: dict[str, float]
    metric_pass_fail: dict[str, bool]
    subgroup_analysis: list[dict[str, object]]
    fairness_flags: list[str]
    confidence_interval: dict[str, float]
    limitations: list[str]
    ethical_considerations: list[str]


class ModelCardGenerator:
    """Generates model cards from validation reports.

    Parameters
    ----------
    model_name : str
        Name of the model for the card header.
    """

    def __init__(self, model_name: str = "ResisTrack AMR Ensemble") -> None:
        self._model_name = model_name

    def generate(self, report: ValidationReport) -> ModelCard:
        """Generate a model card from a validation report.

        Parameters
        ----------
        report : ValidationReport
            Completed validation report with metrics and subgroup analysis.

        Returns
        -------
        ModelCard
            Structured model card ready for documentation.
        """
        metrics_dict = {
            "auc_roc": report.metrics.auc_roc,
            "auprc": report.metrics.auprc,
            "sensitivity_at_80_spec": report.metrics.sensitivity_at_80_spec,
            "fpr": report.metrics.fpr,
            "brier": report.metrics.brier,
            "p95_latency_ms": report.metrics.p95_latency_ms,
        }

        subgroup_list = [
            {
                "name": sg.name,
                "n_samples": sg.n_samples,
                "auc_roc": round(sg.auc_roc, 4),
                "auc_gap": round(sg.auc_gap, 4),
                "flagged": sg.flagged,
            }
            for sg in report.subgroup_analysis.subgroups
        ]

        fairness_flags = [
            f"{sg.name}: AUC gap {sg.auc_gap:.2%}"
            for sg in report.subgroup_analysis.flagged_subgroups
        ]

        return ModelCard(
            model_name=self._model_name,
            model_version=report.model_version,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            intended_use=(
                "Clinical decision support for antimicrobial resistance risk "
                "prediction in hospital settings. Not for standalone diagnostic use."
            ),
            validation_summary={
                "n_records": report.n_records,
                "overall_pass": report.overall_pass,
                "all_metrics_pass": report.metrics.all_pass,
                "fairness_pass": not report.subgroup_analysis.any_flagged,
            },
            metrics=metrics_dict,
            metric_thresholds=dict(METRIC_THRESHOLDS),
            metric_pass_fail=report.metrics.passes_thresholds(),
            subgroup_analysis=subgroup_list,
            fairness_flags=fairness_flags,
            confidence_interval={
                "auc_roc_lower_95": round(report.ci_lower, 4),
                "auc_roc_upper_95": round(report.ci_upper, 4),
            },
            limitations=[
                "Validated on synthetic/simulated data only",
                "Performance may vary across hospital populations",
                "Requires minimum 1000 held-out records for validation",
                "Subgroup analysis limited to age and ICU status",
            ],
            ethical_considerations=[
                "Model should augment, not replace, clinical judgment",
                "Override monitoring required per M3.3 spec",
                "No PHI included in model card or validation artifacts",
                "Bias monitoring required for demographic subgroups",
            ],
        )

    def to_markdown(self, card: ModelCard) -> str:
        """Render model card as Markdown for documentation."""
        lines = [
            f"# Model Card: {card.model_name}",
            f"**Version**: {card.model_version}",
            f"**Generated**: {card.generated_at}",
            "",
            "## Intended Use",
            card.intended_use,
            "",
            "## Validation Summary",
            f"- Records: {card.validation_summary['n_records']}",
            f"- Overall Pass: {card.validation_summary['overall_pass']}",
            f"- Metrics Pass: {card.validation_summary['all_metrics_pass']}",
            f"- Fairness Pass: {card.validation_summary['fairness_pass']}",
            "",
            "## Performance Metrics",
            "| Metric | Value | Threshold | Pass |",
            "|--------|-------|-----------|------|",
        ]

        for metric, value in card.metrics.items():
            threshold = card.metric_thresholds.get(metric, "N/A")
            passed = card.metric_pass_fail.get(metric, False)
            status = "✅" if passed else "❌"
            lines.append(
                f"| {metric} | {value:.4f} | {threshold} | {status} |"
            )

        lines.extend([
            "",
            "## Confidence Interval (AUC-ROC, 95%)",
            f"- Lower: {card.confidence_interval['auc_roc_lower_95']:.4f}",
            f"- Upper: {card.confidence_interval['auc_roc_upper_95']:.4f}",
            "",
            "## Subgroup Analysis",
        ])

        if card.subgroup_analysis:
            lines.extend([
                "| Subgroup | N | AUC-ROC | Gap | Flagged |",
                "|----------|---|---------|-----|---------|",
            ])
            for sg in card.subgroup_analysis:
                flag = "⚠️" if sg["flagged"] else "—"
                lines.append(
                    f"| {sg['name']} | {sg['n_samples']} "
                    f"| {sg['auc_roc']:.4f} | {sg['auc_gap']:.4f} | {flag} |"
                )
        else:
            lines.append("No subgroup data provided.")

        if card.fairness_flags:
            lines.extend(["", "### Fairness Flags"])
            for flag in card.fairness_flags:
                lines.append(f"- ⚠️ {flag}")

        lines.extend(["", "## Limitations"])
        for lim in card.limitations:
            lines.append(f"- {lim}")

        lines.extend(["", "## Ethical Considerations"])
        for ec in card.ethical_considerations:
            lines.append(f"- {ec}")

        return "\n".join(lines)
