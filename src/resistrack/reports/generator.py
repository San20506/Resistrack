"""M4.5 Stewardship Report Generator.

Generates weekly antimicrobial stewardship reports in CSV and summary formats.
Reports include: risk distribution, antibiotic usage, resistance trends,
and intervention recommendations.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for report generation."""

    report_period_days: int = 7
    include_patient_details: bool = False  # PHI guard: default off
    output_formats: tuple[str, ...] = ("csv", "json")
    s3_bucket: str = ""
    s3_prefix: str = "reports/stewardship/"


@dataclass
class ReportOutput:
    """Generated report output."""

    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    summary: dict[str, Any] = field(default_factory=dict)
    csv_content: str = ""
    json_content: str = ""


class ReportGenerator:
    """Generate antimicrobial stewardship reports.

    Produces weekly reports with:
    - Risk tier distribution across patients
    - Top antibiotic classes by resistance risk
    - Ward-level AMR trends
    - Intervention recommendations based on thresholds
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()

    def generate(
        self,
        predictions: list[dict[str, Any]],
        period_start: str,
        period_end: str,
    ) -> ReportOutput:
        """Generate a stewardship report from prediction data.

        Args:
            predictions: List of AMR prediction dicts (tokenized, no PHI).
            period_start: ISO date string for report period start.
            period_end: ISO date string for report period end.

        Returns:
            ReportOutput with summary, CSV, and JSON content.
        """
        report_id = f"RPT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        summary = self._compute_summary(predictions)
        csv_content = self._generate_csv(predictions, summary)
        json_content = self._generate_json(summary, period_start, period_end, report_id)

        return ReportOutput(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            period_start=period_start,
            period_end=period_end,
            summary=summary,
            csv_content=csv_content,
            json_content=json_content,
        )

    def _compute_summary(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute summary statistics from predictions."""
        total = len(predictions)
        if total == 0:
            return {
                "total_patients": 0,
                "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
                "avg_risk_score": 0.0,
                "low_confidence_count": 0,
                "high_risk_rate": 0.0,
                "top_antibiotic_risks": {},
                "recommendations": [],
            }

        tier_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        total_score = 0.0
        low_conf_count = 0
        antibiotic_totals: dict[str, float] = {
            "penicillins": 0.0,
            "cephalosporins": 0.0,
            "carbapenems": 0.0,
            "fluoroquinolones": 0.0,
            "aminoglycosides": 0.0,
        }

        for pred in predictions:
            tier = pred.get("risk_tier", "LOW")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            total_score += pred.get("amr_risk_score", 0)

            if pred.get("low_confidence_flag", False):
                low_conf_count += 1

            abx_risk = pred.get("antibiotic_class_risk", {})
            if isinstance(abx_risk, dict):
                for cls_name, risk_val in abx_risk.items():
                    if cls_name in antibiotic_totals and isinstance(risk_val, (int, float)):
                        antibiotic_totals[cls_name] += float(risk_val)

        high_risk_count = tier_counts.get("HIGH", 0) + tier_counts.get("CRITICAL", 0)
        high_risk_rate = high_risk_count / total if total > 0 else 0.0

        avg_abx = {k: v / total for k, v in antibiotic_totals.items()}
        top_abx = dict(sorted(avg_abx.items(), key=lambda x: x[1], reverse=True))

        recommendations = self._generate_recommendations(
            high_risk_rate, low_conf_count / total if total > 0 else 0.0, top_abx
        )

        return {
            "total_patients": total,
            "risk_distribution": tier_counts,
            "avg_risk_score": round(total_score / total, 2),
            "low_confidence_count": low_conf_count,
            "high_risk_rate": round(high_risk_rate, 4),
            "top_antibiotic_risks": {k: round(v, 4) for k, v in top_abx.items()},
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self,
        high_risk_rate: float,
        low_conf_rate: float,
        top_abx: dict[str, float],
    ) -> list[str]:
        """Generate actionable recommendations based on thresholds."""
        recs: list[str] = []

        if high_risk_rate > 0.20:
            recs.append(
                f"ALERT: {high_risk_rate:.0%} of patients are HIGH/CRITICAL risk. "
                "Review empiric antibiotic protocols."
            )

        if low_conf_rate > 0.15:
            recs.append(
                f"DATA QUALITY: {low_conf_rate:.0%} of predictions have low confidence. "
                "Investigate data completeness."
            )

        highest_abx = max(top_abx.items(), key=lambda x: x[1], default=("", 0.0))
        if highest_abx[1] > 0.5:
            recs.append(
                f"RESISTANCE TREND: {highest_abx[0]} shows elevated resistance risk "
                f"({highest_abx[1]:.2f}). Consider stewardship intervention."
            )

        if not recs:
            recs.append("No critical alerts. Continue routine monitoring.")

        return recs

    def _generate_csv(self, predictions: list[dict[str, Any]], summary: dict[str, Any]) -> str:
        """Generate CSV report content."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Patients", summary["total_patients"]])
        writer.writerow(["Average Risk Score", summary["avg_risk_score"]])
        writer.writerow(["High Risk Rate", f"{summary['high_risk_rate']:.2%}"])
        writer.writerow(["Low Confidence Count", summary["low_confidence_count"]])
        writer.writerow([])

        writer.writerow(["Risk Tier", "Count"])
        for tier, count in summary["risk_distribution"].items():
            writer.writerow([tier, count])
        writer.writerow([])

        writer.writerow(["Antibiotic Class", "Avg Risk"])
        for cls_name, risk in summary["top_antibiotic_risks"].items():
            writer.writerow([cls_name, f"{risk:.4f}"])

        return output.getvalue()

    def _generate_json(
        self, summary: dict[str, Any], period_start: str, period_end: str, report_id: str
    ) -> str:
        """Generate JSON report content."""
        report_data = {
            "report_id": report_id,
            "period": {"start": period_start, "end": period_end},
            "summary": summary,
        }
        return json.dumps(report_data, indent=2)
