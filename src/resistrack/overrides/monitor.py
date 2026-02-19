"""M3.3 – Clinician override rate monitoring.

Tracks clinician override rates over a rolling 30-day window.
If a clinician's override rate exceeds 60%, auto-generates a
model feedback report for clinical informatics review.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# --- Configuration ---


@dataclass(frozen=True)
class MonitorConfig:
    """Configuration for override monitoring."""

    rolling_window_days: int = 30
    override_rate_threshold: float = 0.60
    min_alerts_for_report: int = 5


# --- Data Models ---


@dataclass(frozen=True)
class OverrideEvent:
    """A single clinician response to a CDS alert."""

    clinician_id: str
    patient_token: str
    alert_id: str
    action: str  # "acknowledged", "override", "escalate"
    override_reason: str | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def is_override(self) -> bool:
        return self.action == "override"


@dataclass(frozen=True)
class OverrideRate:
    """Override rate for a single clinician."""

    clinician_id: str
    override_count: int
    total_alerts: int
    rate: float
    exceeds_threshold: bool
    window_start: float
    window_end: float


@dataclass(frozen=True)
class OverrideReport:
    """Report of override rates across clinicians."""

    rates: tuple[OverrideRate, ...]
    flagged_clinicians: tuple[str, ...]
    report_timestamp: float
    window_days: int
    threshold: float
    requires_review: bool


# --- Override Monitor ---


class OverrideMonitor:
    """Monitors clinician override rates and generates reports."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig()
        self._events: list[OverrideEvent] = []
        self._reports: list[OverrideReport] = []

    @property
    def config(self) -> MonitorConfig:
        return self._config

    def record_event(self, event: OverrideEvent) -> None:
        """Record a clinician response event."""
        self._events.append(event)

    def record_events(self, events: list[OverrideEvent]) -> None:
        """Record multiple events at once."""
        self._events.extend(events)

    def _get_window_events(self) -> list[OverrideEvent]:
        """Get events within the rolling window."""
        window_seconds = self._config.rolling_window_days * 86400
        cutoff = time.time() - window_seconds
        return [e for e in self._events if e.timestamp >= cutoff]

    def compute_rates(self) -> list[OverrideRate]:
        """Compute override rates per clinician within the rolling window."""
        window_events = self._get_window_events()
        if not window_events:
            return []

        clinician_events: dict[str, list[OverrideEvent]] = defaultdict(list)
        for event in window_events:
            clinician_events[event.clinician_id].append(event)

        now = time.time()
        window_start = now - self._config.rolling_window_days * 86400
        rates: list[OverrideRate] = []

        for cid, events in sorted(clinician_events.items()):
            total = len(events)
            overrides = sum(1 for e in events if e.is_override)
            rate = overrides / total if total > 0 else 0.0

            rates.append(OverrideRate(
                clinician_id=cid,
                override_count=overrides,
                total_alerts=total,
                rate=rate,
                exceeds_threshold=(
                    rate > self._config.override_rate_threshold
                    and total >= self._config.min_alerts_for_report
                ),
                window_start=window_start,
                window_end=now,
            ))

        return rates

    def generate_report(self) -> OverrideReport:
        """Generate an override monitoring report."""
        rates = self.compute_rates()
        flagged = [r.clinician_id for r in rates if r.exceeds_threshold]

        report = OverrideReport(
            rates=tuple(rates),
            flagged_clinicians=tuple(flagged),
            report_timestamp=time.time(),
            window_days=self._config.rolling_window_days,
            threshold=self._config.override_rate_threshold,
            requires_review=len(flagged) > 0,
        )

        self._reports.append(report)
        return report

    def get_clinician_rate(self, clinician_id: str) -> OverrideRate | None:
        """Get override rate for a specific clinician."""
        rates = self.compute_rates()
        for rate in rates:
            if rate.clinician_id == clinician_id:
                return rate
        return None

    def get_report_history(self) -> list[OverrideReport]:
        """Get all generated reports."""
        return list(self._reports)

    def get_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        window_events = self._get_window_events()
        rates = self.compute_rates()
        return {
            "total_events": len(self._events),
            "window_events": len(window_events),
            "clinicians_tracked": len(rates),
            "flagged_clinicians": sum(1 for r in rates if r.exceeds_threshold),
            "reports_generated": len(self._reports),
            "config": {
                "window_days": self._config.rolling_window_days,
                "threshold": self._config.override_rate_threshold,
                "min_alerts": self._config.min_alerts_for_report,
            },
        }
