"""Monitoring and alerting service.

CloudWatch-style dashboards, metric alarms, and SNS notifications.
No PHI in any monitoring data — only de-identified tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from resistrack.common.constants import RANDOM_STATE


# --- Enums ---


class AlarmState(StrEnum):
    """Alarm state values."""

    OK = "OK"
    ALARM = "ALARM"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AlarmSeverity(StrEnum):
    """Alarm severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --- Data Classes ---


@dataclass(frozen=True)
class AlarmConfig:
    """Configuration for a monitoring alarm."""

    latency_p95_threshold_ms: float = 1500.0
    error_rate_threshold_pct: float = 1.0
    psi_drift_threshold: float = 0.20
    evaluation_periods: int = 3
    datapoints_to_alarm: int = 2
    sns_topic_arn: str = "arn:aws:sns:us-east-1:*:resistrack-alerts"


@dataclass(frozen=True)
class MetricDatapoint:
    """A single metric observation."""

    metric_name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    dimensions: dict[str, str] = field(default_factory=dict)
    unit: str = "None"


@dataclass(frozen=True)
class Alarm:
    """A monitoring alarm."""

    alarm_name: str
    metric_name: str
    threshold: float
    comparison: str  # GreaterThan, LessThan
    severity: AlarmSeverity
    state: AlarmState = AlarmState.INSUFFICIENT_DATA
    last_state_change: float = field(default_factory=time.time)
    evaluation_count: int = 0


@dataclass(frozen=True)
class SNSNotification:
    """SNS notification payload (no PHI)."""

    topic_arn: str
    subject: str
    message: str
    alarm_name: str
    severity: AlarmSeverity
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class DashboardMetric:
    """A metric displayed on a dashboard panel."""

    name: str
    current_value: float
    unit: str
    trend: str  # up, down, stable


@dataclass(frozen=True)
class DashboardPanel:
    """A dashboard panel with metrics."""

    title: str
    panel_type: str  # line_chart, gauge, number
    metrics: list[DashboardMetric]


# --- Monitoring Dashboard ---


class MonitoringDashboard:
    """CloudWatch-style monitoring dashboard."""

    def __init__(self, name: str = "ResisTrack-Production") -> None:
        self._name = name
        self._metrics: dict[str, list[MetricDatapoint]] = {}

    @property
    def name(self) -> str:
        return self._name

    def record_metric(self, datapoint: MetricDatapoint) -> None:
        """Record a metric datapoint."""
        if datapoint.metric_name not in self._metrics:
            self._metrics[datapoint.metric_name] = []
        self._metrics[datapoint.metric_name].append(datapoint)

    def get_metric_data(
        self,
        metric_name: str,
        last_n: int | None = None,
    ) -> list[MetricDatapoint]:
        """Retrieve metric datapoints."""
        data = self._metrics.get(metric_name, [])
        if last_n is not None:
            return data[-last_n:]
        return list(data)

    def compute_p95(self, metric_name: str, last_n: int = 100) -> float:
        """Compute p95 of a metric."""
        data = self.get_metric_data(metric_name, last_n)
        if not data:
            return 0.0
        values = [d.value for d in data]
        return float(np.percentile(values, 95))

    def compute_average(self, metric_name: str, last_n: int = 100) -> float:
        """Compute average of a metric."""
        data = self.get_metric_data(metric_name, last_n)
        if not data:
            return 0.0
        return float(np.mean([d.value for d in data]))

    def get_panels(self) -> list[DashboardPanel]:
        """Generate dashboard panels from current metrics."""
        panels: list[DashboardPanel] = []

        # Latency panel
        latency_data = self.get_metric_data("inference_latency_ms", 10)
        if latency_data:
            p95 = self.compute_p95("inference_latency_ms")
            avg = self.compute_average("inference_latency_ms")
            panels.append(
                DashboardPanel(
                    title="Inference Latency",
                    panel_type="line_chart",
                    metrics=[
                        DashboardMetric(name="p95_latency", current_value=p95, unit="ms", trend="stable"),
                        DashboardMetric(name="avg_latency", current_value=avg, unit="ms", trend="stable"),
                    ],
                )
            )

        # Error rate panel
        error_data = self.get_metric_data("error_rate", 10)
        if error_data:
            avg_err = self.compute_average("error_rate")
            panels.append(
                DashboardPanel(
                    title="Error Rate",
                    panel_type="gauge",
                    metrics=[
                        DashboardMetric(name="error_rate", current_value=avg_err, unit="%", trend="stable"),
                    ],
                )
            )

        # SageMaker health panel
        health_data = self.get_metric_data("endpoint_health", 10)
        if health_data:
            latest = health_data[-1].value
            panels.append(
                DashboardPanel(
                    title="SageMaker Health",
                    panel_type="number",
                    metrics=[
                        DashboardMetric(name="health_score", current_value=latest, unit="score", trend="stable"),
                    ],
                )
            )

        return panels

    def get_summary(self) -> dict[str, Any]:
        """Dashboard summary."""
        return {
            "name": self._name,
            "metric_count": len(self._metrics),
            "total_datapoints": sum(len(v) for v in self._metrics.values()),
            "panel_count": len(self.get_panels()),
        }


# --- Monitoring Service ---


class MonitoringService:
    """Central monitoring service with alarms and notifications."""

    def __init__(self, config: AlarmConfig | None = None) -> None:
        self._config = config or AlarmConfig()
        self._dashboard = MonitoringDashboard()
        self._alarms: dict[str, Alarm] = {}
        self._notifications: list[SNSNotification] = []
        self._rng = np.random.RandomState(RANDOM_STATE)

        # Register default alarms
        self._register_default_alarms()

    @property
    def config(self) -> AlarmConfig:
        return self._config

    @property
    def dashboard(self) -> MonitoringDashboard:
        return self._dashboard

    def _register_default_alarms(self) -> None:
        """Register standard monitoring alarms."""
        self._alarms["latency_p95"] = Alarm(
            alarm_name="resistrack-latency-p95",
            metric_name="inference_latency_ms",
            threshold=self._config.latency_p95_threshold_ms,
            comparison="GreaterThan",
            severity=AlarmSeverity.HIGH,
        )
        self._alarms["error_rate"] = Alarm(
            alarm_name="resistrack-error-rate",
            metric_name="error_rate",
            threshold=self._config.error_rate_threshold_pct,
            comparison="GreaterThan",
            severity=AlarmSeverity.CRITICAL,
        )
        self._alarms["psi_drift"] = Alarm(
            alarm_name="resistrack-psi-drift",
            metric_name="psi_drift_score",
            threshold=self._config.psi_drift_threshold,
            comparison="GreaterThan",
            severity=AlarmSeverity.HIGH,
        )

    def record_metric(self, datapoint: MetricDatapoint) -> None:
        """Record a metric and evaluate alarms."""
        self._dashboard.record_metric(datapoint)
        self._evaluate_alarms(datapoint)

    def _evaluate_alarms(self, datapoint: MetricDatapoint) -> None:
        """Evaluate alarms for a new datapoint."""
        for alarm_key, alarm in list(self._alarms.items()):
            if alarm.metric_name != datapoint.metric_name:
                continue

            # Check threshold
            if alarm.comparison == "GreaterThan":
                breaching = datapoint.value > alarm.threshold
            else:
                breaching = datapoint.value < alarm.threshold

            new_count = alarm.evaluation_count + (1 if breaching else 0)

            if breaching and new_count >= self._config.datapoints_to_alarm:
                new_state = AlarmState.ALARM
                if alarm.state != AlarmState.ALARM:
                    self._send_notification(alarm, datapoint)
            elif not breaching:
                new_state = AlarmState.OK
                new_count = 0
            else:
                new_state = alarm.state
                if new_state == AlarmState.INSUFFICIENT_DATA:
                    new_state = AlarmState.OK

            self._alarms[alarm_key] = Alarm(
                alarm_name=alarm.alarm_name,
                metric_name=alarm.metric_name,
                threshold=alarm.threshold,
                comparison=alarm.comparison,
                severity=alarm.severity,
                state=new_state,
                last_state_change=time.time() if new_state != alarm.state else alarm.last_state_change,
                evaluation_count=new_count,
            )

    def _send_notification(self, alarm: Alarm, datapoint: MetricDatapoint) -> None:
        """Send SNS notification for alarm state change."""
        notification = SNSNotification(
            topic_arn=self._config.sns_topic_arn,
            subject=f"[{alarm.severity.upper()}] {alarm.alarm_name}",
            message=(
                f"Alarm '{alarm.alarm_name}' triggered. "
                f"Metric '{alarm.metric_name}' = {datapoint.value:.2f} "
                f"(threshold: {alarm.threshold:.2f}). "
                f"No PHI included — de-identified data only."
            ),
            alarm_name=alarm.alarm_name,
            severity=alarm.severity,
        )
        self._notifications.append(notification)

    def get_alarm_states(self) -> dict[str, AlarmState]:
        """Return current alarm states."""
        return {k: v.state for k, v in self._alarms.items()}

    def get_notifications(self) -> list[SNSNotification]:
        """Return sent notifications."""
        return list(self._notifications)

    def get_alarms(self) -> dict[str, Alarm]:
        """Return all alarms."""
        return dict(self._alarms)

    def get_stats(self) -> dict[str, Any]:
        """Monitoring statistics."""
        states = self.get_alarm_states()
        return {
            "alarms_total": len(self._alarms),
            "alarms_in_alarm": sum(1 for s in states.values() if s == AlarmState.ALARM),
            "alarms_ok": sum(1 for s in states.values() if s == AlarmState.OK),
            "notifications_sent": len(self._notifications),
            "dashboard_metrics": self._dashboard.get_summary()["metric_count"],
        }
