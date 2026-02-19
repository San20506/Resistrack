"""Tests for M5.4 – Monitoring and alerting."""

from __future__ import annotations

import time

import pytest

from resistrack.monitoring.alerting import (
    Alarm,
    AlarmConfig,
    AlarmSeverity,
    AlarmState,
    DashboardMetric,
    DashboardPanel,
    MetricDatapoint,
    MonitoringDashboard,
    MonitoringService,
    SNSNotification,
)


# --- Enum Tests ---


def test_alarm_severity_enum():
    assert AlarmSeverity.LOW == "low"
    assert AlarmSeverity.MEDIUM == "medium"
    assert AlarmSeverity.HIGH == "high"
    assert AlarmSeverity.CRITICAL == "critical"


def test_alarm_state_enum():
    assert AlarmState.OK == "OK"
    assert AlarmState.ALARM == "ALARM"
    assert AlarmState.INSUFFICIENT_DATA == "INSUFFICIENT_DATA"


# --- AlarmConfig Tests ---


def test_alarm_config_defaults():
    cfg = AlarmConfig()
    assert cfg.latency_p95_threshold_ms == 1500.0
    assert cfg.error_rate_threshold_pct == 1.0
    assert cfg.psi_drift_threshold == 0.20
    assert cfg.evaluation_periods == 3
    assert cfg.datapoints_to_alarm == 2


def test_alarm_config_custom():
    cfg = AlarmConfig(latency_p95_threshold_ms=2000.0, error_rate_threshold_pct=0.5)
    assert cfg.latency_p95_threshold_ms == 2000.0
    assert cfg.error_rate_threshold_pct == 0.5


# --- Alarm Tests ---


def test_alarm_creation():
    alarm = Alarm(
        alarm_name="test-alarm",
        metric_name="latency",
        threshold=1500.0,
        comparison="GreaterThan",
        severity=AlarmSeverity.HIGH,
    )
    assert alarm.state == AlarmState.INSUFFICIENT_DATA
    assert alarm.evaluation_count == 0


# --- MetricDatapoint Tests ---


def test_metric_datapoint():
    dp = MetricDatapoint(metric_name="inference_latency_ms", value=1200.0, unit="ms")
    assert dp.metric_name == "inference_latency_ms"
    assert dp.value == 1200.0


# --- MonitoringDashboard Tests ---


def test_dashboard_record_metric():
    dashboard = MonitoringDashboard()
    dp = MetricDatapoint(metric_name="inference_latency_ms", value=1200.0)
    dashboard.record_metric(dp)
    data = dashboard.get_metric_data("inference_latency_ms")
    assert len(data) == 1
    assert data[0].value == 1200.0


def test_dashboard_p95():
    dashboard = MonitoringDashboard()
    for v in [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]:
        dashboard.record_metric(MetricDatapoint(metric_name="latency", value=v))
    p95 = dashboard.compute_p95("latency")
    assert p95 > 900.0


def test_dashboard_panels():
    dashboard = MonitoringDashboard()
    for v in [1200.0, 1300.0, 1400.0]:
        dashboard.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=v))
    panels = dashboard.get_panels()
    assert len(panels) >= 1
    assert panels[0].title == "Inference Latency"


# --- MonitoringService Tests ---


def test_service_creation():
    service = MonitoringService()
    alarms = service.get_alarms()
    assert "latency_p95" in alarms
    assert "error_rate" in alarms
    assert "psi_drift" in alarms


def test_service_default_alarms():
    service = MonitoringService()
    alarms = service.get_alarms()
    assert len(alarms) == 3
    assert alarms["latency_p95"].severity == AlarmSeverity.HIGH
    assert alarms["error_rate"].severity == AlarmSeverity.CRITICAL


def test_service_record_metric_no_alarm():
    service = MonitoringService()
    dp = MetricDatapoint(metric_name="inference_latency_ms", value=500.0)
    service.record_metric(dp)
    states = service.get_alarm_states()
    assert states["latency_p95"] == AlarmState.OK


def test_service_alarm_triggers():
    service = MonitoringService()
    # Need 2 breaching datapoints to trigger (datapoints_to_alarm=2)
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    states = service.get_alarm_states()
    assert states["latency_p95"] == AlarmState.ALARM


def test_service_alarm_single_breach_no_trigger():
    service = MonitoringService()
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    states = service.get_alarm_states()
    # After 1 breach, not yet ALARM (need 2)
    assert states["latency_p95"] != AlarmState.ALARM


def test_service_alarm_resets():
    service = MonitoringService()
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    assert service.get_alarm_states()["latency_p95"] == AlarmState.ALARM
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=500.0))
    assert service.get_alarm_states()["latency_p95"] == AlarmState.OK


def test_service_notification_on_alarm():
    service = MonitoringService()
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    notifications = service.get_notifications()
    assert len(notifications) >= 1
    assert isinstance(notifications[0], SNSNotification)


def test_service_notifications_no_phi():
    service = MonitoringService()
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=2000.0))
    for n in service.get_notifications():
        assert "patient" not in n.message.lower() or "phi" not in n.message.lower()
        assert "No PHI" in n.message


def test_service_error_rate_alarm():
    service = MonitoringService()
    service.record_metric(MetricDatapoint(metric_name="error_rate", value=5.0))
    service.record_metric(MetricDatapoint(metric_name="error_rate", value=5.0))
    states = service.get_alarm_states()
    assert states["error_rate"] == AlarmState.ALARM


def test_service_psi_drift_alarm():
    service = MonitoringService()
    service.record_metric(MetricDatapoint(metric_name="psi_drift_score", value=0.30))
    service.record_metric(MetricDatapoint(metric_name="psi_drift_score", value=0.30))
    states = service.get_alarm_states()
    assert states["psi_drift"] == AlarmState.ALARM


def test_service_stats():
    service = MonitoringService()
    service.record_metric(MetricDatapoint(metric_name="inference_latency_ms", value=500.0))
    stats = service.get_stats()
    assert stats["alarms_total"] == 3
    assert stats["dashboard_metrics"] >= 1


def test_sns_notification_format():
    notif = SNSNotification(
        topic_arn="arn:aws:sns:us-east-1:123:test",
        subject="[high] test-alarm",
        message="Test alert message",
        alarm_name="test-alarm",
        severity=AlarmSeverity.HIGH,
    )
    assert notif.severity == AlarmSeverity.HIGH
    assert "test-alarm" in notif.subject


def test_dashboard_panel_creation():
    panel = DashboardPanel(
        title="Test Panel",
        panel_type="line_chart",
        metrics=[
            DashboardMetric(name="latency", current_value=1200.0, unit="ms", trend="stable"),
        ],
    )
    assert panel.title == "Test Panel"
    assert len(panel.metrics) == 1
    assert panel.metrics[0].unit == "ms"


def test_dashboard_summary():
    dashboard = MonitoringDashboard("TestDash")
    assert dashboard.name == "TestDash"
    summary = dashboard.get_summary()
    assert summary["name"] == "TestDash"
    assert summary["metric_count"] == 0
