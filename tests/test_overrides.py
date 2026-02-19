"""Tests for M3.3 – Override monitoring."""

from __future__ import annotations

import time

import pytest

from resistrack.overrides.monitor import (
    MonitorConfig,
    OverrideEvent,
    OverrideMonitor,
    OverrideRate,
    OverrideReport,
)


# --- Helpers ---


def _event(clinician_id: str, action: str, reason: str | None = None) -> OverrideEvent:
    return OverrideEvent(
        clinician_id=clinician_id,
        patient_token="pt_test",
        alert_id=f"alert_{time.time()}",
        action=action,
        override_reason=reason,
    )


def _many_events(cid: str, overrides: int, acks: int) -> list[OverrideEvent]:
    events = []
    for _ in range(overrides):
        events.append(_event(cid, "override", "Clinical judgment"))
    for _ in range(acks):
        events.append(_event(cid, "acknowledged"))
    return events


# --- Config Tests ---


def test_config_defaults():
    cfg = MonitorConfig()
    assert cfg.rolling_window_days == 30
    assert cfg.override_rate_threshold == 0.60
    assert cfg.min_alerts_for_report == 5


def test_config_custom():
    cfg = MonitorConfig(rolling_window_days=7, override_rate_threshold=0.50)
    assert cfg.rolling_window_days == 7
    assert cfg.override_rate_threshold == 0.50


# --- OverrideEvent Tests ---


def test_event_is_override():
    event = _event("doc1", "override")
    assert event.is_override is True


def test_event_is_not_override():
    event = _event("doc1", "acknowledged")
    assert event.is_override is False


def test_event_escalate():
    event = _event("doc1", "escalate")
    assert event.is_override is False


def test_event_with_reason():
    event = _event("doc1", "override", "Patient allergy")
    assert event.override_reason == "Patient allergy"


# --- OverrideMonitor Tests ---


def test_monitor_creation():
    monitor = OverrideMonitor()
    assert monitor.config.rolling_window_days == 30


def test_record_event():
    monitor = OverrideMonitor()
    monitor.record_event(_event("doc1", "override"))
    stats = monitor.get_stats()
    assert stats["total_events"] == 1


def test_record_events_batch():
    monitor = OverrideMonitor()
    events = _many_events("doc1", 3, 2)
    monitor.record_events(events)
    stats = monitor.get_stats()
    assert stats["total_events"] == 5


def test_compute_rates_single_clinician():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=1))
    events = _many_events("doc1", 4, 6)
    monitor.record_events(events)
    rates = monitor.compute_rates()
    assert len(rates) == 1
    assert rates[0].clinician_id == "doc1"
    assert rates[0].override_count == 4
    assert rates[0].total_alerts == 10
    assert rates[0].rate == pytest.approx(0.4)
    assert rates[0].exceeds_threshold is False


def test_compute_rates_exceeds_threshold():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=1))
    events = _many_events("doc1", 8, 2)
    monitor.record_events(events)
    rates = monitor.compute_rates()
    assert rates[0].rate == pytest.approx(0.8)
    assert rates[0].exceeds_threshold is True


def test_compute_rates_multiple_clinicians():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=1))
    monitor.record_events(_many_events("doc1", 1, 9))
    monitor.record_events(_many_events("doc2", 8, 2))
    rates = monitor.compute_rates()
    assert len(rates) == 2
    doc1 = next(r for r in rates if r.clinician_id == "doc1")
    doc2 = next(r for r in rates if r.clinician_id == "doc2")
    assert doc1.exceeds_threshold is False
    assert doc2.exceeds_threshold is True


def test_min_alerts_respected():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=10))
    events = _many_events("doc1", 4, 1)  # 5 total, under threshold of 10
    monitor.record_events(events)
    rates = monitor.compute_rates()
    # Rate is 80% but only 5 alerts, under min of 10
    assert rates[0].exceeds_threshold is False


def test_generate_report_no_flagged():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=1))
    monitor.record_events(_many_events("doc1", 1, 9))
    report = monitor.generate_report()
    assert isinstance(report, OverrideReport)
    assert len(report.flagged_clinicians) == 0
    assert report.requires_review is False


def test_generate_report_flagged():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=1))
    monitor.record_events(_many_events("doc1", 8, 2))
    report = monitor.generate_report()
    assert len(report.flagged_clinicians) == 1
    assert "doc1" in report.flagged_clinicians
    assert report.requires_review is True


def test_report_metadata():
    monitor = OverrideMonitor()
    report = monitor.generate_report()
    assert report.window_days == 30
    assert report.threshold == 0.60
    assert report.report_timestamp > 0


def test_get_clinician_rate():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=1))
    monitor.record_events(_many_events("doc1", 5, 5))
    rate = monitor.get_clinician_rate("doc1")
    assert rate is not None
    assert rate.rate == pytest.approx(0.5)


def test_get_clinician_rate_missing():
    monitor = OverrideMonitor()
    rate = monitor.get_clinician_rate("nonexistent")
    assert rate is None


def test_report_history():
    monitor = OverrideMonitor()
    monitor.generate_report()
    monitor.generate_report()
    history = monitor.get_report_history()
    assert len(history) == 2


def test_stats():
    monitor = OverrideMonitor(MonitorConfig(min_alerts_for_report=1))
    monitor.record_events(_many_events("doc1", 8, 2))
    monitor.record_events(_many_events("doc2", 1, 9))
    stats = monitor.get_stats()
    assert stats["total_events"] == 20
    assert stats["clinicians_tracked"] == 2
    assert stats["flagged_clinicians"] == 1


def test_empty_monitor():
    monitor = OverrideMonitor()
    rates = monitor.compute_rates()
    assert len(rates) == 0
    report = monitor.generate_report()
    assert len(report.rates) == 0
    assert not report.requires_review
