"""Tests for M3.4 – Notification dispatch."""

from __future__ import annotations

import time

import pytest

from resistrack.notifications.dispatch import (
    AlertPriority,
    NotificationChannel,
    NotificationConfig,
    NotificationDispatcher,
    NotificationEvent,
    NotificationResult,
    RoutingRule,
)


# --- Helpers ---


def _high_event(mdro: bool = False) -> NotificationEvent:
    return NotificationEvent(
        patient_token="pt_tok_123",
        risk_tier="HIGH",
        amr_risk_score=65.0,
        mdro_flag=mdro,
        alert_id="alert_001",
        model_version="3.4.0",
    )


def _critical_event(mdro: bool = False) -> NotificationEvent:
    return NotificationEvent(
        patient_token="pt_tok_456",
        risk_tier="CRITICAL",
        amr_risk_score=85.0,
        mdro_flag=mdro,
        alert_id="alert_002",
        model_version="3.4.0",
    )


def _low_event() -> NotificationEvent:
    return NotificationEvent(
        patient_token="pt_tok_789",
        risk_tier="LOW",
        amr_risk_score=15.0,
    )


# --- Enum Tests ---


def test_channel_enum():
    assert NotificationChannel.SMS == "sms"
    assert NotificationChannel.EMAIL == "email"
    assert NotificationChannel.PAGERDUTY == "pagerduty"


def test_priority_enum():
    assert AlertPriority.HIGH == "HIGH"
    assert AlertPriority.CRITICAL == "CRITICAL"


# --- Event Tests ---


def test_event_is_high_or_critical():
    assert _high_event().is_high_or_critical is True
    assert _critical_event().is_high_or_critical is True
    assert _low_event().is_high_or_critical is False


def test_event_is_critical_mdro():
    assert _critical_event(mdro=True).is_critical_mdro is True
    assert _critical_event(mdro=False).is_critical_mdro is False
    assert _high_event(mdro=True).is_critical_mdro is False


# --- Config Tests ---


def test_config_defaults():
    cfg = NotificationConfig()
    assert NotificationChannel.EMAIL in cfg.pharmacy_channels
    assert NotificationChannel.SMS in cfg.pharmacy_channels
    assert NotificationChannel.PAGERDUTY in cfg.ic_officer_channels


# --- Dispatcher Tests ---


def test_dispatcher_creation():
    dispatcher = NotificationDispatcher()
    rules = dispatcher.get_routing_rules()
    assert len(rules) == 2


def test_dispatch_high_to_pharmacy():
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(_high_event())
    assert len(results) == 2  # EMAIL + SMS to pharmacy
    assert all(r.target_team == "pharmacy" for r in results)
    assert all(r.delivered is True for r in results)


def test_dispatch_critical_to_pharmacy():
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(_critical_event(mdro=False))
    assert len(results) == 2  # EMAIL + SMS to pharmacy only (no MDRO)
    assert all(r.target_team == "pharmacy" for r in results)


def test_dispatch_critical_mdro_to_both():
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(_critical_event(mdro=True))
    teams = {r.target_team for r in results}
    assert "pharmacy" in teams
    assert "infection_control" in teams
    assert len(results) == 4  # 2 pharmacy + 2 IC


def test_dispatch_low_no_notification():
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(_low_event())
    assert len(results) == 0


def test_dispatch_medium_no_notification():
    event = NotificationEvent(
        patient_token="pt_tok_med",
        risk_tier="MEDIUM",
        amr_risk_score=35.0,
    )
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(event)
    assert len(results) == 0


def test_dispatch_channels_high():
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(_high_event())
    channels = {r.channel for r in results}
    assert NotificationChannel.EMAIL in channels
    assert NotificationChannel.SMS in channels


def test_dispatch_channels_critical_mdro():
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(_critical_event(mdro=True))
    channels = {r.channel for r in results}
    assert NotificationChannel.PAGERDUTY in channels
    assert NotificationChannel.EMAIL in channels
    assert NotificationChannel.SMS in channels


def test_dispatch_result_has_message_id():
    dispatcher = NotificationDispatcher()
    results = dispatcher.dispatch(_high_event())
    for r in results:
        assert r.message_id.startswith("msg_")


def test_dispatch_batch():
    dispatcher = NotificationDispatcher()
    events = [_high_event(), _critical_event(mdro=True), _low_event()]
    batch_results = dispatcher.dispatch_batch(events)
    assert len(batch_results) == 3
    assert len(batch_results[0]) == 2   # HIGH -> pharmacy only
    assert len(batch_results[1]) == 4   # CRITICAL+MDRO -> pharmacy+IC
    assert len(batch_results[2]) == 0   # LOW -> nothing


def test_delivery_log():
    dispatcher = NotificationDispatcher()
    dispatcher.dispatch(_high_event())
    log = dispatcher.get_delivery_log()
    assert len(log) == 2
    for entry in log:
        assert "message_id" in entry
        assert "channel" in entry
        assert "payload" in entry


def test_payload_no_phi():
    dispatcher = NotificationDispatcher()
    dispatcher.dispatch(_high_event())
    log = dispatcher.get_delivery_log()
    for entry in log:
        payload = entry["payload"]
        assert dispatcher.verify_no_phi(payload)
        assert "patient_token" in payload
        assert "patient_name" not in payload
        assert "date_of_birth" not in payload
        assert "ssn" not in payload


def test_verify_no_phi_clean():
    dispatcher = NotificationDispatcher()
    payload = {"patient_token": "pt_123", "risk_tier": "HIGH"}
    assert dispatcher.verify_no_phi(payload) is True


def test_verify_no_phi_dirty():
    dispatcher = NotificationDispatcher()
    payload = {"patient_name": "John Doe", "risk_tier": "HIGH"}
    assert dispatcher.verify_no_phi(payload) is False


def test_get_results():
    dispatcher = NotificationDispatcher()
    dispatcher.dispatch(_high_event())
    results = dispatcher.get_results()
    assert len(results) == 2


def test_stats():
    dispatcher = NotificationDispatcher()
    dispatcher.dispatch(_high_event())
    dispatcher.dispatch(_critical_event(mdro=True))
    stats = dispatcher.get_stats()
    assert stats["total_notifications"] == 6
    assert stats["delivered"] == 6
    assert stats["failed"] == 0
    assert "pharmacy" in stats["by_team"]
    assert "infection_control" in stats["by_team"]


def test_custom_config():
    cfg = NotificationConfig(
        pharmacy_channels=(NotificationChannel.EMAIL,),
        ic_officer_channels=(NotificationChannel.PAGERDUTY,),
    )
    dispatcher = NotificationDispatcher(cfg)
    results = dispatcher.dispatch(_high_event())
    assert len(results) == 1
    assert results[0].channel == NotificationChannel.EMAIL


def test_routing_rules():
    dispatcher = NotificationDispatcher()
    rules = dispatcher.get_routing_rules()
    pharmacy_rule = rules[0]
    ic_rule = rules[1]
    assert pharmacy_rule.target_team == "pharmacy"
    assert ic_rule.target_team == "infection_control"
    assert ic_rule.requires_mdro is True
