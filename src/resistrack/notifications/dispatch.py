"""M3.4 – Notification dispatch service.

Routes HIGH and CRITICAL AMR alerts to appropriate teams via SNS.
CRITICAL alerts with MDRO flags also notify Infection Control.
Payloads contain only patient_token (no PHI).
Supports SMS, Email, and PagerDuty channels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# --- Enums ---


class NotificationChannel(StrEnum):
    """Supported notification channels."""

    SMS = "sms"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"


class AlertPriority(StrEnum):
    """Alert priority levels."""

    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# --- Configuration ---


@dataclass(frozen=True)
class NotificationConfig:
    """Configuration for notification dispatch."""

    pharmacy_channels: tuple[NotificationChannel, ...] = (
        NotificationChannel.EMAIL,
        NotificationChannel.SMS,
    )
    ic_officer_channels: tuple[NotificationChannel, ...] = (
        NotificationChannel.PAGERDUTY,
        NotificationChannel.EMAIL,
    )
    sns_topic_arn: str = "arn:aws:sns:us-east-1:123456789:resistrack-alerts"
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass(frozen=True)
class RoutingRule:
    """A rule for routing notifications."""

    name: str
    risk_tiers: tuple[str, ...]
    requires_mdro: bool
    target_team: str
    channels: tuple[NotificationChannel, ...]
    priority: AlertPriority


# --- Data Models ---


@dataclass(frozen=True)
class NotificationEvent:
    """An AMR alert event to be dispatched."""

    patient_token: str
    risk_tier: str
    amr_risk_score: float
    mdro_flag: bool = False
    alert_id: str = ""
    model_version: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def is_critical_mdro(self) -> bool:
        return self.risk_tier == "CRITICAL" and self.mdro_flag

    @property
    def is_high_or_critical(self) -> bool:
        return self.risk_tier in ("HIGH", "CRITICAL")


@dataclass(frozen=True)
class NotificationResult:
    """Result of dispatching a notification."""

    event: NotificationEvent
    target_team: str
    channel: NotificationChannel
    delivered: bool
    delivery_timestamp: float
    message_id: str
    retry_count: int = 0
    error: str | None = None


# --- Notification Dispatcher ---


class NotificationDispatcher:
    """Dispatches AMR risk alerts to appropriate teams.

    Routing rules:
    - HIGH/CRITICAL risk → Pharmacy team via Email+SMS
    - CRITICAL + MDRO flag → also Infection Control Officer via PagerDuty+Email
    - Payloads contain only patient_token, never PHI
    """

    def __init__(self, config: NotificationConfig | None = None) -> None:
        self._config = config or NotificationConfig()
        self._results: list[NotificationResult] = []
        self._routing_rules = self._build_default_rules()
        self._delivery_log: list[dict[str, Any]] = []

    @property
    def config(self) -> NotificationConfig:
        return self._config

    def _build_default_rules(self) -> list[RoutingRule]:
        """Build default routing rules."""
        return [
            RoutingRule(
                name="pharmacy_high_critical",
                risk_tiers=("HIGH", "CRITICAL"),
                requires_mdro=False,
                target_team="pharmacy",
                channels=self._config.pharmacy_channels,
                priority=AlertPriority.HIGH,
            ),
            RoutingRule(
                name="ic_officer_critical_mdro",
                risk_tiers=("CRITICAL",),
                requires_mdro=True,
                target_team="infection_control",
                channels=self._config.ic_officer_channels,
                priority=AlertPriority.CRITICAL,
            ),
        ]

    def _matches_rule(self, event: NotificationEvent, rule: RoutingRule) -> bool:
        """Check if an event matches a routing rule."""
        if event.risk_tier not in rule.risk_tiers:
            return False
        if rule.requires_mdro and not event.mdro_flag:
            return False
        return True

    def _build_payload(
        self, event: NotificationEvent, rule: RoutingRule,
    ) -> dict[str, Any]:
        """Build notification payload with no PHI."""
        return {
            "patient_token": event.patient_token,
            "risk_tier": event.risk_tier,
            "amr_risk_score": event.amr_risk_score,
            "mdro_flag": event.mdro_flag,
            "alert_id": event.alert_id,
            "target_team": rule.target_team,
            "priority": rule.priority,
            "model_version": event.model_version,
            "timestamp": event.timestamp,
        }

    def _send_notification(
        self,
        event: NotificationEvent,
        rule: RoutingRule,
        channel: NotificationChannel,
    ) -> NotificationResult:
        """Simulate sending a notification through a channel."""
        payload = self._build_payload(event, rule)
        message_id = f"msg_{int(time.time() * 1000)}_{channel}"

        self._delivery_log.append({
            "message_id": message_id,
            "channel": channel,
            "target_team": rule.target_team,
            "payload": payload,
            "timestamp": time.time(),
        })

        return NotificationResult(
            event=event,
            target_team=rule.target_team,
            channel=channel,
            delivered=True,
            delivery_timestamp=time.time(),
            message_id=message_id,
        )

    def dispatch(self, event: NotificationEvent) -> list[NotificationResult]:
        """Dispatch notifications for an AMR alert event.

        Returns a list of notification results for each matched rule+channel.
        Only HIGH and CRITICAL risk tiers trigger notifications.
        """
        if not event.is_high_or_critical:
            return []

        results: list[NotificationResult] = []
        for rule in self._routing_rules:
            if not self._matches_rule(event, rule):
                continue
            for channel in rule.channels:
                result = self._send_notification(event, rule, channel)
                results.append(result)
                self._results.append(result)

        return results

    def dispatch_batch(
        self, events: list[NotificationEvent],
    ) -> list[list[NotificationResult]]:
        """Dispatch notifications for a batch of events."""
        return [self.dispatch(event) for event in events]

    def get_delivery_log(self) -> list[dict[str, Any]]:
        """Get the delivery log."""
        return list(self._delivery_log)

    def get_results(self) -> list[NotificationResult]:
        """Get all dispatch results."""
        return list(self._results)

    def get_routing_rules(self) -> list[RoutingRule]:
        """Get current routing rules."""
        return list(self._routing_rules)

    def verify_no_phi(self, payload: dict[str, Any]) -> bool:
        """Verify a payload contains no PHI fields."""
        phi_fields = {
            "patient_name", "date_of_birth", "ssn", "mrn",
            "address", "phone", "email_address", "insurance_id",
        }
        return not bool(set(payload.keys()) & phi_fields)

    def get_stats(self) -> dict[str, Any]:
        """Get dispatch statistics."""
        total = len(self._results)
        delivered = sum(1 for r in self._results if r.delivered)
        by_team: dict[str, int] = {}
        by_channel: dict[str, int] = {}
        for r in self._results:
            by_team[r.target_team] = by_team.get(r.target_team, 0) + 1
            by_channel[r.channel] = by_channel.get(r.channel, 0) + 1

        return {
            "total_notifications": total,
            "delivered": delivered,
            "failed": total - delivered,
            "by_team": by_team,
            "by_channel": by_channel,
            "routing_rules": len(self._routing_rules),
        }
