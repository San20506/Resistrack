"""CDS Hooks service for clinical decision support.

Implements HL7 CDS Hooks for patient-view, order-sign, and
encounter-discharge triggers with cached fallback on timeout.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

import numpy as np

from resistrack.common.constants import (
    ANTIBIOTIC_CLASSES,
    CONFIDENCE_THRESHOLD,
    RANDOM_STATE,
    RISK_TIER_RANGES,
    RiskTier,
)
from resistrack.common.schemas import AMRPredictionOutput, SHAPFeature


# --- Enums ---


class HookType(StrEnum):
    """CDS Hook trigger types."""

    PATIENT_VIEW = "patient-view"
    ORDER_SIGN = "order-sign"
    ENCOUNTER_DISCHARGE = "encounter-discharge"


class ResponseAction(StrEnum):
    """Clinician response actions for CDS cards."""

    ACKNOWLEDGED = "acknowledged"
    OVERRIDE = "override"
    ESCALATE = "escalate"


# --- Data Classes ---


@dataclass(frozen=True)
class HookConfig:
    """Configuration for CDS Hooks service."""

    timeout_ms: float = 1500.0
    sla_ms: float = 2000.0
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 1000
    max_shap_factors: int = 3
    audit_enabled: bool = True


@dataclass(frozen=True)
class CDSCard:
    """A CDS decision support card."""

    summary: str
    detail: str
    indicator: str  # info, warning, critical
    source_label: str = "ResisTrack AMR"
    risk_tier: str = ""
    confidence_score: float = 0.0
    shap_factors: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    suggestion_actions: tuple[str, ...] = (
        ResponseAction.ACKNOWLEDGED,
        ResponseAction.OVERRIDE,
        ResponseAction.ESCALATE,
    )


@dataclass(frozen=True)
class CDSHookRequest:
    """Incoming CDS Hook request."""

    hook_type: HookType
    patient_token: str
    encounter_id: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CDSHookResponse:
    """Response to a CDS Hook request."""

    cards: list[CDSCard]
    hook_type: HookType
    latency_ms: float
    from_cache: bool
    request_id: str


@dataclass(frozen=True)
class AuditEntry:
    """Audit trail entry for CDS interactions."""

    request_id: str
    hook_type: HookType
    patient_token: str
    action: ResponseAction | None
    override_reason: str | None
    timestamp: float
    response_latency_ms: float
    risk_tier: str
    cards_count: int


# --- CDS Hooks Service ---


class CDSHooksService:
    """CDS Hooks service with timeout fallback and audit logging."""

    def __init__(
        self,
        predictor: Any,
        config: HookConfig | None = None,
    ) -> None:
        self._predictor = predictor
        self._config = config or HookConfig()
        self._cache: OrderedDict[str, tuple[CDSHookResponse, float]] = OrderedDict()
        self._audit_log: list[AuditEntry] = []
        self._rng = np.random.RandomState(RANDOM_STATE)
        self._request_counter = 0

    @property
    def config(self) -> HookConfig:
        return self._config

    def _generate_request_id(self) -> str:
        self._request_counter += 1
        raw = f"cds-{self._request_counter}-{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _cache_key(hook_type: HookType, patient_token: str) -> str:
        return f"{hook_type}:{patient_token}"

    def _get_cached(self, key: str) -> CDSHookResponse | None:
        if key in self._cache:
            response, created_at = self._cache[key]
            if time.time() - created_at < self._config.cache_ttl_seconds:
                self._cache.move_to_end(key)
                return CDSHookResponse(
                    cards=response.cards,
                    hook_type=response.hook_type,
                    latency_ms=0.1,
                    from_cache=True,
                    request_id=response.request_id,
                )
            else:
                del self._cache[key]
        return None

    def _put_cache(self, key: str, response: CDSHookResponse) -> None:
        if len(self._cache) >= self._config.max_cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = (response, time.time())

    def _build_card(self, prediction: AMRPredictionOutput) -> CDSCard:
        """Build a CDS card from a prediction."""
        tier = prediction.risk_tier
        if tier in ("CRITICAL", "HIGH"):
            indicator = "critical"
        elif tier == "MEDIUM":
            indicator = "warning"
        else:
            indicator = "info"

        # Top-N SHAP factors in plain English
        shap_factors = tuple(
            f.human_readable
            for f in prediction.shap_top_features[: self._config.max_shap_factors]
        )

        # Stewardship recommendations
        recommendations = self._generate_recommendations(prediction)

        return CDSCard(
            summary=f"AMR Risk: {tier} ({prediction.amr_risk_score:.0f}/100)",
            detail=(
                f"Patient {prediction.patient_token} has {tier} risk for "
                f"antimicrobial resistance (score {prediction.amr_risk_score:.1f}, "
                f"confidence {prediction.confidence_score:.0%})."
            ),
            indicator=indicator,
            risk_tier=tier,
            confidence_score=prediction.confidence_score,
            shap_factors=shap_factors,
            recommendations=recommendations,
        )

    @staticmethod
    def _generate_recommendations(prediction: AMRPredictionOutput) -> tuple[str, ...]:
        """Generate stewardship recommendations based on risk profile."""
        recs: list[str] = []
        tier = prediction.risk_tier

        if tier in ("CRITICAL", "HIGH"):
            recs.append("Consider infectious disease consultation")
            recs.append("Review current antibiotic regimen")

        if prediction.antibiotic_class_risk.carbapenems > 0.5:
            recs.append("Monitor carbapenem susceptibility closely")

        if prediction.low_confidence_flag:
            recs.append("Low confidence — additional culture data recommended")

        if tier == "CRITICAL":
            recs.append("Activate antimicrobial stewardship protocol")

        return tuple(recs) if recs else ("Continue standard monitoring",)

    def process_hook(self, request: CDSHookRequest) -> CDSHookResponse:
        """Process a CDS Hook request, with cache fallback."""
        start = time.time()
        request_id = self._generate_request_id()
        cache_key = self._cache_key(request.hook_type, request.patient_token)

        # Try cache first for speed
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Run prediction
        try:
            prediction_result = self._predictor.predict(request.context)
            if hasattr(prediction_result, "to_amr_output"):
                amr_output = prediction_result.to_amr_output()
            else:
                amr_output = prediction_result
        except Exception:
            # Fallback to cache on error
            if cache_key in self._cache:
                resp, _ = self._cache[cache_key]
                return CDSHookResponse(
                    cards=resp.cards,
                    hook_type=request.hook_type,
                    latency_ms=(time.time() - start) * 1000,
                    from_cache=True,
                    request_id=request_id,
                )
            # Return empty card set
            return CDSHookResponse(
                cards=[],
                hook_type=request.hook_type,
                latency_ms=(time.time() - start) * 1000,
                from_cache=False,
                request_id=request_id,
            )

        card = self._build_card(amr_output)
        latency_ms = (time.time() - start) * 1000

        response = CDSHookResponse(
            cards=[card],
            hook_type=request.hook_type,
            latency_ms=latency_ms,
            from_cache=False,
            request_id=request_id,
        )

        self._put_cache(cache_key, response)
        return response

    def record_action(
        self,
        request: CDSHookRequest,
        response: CDSHookResponse,
        action: ResponseAction,
        override_reason: str | None = None,
    ) -> AuditEntry:
        """Record a clinician action in the audit trail."""
        risk_tier = ""
        if response.cards:
            risk_tier = response.cards[0].risk_tier

        entry = AuditEntry(
            request_id=response.request_id,
            hook_type=request.hook_type,
            patient_token=request.patient_token,
            action=action,
            override_reason=override_reason if action == ResponseAction.OVERRIDE else None,
            timestamp=time.time(),
            response_latency_ms=response.latency_ms,
            risk_tier=risk_tier,
            cards_count=len(response.cards),
        )

        if self._config.audit_enabled:
            self._audit_log.append(entry)

        return entry

    def get_audit_log(self) -> list[AuditEntry]:
        """Return audit log entries (no PHI — only patient_token)."""
        return list(self._audit_log)

    def get_registered_hooks(self) -> list[dict[str, str]]:
        """Return registered hook definitions per CDS Hooks spec."""
        return [
            {
                "hook": HookType.PATIENT_VIEW,
                "title": "AMR Risk on Patient View",
                "description": "Displays AMR risk when viewing patient chart",
            },
            {
                "hook": HookType.ORDER_SIGN,
                "title": "AMR Risk on Order Sign",
                "description": "AMR risk alert when signing antibiotic orders",
            },
            {
                "hook": HookType.ENCOUNTER_DISCHARGE,
                "title": "AMR Risk on Discharge",
                "description": "AMR risk summary at encounter discharge",
            },
        ]

    def get_stats(self) -> dict[str, Any]:
        """Service statistics."""
        return {
            "cache_size": len(self._cache),
            "audit_entries": len(self._audit_log),
            "requests_processed": self._request_counter,
            "registered_hooks": len(self.get_registered_hooks()),
        }
