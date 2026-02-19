"""Tests for M3.2 – CDS Hooks service."""

from __future__ import annotations

import time

import pytest

from resistrack.common.constants import RiskTier
from resistrack.common.schemas import (
    AMRPredictionOutput,
    AntibioticClassRisk,
    SHAPFeature,
)
from resistrack.cds.hooks import (
    AuditEntry,
    CDSCard,
    CDSHookRequest,
    CDSHookResponse,
    CDSHooksService,
    HookConfig,
    HookType,
    ResponseAction,
)


# --- Helpers ---


def _make_prediction(
    score: float = 75.0,
    risk_tier: str = "HIGH",
    confidence: float = 0.85,
) -> AMRPredictionOutput:
    return AMRPredictionOutput(
        patient_token="pt_test123",
        amr_risk_score=score,
        risk_tier=risk_tier,
        confidence_score=confidence,
        data_completeness_score=0.90,
        data_quality_flag=True,
        antibiotic_class_risk=AntibioticClassRisk(
            penicillins=0.7, cephalosporins=0.5,
            carbapenems=0.6, fluoroquinolones=0.4,
            aminoglycosides=0.2,
        ),
        shap_top_features=[
            SHAPFeature(name="prior_resistance", value=0.35,
                        direction="positive",
                        human_readable="Prior resistance history increases risk"),
            SHAPFeature(name="recent_antibiotics", value=0.25,
                        direction="positive",
                        human_readable="Recent antibiotic use raises concern"),
            SHAPFeature(name="age", value=0.15,
                        direction="positive",
                        human_readable="Patient age is a contributing factor"),
        ],
        recommended_action="Review HIGH risk patient",
        model_version="3.2.0",
    )


class _MockPredictor:
    def __init__(self, prediction=None, should_fail=False):
        self._prediction = prediction or _make_prediction()
        self._should_fail = should_fail

    def predict(self, features):
        if self._should_fail:
            raise RuntimeError("Prediction failed")
        return self._prediction


def _make_request(hook_type=HookType.PATIENT_VIEW) -> CDSHookRequest:
    return CDSHookRequest(
        hook_type=hook_type,
        patient_token="pt_test123",
        encounter_id="enc_001",
        context={"temp": 38.5, "wbc": 12.0},
    )


# --- HookType & ResponseAction Tests ---


def test_hook_types():
    assert HookType.PATIENT_VIEW == "patient-view"
    assert HookType.ORDER_SIGN == "order-sign"
    assert HookType.ENCOUNTER_DISCHARGE == "encounter-discharge"


def test_response_actions():
    assert ResponseAction.ACKNOWLEDGED == "acknowledged"
    assert ResponseAction.OVERRIDE == "override"
    assert ResponseAction.ESCALATE == "escalate"


# --- CDSCard Tests ---


def test_cds_card_creation():
    card = CDSCard(
        summary="AMR Risk: HIGH",
        detail="High risk patient",
        indicator="critical",
        risk_tier="HIGH",
        confidence_score=0.85,
        shap_factors=("Prior resistance",),
        recommendations=("Review antibiotic regimen",),
    )
    assert card.indicator == "critical"
    assert card.risk_tier == "HIGH"
    assert len(card.shap_factors) == 1
    assert card.source_label == "ResisTrack AMR"


# --- Service Tests ---


def test_service_creation():
    service = CDSHooksService(_MockPredictor())
    assert service.config.timeout_ms == 1500.0
    assert service.config.sla_ms == 2000.0


def test_service_custom_config():
    cfg = HookConfig(timeout_ms=1000, sla_ms=1500)
    service = CDSHooksService(_MockPredictor(), config=cfg)
    assert service.config.timeout_ms == 1000


def test_service_process_hook_patient_view():
    service = CDSHooksService(_MockPredictor())
    request = _make_request(HookType.PATIENT_VIEW)
    response = service.process_hook(request)
    assert isinstance(response, CDSHookResponse)
    assert response.hook_type == HookType.PATIENT_VIEW
    assert len(response.cards) == 1
    assert not response.from_cache


def test_service_process_hook_order_sign():
    service = CDSHooksService(_MockPredictor())
    request = _make_request(HookType.ORDER_SIGN)
    response = service.process_hook(request)
    assert response.hook_type == HookType.ORDER_SIGN
    assert len(response.cards) == 1


def test_service_process_hook_encounter_discharge():
    service = CDSHooksService(_MockPredictor())
    request = _make_request(HookType.ENCOUNTER_DISCHARGE)
    response = service.process_hook(request)
    assert response.hook_type == HookType.ENCOUNTER_DISCHARGE


def test_service_card_has_shap_factors():
    service = CDSHooksService(_MockPredictor())
    request = _make_request()
    response = service.process_hook(request)
    card = response.cards[0]
    assert len(card.shap_factors) > 0
    assert len(card.shap_factors) <= 3


def test_service_card_has_recommendations():
    service = CDSHooksService(_MockPredictor())
    request = _make_request()
    response = service.process_hook(request)
    card = response.cards[0]
    assert len(card.recommendations) > 0


def test_service_audit_logging():
    service = CDSHooksService(_MockPredictor())
    request = _make_request()
    response = service.process_hook(request)
    entry = service.record_action(request, response, ResponseAction.ACKNOWLEDGED)
    assert isinstance(entry, AuditEntry)
    assert entry.action == ResponseAction.ACKNOWLEDGED
    log = service.get_audit_log()
    assert len(log) == 1


def test_service_respond_acknowledged():
    service = CDSHooksService(_MockPredictor())
    request = _make_request()
    response = service.process_hook(request)
    entry = service.record_action(request, response, ResponseAction.ACKNOWLEDGED)
    assert entry.action == ResponseAction.ACKNOWLEDGED
    assert entry.override_reason is None


def test_service_respond_override():
    service = CDSHooksService(_MockPredictor())
    request = _make_request()
    response = service.process_hook(request)
    entry = service.record_action(
        request, response, ResponseAction.OVERRIDE,
        override_reason="Clinical judgment",
    )
    assert entry.action == ResponseAction.OVERRIDE
    assert entry.override_reason == "Clinical judgment"


def test_service_respond_escalate():
    service = CDSHooksService(_MockPredictor())
    request = _make_request()
    response = service.process_hook(request)
    entry = service.record_action(request, response, ResponseAction.ESCALATE)
    assert entry.action == ResponseAction.ESCALATE


def test_service_latency_under_sla():
    service = CDSHooksService(_MockPredictor())
    request = _make_request()
    response = service.process_hook(request)
    assert response.latency_ms < 2000.0


def test_service_cache_fallback():
    predictor = _MockPredictor()
    service = CDSHooksService(predictor)
    request = _make_request()
    service.process_hook(request)
    # Second call hits cache
    response2 = service.process_hook(request)
    assert response2.from_cache is True


def test_service_error_fallback_empty():
    service = CDSHooksService(_MockPredictor(should_fail=True))
    request = _make_request()
    response = service.process_hook(request)
    assert len(response.cards) == 0


def test_service_multiple_hooks():
    service = CDSHooksService(_MockPredictor())
    for ht in [HookType.PATIENT_VIEW, HookType.ORDER_SIGN, HookType.ENCOUNTER_DISCHARGE]:
        response = service.process_hook(_make_request(ht))
        assert len(response.cards) == 1


def test_service_critical_indicator():
    pred = _make_prediction(score=90.0, risk_tier="CRITICAL")
    service = CDSHooksService(_MockPredictor(prediction=pred))
    response = service.process_hook(_make_request())
    assert response.cards[0].indicator == "critical"


def test_service_low_risk_indicator():
    pred = _make_prediction(score=10.0, risk_tier="LOW")
    service = CDSHooksService(_MockPredictor(prediction=pred))
    response = service.process_hook(_make_request())
    assert response.cards[0].indicator == "info"


def test_registered_hooks():
    service = CDSHooksService(_MockPredictor())
    hooks = service.get_registered_hooks()
    assert len(hooks) == 3
    hook_names = [h["hook"] for h in hooks]
    assert "patient-view" in hook_names


def test_service_stats():
    service = CDSHooksService(_MockPredictor())
    service.process_hook(_make_request())
    stats = service.get_stats()
    assert stats["requests_processed"] == 1
    assert stats["registered_hooks"] == 3
