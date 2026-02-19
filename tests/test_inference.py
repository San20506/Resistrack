"""Tests for M2.7 – SageMaker inference endpoint."""

from __future__ import annotations

import time

import numpy as np
import pytest

from resistrack.common.constants import RANDOM_STATE, RiskTier
from resistrack.common.schemas import (
    AMRPredictionOutput,
    AntibioticClassRisk,
    SHAPFeature,
)
from resistrack.inference.endpoint import (
    AutoScaler,
    CacheEntry,
    EndpointConfig,
    EndpointHealthStatus,
    InferenceResult,
    ResponseCache,
    SageMakerEndpoint,
)


# --- Helpers ---

def _make_prediction(score: float = 55.0, risk_tier: str = "HIGH") -> AMRPredictionOutput:
    return AMRPredictionOutput(
        patient_token="pt_test123",
        amr_risk_score=score,
        risk_tier=risk_tier,
        confidence_score=0.85,
        data_completeness_score=0.90,
        data_quality_flag=True,
        antibiotic_class_risk=AntibioticClassRisk(
            penicillins=0.7, cephalosporins=0.5,
            carbapenems=0.3, fluoroquinolones=0.4,
            aminoglycosides=0.2,
        ),
        shap_top_features=[
            SHAPFeature(
                name="prior_resistance",
                value=0.35,
                direction="positive",
                human_readable="Prior resistance history increases risk",
            ),
        ],
        recommended_action="Review HIGH risk patient",
        model_version="2.7.0",
    )


class _MockPredictor:
    """A mock predictor that returns a simple object with the right fields."""

    def predict(self, features: dict) -> object:
        class _Pred:
            calibrated_score = 0.55
            confidence = 0.85
            risk_tier = RiskTier.HIGH
            antibiotic_risks = {
                "penicillins": 0.7, "cephalosporins": 0.5,
                "carbapenems": 0.3, "fluoroquinolones": 0.4,
                "aminoglycosides": 0.2,
            }
            shap_features = [
                SHAPFeature(
                    name="prior_resistance", value=0.35,
                    direction="positive",
                    human_readable="Prior resistance history increases risk",
                ),
            ]

            def to_amr_output(self) -> AMRPredictionOutput:
                return _make_prediction()
        return _Pred()


# --- EndpointConfig Tests ---

def test_endpoint_config_defaults():
    cfg = EndpointConfig(model_name="amr-v2")
    assert cfg.model_name == "amr-v2"
    assert cfg.instance_type == "ml.m5.xlarge"
    assert cfg.min_instances == 1
    assert cfg.max_instances == 20
    assert cfg.target_latency_ms == 2000.0
    assert cfg.cache_ttl_seconds == 86400


def test_endpoint_config_custom():
    cfg = EndpointConfig(model_name="amr-custom", min_instances=2, max_instances=10)
    assert cfg.min_instances == 2
    assert cfg.max_instances == 10


# --- CacheEntry Tests ---

def test_cache_entry_not_expired():
    entry = CacheEntry(
        key="test_key",
        prediction=_make_prediction(),
        created_at=time.time(),
        ttl_seconds=86400,
    )
    assert not entry.is_expired
    assert entry.cached_result is True


def test_cache_entry_expired():
    entry = CacheEntry(
        key="test_key",
        prediction=_make_prediction(),
        created_at=time.time() - 90000,
        ttl_seconds=86400,
    )
    assert entry.is_expired


# --- ResponseCache Tests ---

def test_cache_put_and_get():
    cache = ResponseCache(max_size=100, ttl_seconds=86400)
    pred = _make_prediction()
    cache.put("key1", pred)
    entry = cache.get("key1")
    assert entry is not None
    assert entry.prediction == pred


def test_cache_miss():
    cache = ResponseCache(max_size=100, ttl_seconds=86400)
    entry = cache.get("nonexistent")
    assert entry is None
    assert cache._misses == 1


def test_cache_hit_counter():
    cache = ResponseCache(max_size=100, ttl_seconds=86400)
    cache.put("key1", _make_prediction())
    cache.get("key1")
    cache.get("key1")
    assert cache._hits == 2
    assert cache._misses == 0


def test_cache_expired_entry_returns_none():
    cache = ResponseCache(max_size=100, ttl_seconds=1)
    cache.put("key1", _make_prediction())
    # Manually expire
    old_entry = cache._store["key1"]
    cache._store["key1"] = CacheEntry(
        key=old_entry.key,
        prediction=old_entry.prediction,
        created_at=time.time() - 10,
        ttl_seconds=1,
    )
    result = cache.get("key1")
    assert result is None


def test_cache_evict_expired():
    cache = ResponseCache(max_size=100, ttl_seconds=1)
    cache.put("key1", _make_prediction())
    cache.put("key2", _make_prediction())
    # Expire them
    for k in list(cache._store.keys()):
        entry = cache._store[k]
        cache._store[k] = CacheEntry(
            key=entry.key,
            prediction=entry.prediction,
            created_at=time.time() - 10,
            ttl_seconds=1,
        )
    evicted = cache.evict_expired()
    assert evicted == 2
    assert len(cache._store) == 0


def test_cache_lru_eviction():
    cache = ResponseCache(max_size=3, ttl_seconds=86400)
    for i in range(4):
        cache.put(f"key{i}", _make_prediction())
    assert len(cache._store) == 3
    assert cache.get("key0") is None  # Evicted


def test_cache_stats():
    cache = ResponseCache(max_size=100, ttl_seconds=86400)
    cache.put("key1", _make_prediction())
    cache.get("key1")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["size"] == 1


def test_cache_make_key_deterministic():
    key1 = ResponseCache._make_key({"a": 1.0, "b": 2.0})
    key2 = ResponseCache._make_key({"b": 2.0, "a": 1.0})
    assert key1 == key2


# --- AutoScaler Tests ---

def test_autoscaler_scale_up_high_latency():
    scaler = AutoScaler(min_instances=1, max_instances=20, target_latency_ms=2000.0)
    # 80% of 2000 = 1600, so > 1600 should scale up
    assert scaler.should_scale_up(avg_latency_ms=1700.0, invocations_per_min=10.0, current_instances=2)


def test_autoscaler_no_scale_up_low_latency():
    scaler = AutoScaler(min_instances=1, max_instances=20, target_latency_ms=2000.0)
    assert not scaler.should_scale_up(avg_latency_ms=500.0, invocations_per_min=10.0, current_instances=2)


def test_autoscaler_scale_down_low_latency():
    scaler = AutoScaler(min_instances=1, max_instances=20, target_latency_ms=2000.0)
    # 40% of 2000 = 800, so < 800 should scale down if > min
    assert scaler.should_scale_down(avg_latency_ms=300.0, invocations_per_min=10.0, current_instances=5)


def test_autoscaler_no_scale_down_at_min():
    scaler = AutoScaler(min_instances=1, max_instances=20, target_latency_ms=2000.0)
    assert not scaler.should_scale_down(avg_latency_ms=300.0, invocations_per_min=10.0, current_instances=1)


def test_autoscaler_evaluate_scale_up():
    scaler = AutoScaler(min_instances=1, max_instances=20, target_latency_ms=2000.0)
    new_count = scaler.evaluate(current_instances=5, avg_latency_ms=1800.0, invocations_per_min=10.0)
    assert new_count == 6


def test_autoscaler_evaluate_scale_down():
    scaler = AutoScaler(min_instances=1, max_instances=20, target_latency_ms=2000.0)
    new_count = scaler.evaluate(current_instances=5, avg_latency_ms=300.0, invocations_per_min=10.0)
    assert new_count == 4


def test_autoscaler_evaluate_no_change():
    scaler = AutoScaler(min_instances=1, max_instances=20, target_latency_ms=2000.0)
    new_count = scaler.evaluate(current_instances=5, avg_latency_ms=1200.0, invocations_per_min=100.0)
    assert new_count == 5


def test_autoscaler_respects_max():
    scaler = AutoScaler(min_instances=1, max_instances=3, target_latency_ms=2000.0)
    new_count = scaler.evaluate(current_instances=3, avg_latency_ms=1800.0, invocations_per_min=10.0)
    assert new_count == 3


# --- SageMakerEndpoint Tests ---

def test_endpoint_predict():
    cfg = EndpointConfig(model_name="amr-test")
    endpoint = SageMakerEndpoint(cfg, _MockPredictor())
    features = {"temp": 38.5, "wbc": 12.0}
    result = endpoint.predict(features)
    assert isinstance(result, InferenceResult)
    assert not result.cached_result
    assert result.latency_ms >= 0
    assert result.prediction.patient_token == "pt_test123"


def test_endpoint_predict_cache_hit():
    cfg = EndpointConfig(model_name="amr-test")
    endpoint = SageMakerEndpoint(cfg, _MockPredictor())
    features = {"temp": 38.5, "wbc": 12.0}
    result1 = endpoint.predict(features)
    result2 = endpoint.predict(features)
    assert not result1.cached_result
    assert result2.cached_result


def test_endpoint_batch_predict():
    cfg = EndpointConfig(model_name="amr-test")
    endpoint = SageMakerEndpoint(cfg, _MockPredictor())
    batch = [{"temp": 38.5}, {"temp": 37.0}, {"temp": 39.0}]
    results = endpoint.batch_predict(batch)
    assert len(results) == 3
    assert all(isinstance(r, InferenceResult) for r in results)


def test_endpoint_health_check():
    cfg = EndpointConfig(model_name="amr-test")
    endpoint = SageMakerEndpoint(cfg, _MockPredictor())
    endpoint.predict({"temp": 38.5})
    health = endpoint.health_check()
    assert isinstance(health, EndpointHealthStatus)
    assert health.is_healthy is True
    assert health.active_instances >= 1
    assert health.error_rate == 0.0


def test_endpoint_cache_stats():
    cfg = EndpointConfig(model_name="amr-test")
    endpoint = SageMakerEndpoint(cfg, _MockPredictor())
    endpoint.predict({"temp": 38.5})
    endpoint.predict({"temp": 38.5})  # cache hit
    stats = endpoint.get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_endpoint_auto_scale():
    cfg = EndpointConfig(model_name="amr-test", min_instances=1, max_instances=5)
    endpoint = SageMakerEndpoint(cfg, _MockPredictor())
    # Artificially inject high latencies
    endpoint._latencies = [1900.0] * 200
    new_count = endpoint.auto_scale(invocations_per_min=100.0)
    assert new_count == 2
