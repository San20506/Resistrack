"""SageMaker Endpoint — real-time inference with auto-scaling, caching, monitoring.

Provides a simulated SageMaker endpoint wrapping the EnsemblePredictor
with LRU response caching (24 h TTL), latency-based auto-scaling (1–20
instances), and CloudWatch-compatible health reporting.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from resistrack.common.constants import (
    CONFIDENCE_THRESHOLD,
    RANDOM_STATE,
    RISK_TIER_RANGES,
    RiskTier,
)
from resistrack.common.schemas import AMRPredictionOutput

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
_DEFAULT_TTL_SECONDS = 86_400  # 24 h
_DEFAULT_MAX_CACHE_SIZE = 10_000
_TARGET_LATENCY_MS = 2_000.0
_SCALE_UP_LATENCY_FACTOR = 0.80   # scale up when latency > 80 % of target
_SCALE_DOWN_LATENCY_FACTOR = 0.40  # scale down when latency < 40 % of target
_SCALE_UP_INVOCATIONS_PER_INSTANCE = 50  # invocations/min per instance to trigger scale up
_P95_BUFFER_SIZE = 200  # rolling window for p95 estimation


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EndpointConfig:
    """Configuration for a SageMaker endpoint."""

    model_name: str
    instance_type: str = _DEFAULT_INSTANCE_TYPE
    min_instances: int = 1
    max_instances: int = 20
    target_latency_ms: float = _TARGET_LATENCY_MS
    cache_ttl_seconds: int = _DEFAULT_TTL_SECONDS
    max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE


@dataclass(frozen=True)
class CacheEntry:
    """Single response-cache entry."""

    key: str
    prediction: AMRPredictionOutput
    created_at: float
    ttl_seconds: int = _DEFAULT_TTL_SECONDS
    cached_result: bool = True

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass(frozen=True)
class InferenceResult:
    """Result returned by the endpoint for each prediction request."""

    prediction: AMRPredictionOutput
    latency_ms: float
    cached_result: bool
    endpoint_instance_id: str


@dataclass(frozen=True)
class EndpointHealthStatus:
    """Snapshot of endpoint health metrics."""

    is_healthy: bool
    active_instances: int
    p95_latency_ms: float
    error_rate: float
    cache_hit_rate: float


# ---------------------------------------------------------------------------
# Response Cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """LRU cache with TTL-based expiry for inference results."""

    def __init__(self, max_size: int = _DEFAULT_MAX_CACHE_SIZE,
                 ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _make_key(features: dict[str, float]) -> str:
        """Deterministic cache key from feature dict."""
        canonical = json.dumps(features, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    # -- public API ------------------------------------------------------

    def get(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            del self._store[key]
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return entry

    def put(self, key: str, prediction: AMRPredictionOutput) -> CacheEntry:
        if key in self._store:
            del self._store[key]
        entry = CacheEntry(
            key=key,
            prediction=prediction,
            created_at=time.time(),
            ttl_seconds=self._ttl_seconds,
            cached_result=True,
        )
        self._store[key] = entry
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)
        return entry

    def evict_expired(self) -> int:
        now = time.time()
        expired = [
            k for k, v in self._store.items()
            if (now - v.created_at) > v.ttl_seconds
        ]
        for k in expired:
            del self._store[k]
        return len(expired)

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "size": len(self._store),
            "max_size": self._max_size,
        }


# ---------------------------------------------------------------------------
# Auto Scaler
# ---------------------------------------------------------------------------

class AutoScaler:
    """Latency- and invocation-based auto-scaler for endpoint instances."""

    def __init__(self, min_instances: int = 1, max_instances: int = 20,
                 target_latency_ms: float = _TARGET_LATENCY_MS) -> None:
        self._min = min_instances
        self._max = max_instances
        self._target = target_latency_ms

    def should_scale_up(self, avg_latency_ms: float,
                        invocations_per_min: float,
                        current_instances: int) -> bool:
        if avg_latency_ms > self._target * _SCALE_UP_LATENCY_FACTOR:
            return True
        if current_instances > 0 and (
            invocations_per_min / current_instances
            > _SCALE_UP_INVOCATIONS_PER_INSTANCE
        ):
            return True
        return False

    def should_scale_down(self, avg_latency_ms: float,
                          invocations_per_min: float,
                          current_instances: int) -> bool:
        if current_instances <= self._min:
            return False
        if avg_latency_ms < self._target * _SCALE_DOWN_LATENCY_FACTOR:
            return True
        if current_instances > 1 and (
            invocations_per_min / current_instances
            < _SCALE_UP_INVOCATIONS_PER_INSTANCE * 0.3
        ):
            return True
        return False

    def evaluate(self, current_instances: int, avg_latency_ms: float,
                 invocations_per_min: float) -> int:
        """Return recommended instance count."""
        if self.should_scale_up(avg_latency_ms, invocations_per_min,
                                current_instances):
            desired = min(current_instances + 1, self._max)
        elif self.should_scale_down(avg_latency_ms, invocations_per_min,
                                    current_instances):
            desired = max(current_instances - 1, self._min)
        else:
            desired = current_instances
        return max(self._min, min(desired, self._max))


# ---------------------------------------------------------------------------
# SageMaker Endpoint (simulated)
# ---------------------------------------------------------------------------

class SageMakerEndpoint:
    """Simulated SageMaker real-time endpoint wrapping EnsemblePredictor."""

    def __init__(self, config: EndpointConfig,
                 predictor: Any) -> None:
        self._config = config
        self._predictor = predictor
        self._cache = ResponseCache(
            max_size=config.max_cache_size,
            ttl_seconds=config.cache_ttl_seconds,
        )
        self._scaler = AutoScaler(
            min_instances=config.min_instances,
            max_instances=config.max_instances,
            target_latency_ms=config.target_latency_ms,
        )
        self._active_instances = config.min_instances
        self._rng = np.random.RandomState(RANDOM_STATE)
        self._latencies: list[float] = []
        self._total_requests = 0
        self._error_count = 0

    # -- internal --------------------------------------------------------

    def _record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        if len(self._latencies) > _P95_BUFFER_SIZE:
            self._latencies = self._latencies[-_P95_BUFFER_SIZE:]

    def _p95_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.percentile(self._latencies, 95))

    def _instance_id(self) -> str:
        idx = self._rng.randint(0, max(1, self._active_instances))
        return f"{self._config.model_name}-instance-{idx}"

    def _to_amr_output(self, ensemble_pred: Any,
                       features: dict[str, float]) -> AMRPredictionOutput:
        """Convert EnsemblePrediction to AMRPredictionOutput."""
        # Use the ensemble prediction's to_amr_output if available
        if hasattr(ensemble_pred, "to_amr_output"):
            return ensemble_pred.to_amr_output()

        # Fallback: construct manually from EnsemblePrediction fields
        from resistrack.common.schemas import AntibioticClassRisk, SHAPFeature

        risk_score = float(ensemble_pred.calibrated_score * 100)
        risk_tier = str(ensemble_pred.risk_tier)
        confidence = float(ensemble_pred.confidence)

        # Build antibiotic class risk
        ab_risks = ensemble_pred.antibiotic_risks
        ab_risk = AntibioticClassRisk(
            penicillins=float(ab_risks.get("penicillins", 0.5)),
            cephalosporins=float(ab_risks.get("cephalosporins", 0.5)),
            carbapenems=float(ab_risks.get("carbapenems", 0.5)),
            fluoroquinolones=float(ab_risks.get("fluoroquinolones", 0.5)),
            aminoglycosides=float(ab_risks.get("aminoglycosides", 0.5)),
        )

        # Build SHAP features
        shap_features = []
        for sf in ensemble_pred.shap_features[:5]:
            if isinstance(sf, dict):
                shap_features.append(SHAPFeature(**sf))
            else:
                shap_features.append(sf)

        patient_token = hashlib.sha256(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest()[:16]

        return AMRPredictionOutput(
            patient_token=f"pt_{patient_token}",
            amr_risk_score=risk_score,
            risk_tier=risk_tier,
            confidence_score=confidence,
            data_completeness_score=min(len(features) / 20.0, 1.0),
            data_quality_flag=len(features) >= 10,
            antibiotic_class_risk=ab_risk,
            shap_top_features=shap_features,
            recommended_action=f"Review {risk_tier} risk patient",
            model_version="2.7.0",
        )

    # -- public API ------------------------------------------------------

    def predict(self, features: dict[str, float]) -> InferenceResult:
        """Run inference (with cache lookup)."""
        self._total_requests += 1
        start = time.time()

        cache_key = ResponseCache._make_key(features)
        cached = self._cache.get(cache_key)
        if cached is not None:
            latency = (time.time() - start) * 1000
            self._record_latency(latency)
            return InferenceResult(
                prediction=cached.prediction,
                latency_ms=latency,
                cached_result=True,
                endpoint_instance_id=self._instance_id(),
            )

        try:
            ensemble_pred = self._predictor.predict(features)
            amr_output = self._to_amr_output(ensemble_pred, features)
            self._cache.put(cache_key, amr_output)
        except Exception:
            self._error_count += 1
            raise

        latency = (time.time() - start) * 1000
        self._record_latency(latency)

        return InferenceResult(
            prediction=amr_output,
            latency_ms=latency,
            cached_result=False,
            endpoint_instance_id=self._instance_id(),
        )

    def batch_predict(self, feature_batch: list[dict[str, float]],
                      ) -> list[InferenceResult]:
        """Run inference on a batch of feature sets."""
        return [self.predict(f) for f in feature_batch]

    def health_check(self) -> EndpointHealthStatus:
        """Return current endpoint health metrics."""
        cache_stats = self._cache.stats()
        return EndpointHealthStatus(
            is_healthy=self._error_rate() < 0.05,
            active_instances=self._active_instances,
            p95_latency_ms=self._p95_latency(),
            error_rate=self._error_rate(),
            cache_hit_rate=cache_stats["hit_rate"],
        )

    def get_cache_stats(self) -> dict[str, Any]:
        return self._cache.stats()

    def auto_scale(self, invocations_per_min: float) -> int:
        """Run auto-scaling evaluation and adjust instances."""
        new_count = self._scaler.evaluate(
            self._active_instances,
            self._p95_latency(),
            invocations_per_min,
        )
        self._active_instances = new_count
        return new_count

    # -- helpers ---------------------------------------------------------

    def _error_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._error_count / self._total_requests
