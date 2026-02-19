"""M2.7 — SageMaker Endpoint for real-time AMR risk prediction inference."""

from __future__ import annotations

from resistrack.inference.endpoint import (
    AutoScaler,
    CacheEntry,
    EndpointConfig,
    EndpointHealthStatus,
    InferenceResult,
    ResponseCache,
    SageMakerEndpoint,
)

__all__ = [
    "AutoScaler",
    "CacheEntry",
    "EndpointConfig",
    "EndpointHealthStatus",
    "InferenceResult",
    "ResponseCache",
    "SageMakerEndpoint",
]
