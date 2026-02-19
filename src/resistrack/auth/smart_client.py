"""SMART on FHIR OAuth 2.0 client for EHR launch and standalone flows."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone

from resistrack.auth.models import TokenPayload, UserRole, UserSession


@dataclass
class SMARTConfig:
    """Configuration for SMART on FHIR auth client."""

    client_id: str
    redirect_uri: str
    scopes: str = "openid fhirUser launch/patient patient/*.read"
    token_endpoint: str = ""
    authorize_endpoint: str = ""
    fhir_server_url: str = ""


@dataclass
class SMARTAuthClient:
    """SMART on FHIR authentication client.

    Supports both EHR launch and standalone launch flows.
    """

    config: SMARTConfig
    _state_store: dict[str, dict[str, Any]] = field(default_factory=dict)

    def build_authorize_url(self, launch_token: str | None = None) -> tuple[str, str]:
        """Build the OAuth 2.0 authorization URL.

        Returns:
            Tuple of (authorize_url, state_parameter).
        """
        state = hashlib.sha256(
            f"{time.time()}-{self.config.client_id}".encode()
        ).hexdigest()[:32]

        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": self.config.scopes,
            "state": state,
            "aud": self.config.fhir_server_url,
        }

        if launch_token:
            params["launch"] = launch_token

        self._state_store[state] = {
            "created_at": time.time(),
            "launch_token": launch_token,
        }

        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.config.authorize_endpoint}?{query}"
        return url, state

    def validate_state(self, state: str) -> bool:
        """Validate the OAuth state parameter to prevent CSRF."""
        stored = self._state_store.get(state)
        if stored is None:
            return False
        age = time.time() - stored["created_at"]
        if age > 600:  # 10 minute expiry
            del self._state_store[state]
            return False
        return True

    def create_session_from_token(
        self,
        token_response: dict[str, Any],
        token_payload: dict[str, Any],
    ) -> UserSession:
        """Create a user session from a validated token response.

        Args:
            token_response: Raw OAuth token endpoint response.
            token_payload: Decoded and verified JWT payload.
        """
        payload = TokenPayload(**token_payload)
        expires_at = datetime.fromtimestamp(payload.exp, tz=timezone.utc)

        return UserSession(
            user_id=payload.sub,
            role=payload.user_role,
            tenant_id=payload.hospital_tenant_id,
            access_token=token_response.get("access_token", ""),
            refresh_token=token_response.get("refresh_token", ""),
            fhir_server_url=self.config.fhir_server_url,
            expires_at=expires_at,
            launched_from_ehr="launch" in self.config.scopes,
        )

    def verify_token_signature(
        self, token: str, secret: str
    ) -> dict[str, Any] | None:
        """Verify a JWT token signature (HMAC-SHA256 for dev/testing).

        In production, use KMS-based RS256 verification.

        Returns:
            Decoded payload if valid, None if invalid.
        """
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_payload = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(
            secret.encode(), header_payload.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_sig, parts[2]):
            return None

        try:
            import base64
            padding = 4 - len(parts[1]) % 4
            payload_bytes = base64.urlsafe_b64decode(parts[1] + "=" * padding)
            return json.loads(payload_bytes)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            return None


__all__ = ["SMARTConfig", "SMARTAuthClient"]
