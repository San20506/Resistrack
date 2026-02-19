"""Tests for M3.1 SMART on FHIR auth, RBAC, and session management."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from resistrack.auth.models import (
    ROLE_PERMISSIONS,
    TokenPayload,
    UserRole,
    UserSession,
)
from resistrack.auth.rbac import AccessDeniedError, RBACEnforcer
from resistrack.auth.smart_client import SMARTAuthClient, SMARTConfig


# ── UserRole tests ──


class TestUserRole:
    def test_all_roles_exist(self) -> None:
        roles = {r.value for r in UserRole}
        assert roles == {
            "PHYSICIAN", "PHARMACIST", "INFECTION_CONTROL",
            "NURSE", "ADMIN", "READONLY",
        }

    def test_role_from_string(self) -> None:
        assert UserRole("PHYSICIAN") == UserRole.PHYSICIAN


# ── TokenPayload tests ──


class TestTokenPayload:
    def _make_payload(self, **overrides: object) -> dict:
        base: dict = {
            "sub": "user-123",
            "iss": "https://auth.hospital.org",
            "aud": "resistrack",
            "exp": 9999999999,
            "iat": 1700000000,
            "hospital_tenant_id": "hosp-001",
            "user_role": "PHYSICIAN",
        }
        base.update(overrides)
        return base

    def test_valid_payload(self) -> None:
        payload = TokenPayload(**self._make_payload())
        assert payload.sub == "user-123"
        assert payload.user_role == UserRole.PHYSICIAN

    def test_phi_rejected_patient_name(self) -> None:
        with pytest.raises(ValidationError, match="PHI detected"):
            TokenPayload(**self._make_payload(patient_name="John Doe"))

    def test_phi_rejected_mrn(self) -> None:
        with pytest.raises(ValidationError, match="PHI detected"):
            TokenPayload(**self._make_payload(mrn="MRN-12345"))

    def test_phi_rejected_ssn(self) -> None:
        with pytest.raises(ValidationError, match="PHI detected"):
            TokenPayload(**self._make_payload(ssn="123-45-6789"))

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            TokenPayload(**self._make_payload(user_role="SUPERADMIN"))


# ── UserSession tests ──


class TestUserSession:
    def _make_session(self, **overrides: object) -> dict:
        base: dict = {
            "user_id": "user-123",
            "role": "PHYSICIAN",
            "tenant_id": "hosp-001",
            "access_token": "tok_abc",
            "expires_at": datetime(2030, 1, 1, tzinfo=timezone.utc),
        }
        base.update(overrides)
        return base

    def test_permissions_auto_populated(self) -> None:
        session = UserSession(**self._make_session())
        assert "view_patient" in session.permissions
        assert "override_recommendation" in session.permissions

    def test_readonly_limited_permissions(self) -> None:
        session = UserSession(**self._make_session(role="READONLY"))
        assert "view_predictions" in session.permissions
        assert "manage_users" not in session.permissions
        assert "view_patient" not in session.permissions

    def test_admin_has_all_permissions(self) -> None:
        session = UserSession(**self._make_session(role="ADMIN"))
        assert "manage_users" in session.permissions
        assert "view_audit_logs" in session.permissions
        assert "manage_config" in session.permissions

    def test_explicit_permissions_preserved(self) -> None:
        session = UserSession(
            **self._make_session(permissions=frozenset({"custom_perm"}))
        )
        assert session.permissions == frozenset({"custom_perm"})


# ── RBAC tests ──


class TestRBACEnforcer:
    def _session(self, role: str = "PHYSICIAN") -> UserSession:
        return UserSession(
            user_id="user-123",
            role=UserRole(role),
            tenant_id="hosp-001",
            access_token="tok",
            expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )

    def test_check_permission_granted(self) -> None:
        session = self._session("PHYSICIAN")
        assert RBACEnforcer.check_permission(session, "view_patient") is True

    def test_check_permission_denied(self) -> None:
        session = self._session("READONLY")
        assert RBACEnforcer.check_permission(session, "manage_users") is False

    def test_require_permission_passes(self) -> None:
        session = self._session("ADMIN")
        RBACEnforcer.require_permission(session, "manage_users")

    def test_require_permission_raises(self) -> None:
        session = self._session("NURSE")
        with pytest.raises(AccessDeniedError) as exc_info:
            RBACEnforcer.require_permission(session, "manage_users")
        assert exc_info.value.permission == "manage_users"
        assert exc_info.value.role == UserRole.NURSE

    def test_has_any_permission(self) -> None:
        session = self._session("PHARMACIST")
        assert RBACEnforcer.has_any_permission(
            session, {"view_antibiotic_risks", "manage_users"}
        ) is True

    def test_has_all_permissions(self) -> None:
        session = self._session("PHARMACIST")
        assert RBACEnforcer.has_all_permissions(
            session, {"view_patient", "view_predictions"}
        ) is True
        assert RBACEnforcer.has_all_permissions(
            session, {"view_patient", "manage_users"}
        ) is False

    def test_get_role_permissions(self) -> None:
        perms = RBACEnforcer.get_role_permissions(UserRole.INFECTION_CONTROL)
        assert "view_reports" in perms
        assert "export_reports" in perms


# ── SMART Auth Client tests ──


class TestSMARTAuthClient:
    def _client(self) -> SMARTAuthClient:
        config = SMARTConfig(
            client_id="resistrack-app",
            redirect_uri="https://app.resistrack.com/callback",
            authorize_endpoint="https://auth.hospital.org/authorize",
            token_endpoint="https://auth.hospital.org/token",
            fhir_server_url="https://fhir.hospital.org/r4",
        )
        return SMARTAuthClient(config=config)

    def test_build_authorize_url_standalone(self) -> None:
        client = self._client()
        url, state = client.build_authorize_url()
        assert "response_type=code" in url
        assert "client_id=resistrack-app" in url
        assert len(state) == 32

    def test_build_authorize_url_ehr_launch(self) -> None:
        client = self._client()
        url, state = client.build_authorize_url(launch_token="xyz123")
        assert "launch=xyz123" in url

    def test_validate_state_valid(self) -> None:
        client = self._client()
        _, state = client.build_authorize_url()
        assert client.validate_state(state) is True

    def test_validate_state_invalid(self) -> None:
        client = self._client()
        assert client.validate_state("nonexistent") is False

    def test_create_session_from_token(self) -> None:
        client = self._client()
        token_response = {
            "access_token": "access_tok_123",
            "refresh_token": "refresh_tok_456",
        }
        token_payload = {
            "sub": "dr-smith",
            "iss": "https://auth.hospital.org",
            "aud": "resistrack",
            "exp": 9999999999,
            "iat": 1700000000,
            "hospital_tenant_id": "hosp-001",
            "user_role": "PHYSICIAN",
        }
        session = client.create_session_from_token(token_response, token_payload)
        assert session.user_id == "dr-smith"
        assert session.role == UserRole.PHYSICIAN
        assert session.tenant_id == "hosp-001"
        assert "view_patient" in session.permissions

    def test_verify_token_invalid_format(self) -> None:
        client = self._client()
        result = client.verify_token_signature("not-a-jwt", "secret")
        assert result is None

    def test_all_roles_have_permissions(self) -> None:
        for role in UserRole:
            perms = ROLE_PERMISSIONS[role]
            assert len(perms) > 0, f"Role {role} has no permissions"
