"""Authentication models for SMART on FHIR auth."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class UserRole(StrEnum):
    """User roles for role-based access control."""

    PHYSICIAN = "PHYSICIAN"
    PHARMACIST = "PHARMACIST"
    INFECTION_CONTROL = "INFECTION_CONTROL"
    NURSE = "NURSE"
    ADMIN = "ADMIN"
    READONLY = "READONLY"


# Permissions per role
ROLE_PERMISSIONS: dict[UserRole, frozenset[str]] = {
    UserRole.PHYSICIAN: frozenset({
        "view_patient", "view_predictions", "view_shap",
        "override_recommendation", "view_heatmap",
    }),
    UserRole.PHARMACIST: frozenset({
        "view_patient", "view_predictions", "view_antibiotic_risks",
        "view_pharmacy_dashboard", "view_heatmap",
    }),
    UserRole.INFECTION_CONTROL: frozenset({
        "view_patient", "view_predictions", "view_heatmap",
        "view_reports", "export_reports", "view_trends",
    }),
    UserRole.NURSE: frozenset({
        "view_patient", "view_predictions", "view_heatmap",
    }),
    UserRole.ADMIN: frozenset({
        "view_patient", "view_predictions", "view_shap",
        "override_recommendation", "view_heatmap",
        "view_antibiotic_risks", "view_pharmacy_dashboard",
        "view_reports", "export_reports", "view_trends",
        "manage_users", "view_audit_logs", "manage_config",
    }),
    UserRole.READONLY: frozenset({
        "view_predictions", "view_heatmap",
    }),
}


class TokenPayload(BaseModel):
    """JWT token payload — no PHI allowed."""

    sub: str = Field(description="Subject identifier (user ID, not name)")
    iss: str = Field(description="Token issuer URL")
    aud: str = Field(description="Intended audience")
    exp: int = Field(description="Expiration timestamp")
    iat: int = Field(description="Issued-at timestamp")
    hospital_tenant_id: str = Field(description="Tenant identifier")
    user_role: UserRole
    scope: str = Field(default="openid fhirUser launch/patient")

    @model_validator(mode="before")
    @classmethod
    def validate_no_phi(cls, data: Any) -> Any:
        """Ensure no PHI fields are present in the JWT payload."""
        phi_fields = {"patient_name", "mrn", "ssn", "dob", "address", "phone", "email"}
        if isinstance(data, dict):
            found = phi_fields & set(data.keys())
            if found:
                raise ValueError(f"PHI detected in token payload: {found}")
        return data


class UserSession(BaseModel):
    """Authenticated user session."""

    user_id: str
    role: UserRole
    tenant_id: str
    permissions: frozenset[str]
    fhir_server_url: str = ""
    access_token: str = Field(repr=False)
    refresh_token: str = Field(default="", repr=False)
    expires_at: datetime
    launched_from_ehr: bool = False

    @model_validator(mode="before")
    @classmethod
    def set_permissions(cls, data: Any) -> Any:
        """Auto-populate permissions from role if not provided."""
        if isinstance(data, dict) and "permissions" not in data:
            role = data.get("role")
            if role is not None:
                data["permissions"] = ROLE_PERMISSIONS.get(UserRole(role), frozenset())
        return data


__all__ = ["UserRole", "ROLE_PERMISSIONS", "TokenPayload", "UserSession"]
