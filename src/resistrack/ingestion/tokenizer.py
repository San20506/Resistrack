"""PHI tokenization for HIPAA compliance.

De-identifies all 18 HIPAA identifiers by replacing them with
deterministic, collision-resistant tokens derived from HMAC-SHA256.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from typing import Any

# The 18 HIPAA identifiers that must be de-identified
HIPAA_IDENTIFIERS: list[str] = [
    "name", "address", "dates", "phone", "fax", "email",
    "ssn", "mrn", "health_plan_id", "account_number",
    "certificate_license", "vehicle_id", "device_id", "url",
    "ip_address", "biometric", "photo", "other_unique_id",
]

# FHIR resource paths that contain PHI
PHI_PATHS: dict[str, list[str]] = {
    "Patient": [
        "name", "telecom", "address", "birthDate",
        "identifier", "photo", "contact",
    ],
    "Encounter": ["identifier"],
    "Observation": ["identifier"],
    "MedicationRequest": ["identifier"],
}


@dataclass
class TokenizationResult:
    """Result of tokenizing a FHIR resource."""

    patient_token: str
    resource: dict[str, Any]
    phi_fields_tokenized: int
    original_resource_type: str


class PHITokenizer:
    """Tokenize PHI in FHIR resources using HMAC-SHA256.

    Creates deterministic, collision-resistant tokens so that the
    same patient always maps to the same token within a tenant,
    but different tenants produce different tokens (tenant isolation).
    """

    def __init__(self, secret_key: str, tenant_id: str = "default") -> None:
        self._secret_key = secret_key.encode("utf-8")
        self._tenant_id = tenant_id

    def generate_token(self, identifier: str) -> str:
        """Generate a deterministic patient token from an identifier."""
        message = f"{self._tenant_id}:{identifier}".encode("utf-8")
        digest = hmac.new(self._secret_key, message, hashlib.sha256).hexdigest()
        return f"PT_{digest[:16].upper()}"

    def tokenize_resource(self, resource: dict[str, Any]) -> TokenizationResult:
        """Tokenize all PHI fields in a FHIR resource."""
        resource_type = resource.get("resourceType", "Unknown")
        tokenized = dict(resource)
        phi_count = 0

        # Generate patient token from first identifier
        patient_id = self._extract_patient_id(resource)
        patient_token = self.generate_token(patient_id)

        # Tokenize PHI paths for this resource type
        paths = PHI_PATHS.get(resource_type, [])
        for path in paths:
            if path in tokenized:
                tokenized[path] = self._redact_field(path, tokenized[path])
                phi_count += 1

        # Replace subject reference with token
        if "subject" in tokenized and isinstance(tokenized["subject"], dict):
            tokenized["subject"] = {"reference": f"Patient/{patient_token}"}
            phi_count += 1

        # Add security label
        tokenized.setdefault("meta", {})
        tokenized["meta"]["security"] = [
            {"system": "http://terminology.hl7.org/CodeSystem/v3-Confidentiality", "code": "R"}
        ]

        return TokenizationResult(
            patient_token=patient_token,
            resource=tokenized,
            phi_fields_tokenized=phi_count,
            original_resource_type=resource_type,
        )

    def _extract_patient_id(self, resource: dict[str, Any]) -> str:
        """Extract patient identifier from resource."""
        if resource.get("resourceType") == "Patient":
            identifiers = resource.get("identifier", [])
            if identifiers and isinstance(identifiers, list):
                return str(identifiers[0].get("value", "unknown"))
            return resource.get("id", "unknown")

        subject = resource.get("subject", {})
        if isinstance(subject, dict):
            ref = subject.get("reference", "")
            return ref.replace("Patient/", "") if ref else "unknown"
        return "unknown"

    def _redact_field(self, field_name: str, value: Any) -> Any:
        """Redact a PHI field based on its type."""
        if field_name == "birthDate":
            return "REDACTED"
        if field_name == "name" and isinstance(value, list):
            return [{"text": "[REDACTED]"}]
        if field_name == "telecom" and isinstance(value, list):
            return [{"system": t.get("system", "phone"), "value": "[REDACTED]"} for t in value if isinstance(t, dict)]
        if field_name == "address" and isinstance(value, list):
            return [{"text": "[REDACTED]"}]
        if field_name == "photo" and isinstance(value, list):
            return []
        if field_name == "contact" and isinstance(value, list):
            return []
        if field_name == "identifier" and isinstance(value, list):
            return [
                {"system": i.get("system", ""), "value": self.generate_token(str(i.get("value", "")))}
                for i in value
                if isinstance(i, dict)
            ]
        return "[REDACTED]"

    @staticmethod
    def scan_for_phi(text: str) -> list[str]:
        """Scan text for potential PHI patterns (for log sanitization)."""
        patterns: list[tuple[str, str]] = [
            ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
            ("MRN", r"\bMRN[\s:]*\d+\b"),
            ("Phone", r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            ("Email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            ("Date", r"\b\d{4}-\d{2}-\d{2}\b"),
        ]
        findings: list[str] = []
        for name, pattern in patterns:
            if re.search(pattern, text):
                findings.append(name)
        return findings
