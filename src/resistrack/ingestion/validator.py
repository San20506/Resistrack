"""FHIR bundle validation for ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


REQUIRED_FHIR_FIELDS: dict[str, list[str]] = {
    "Patient": ["resourceType", "identifier"],
    "Encounter": ["resourceType", "status", "class"],
    "Observation": ["resourceType", "status", "code"],
    "MedicationRequest": ["resourceType", "status", "intent", "medicationCodeableConcept"],
}

VALID_RESOURCE_TYPES: set[str] = {"Patient", "Encounter", "Observation", "MedicationRequest", "Bundle"}


@dataclass
class ValidationResult:
    """Result of FHIR bundle validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resource_count: int = 0
    resource_types: dict[str, int] = field(default_factory=dict)


class FHIRBundleValidator:
    """Validate incoming FHIR bundles before ingestion."""

    def validate_bundle(self, bundle: dict[str, Any]) -> ValidationResult:
        """Validate a FHIR Bundle resource."""
        errors: list[str] = []
        warnings: list[str] = []
        type_counts: dict[str, int] = {}

        if bundle.get("resourceType") != "Bundle":
            errors.append("Root resource must be a Bundle")
            return ValidationResult(is_valid=False, errors=errors)

        entries = bundle.get("entry", [])
        if not entries:
            errors.append("Bundle contains no entries")
            return ValidationResult(is_valid=False, errors=errors)

        for i, entry in enumerate(entries):
            resource = entry.get("resource", {})
            rt = resource.get("resourceType", "Unknown")

            if rt not in VALID_RESOURCE_TYPES:
                warnings.append(f"Entry {i}: Unknown resource type '{rt}'")
                continue

            type_counts[rt] = type_counts.get(rt, 0) + 1

            required = REQUIRED_FHIR_FIELDS.get(rt, [])
            for req_field in required:
                if req_field not in resource:
                    errors.append(f"Entry {i} ({rt}): Missing required field '{req_field}'")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            resource_count=len(entries),
            resource_types=type_counts,
        )

    def validate_resource(self, resource: dict[str, Any]) -> ValidationResult:
        """Validate a single FHIR resource."""
        rt = resource.get("resourceType", "Unknown")
        errors: list[str] = []

        if rt not in VALID_RESOURCE_TYPES:
            errors.append(f"Unknown resource type: {rt}")
            return ValidationResult(is_valid=False, errors=errors)

        required = REQUIRED_FHIR_FIELDS.get(rt, [])
        for req_field in required:
            if req_field not in resource:
                errors.append(f"Missing required field: {req_field}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            resource_count=1,
            resource_types={rt: 1},
        )
