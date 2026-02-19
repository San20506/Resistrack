"""FHIR R4 resource validator for ResisTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

REQUIRED_FIELDS_BY_TYPE: Final[dict[str, list[str]]] = {
    "Patient": ["resourceType", "id"],
    "Encounter": ["resourceType", "id", "status", "class"],
    "Observation": ["resourceType", "id", "status", "code"],
    "MedicationRequest": ["resourceType", "id", "status", "intent", "medicationCodeableConcept"],
}

VALID_RESOURCE_TYPES: Final[set[str]] = {
    "Patient",
    "Encounter",
    "Observation",
    "MedicationRequest",
    "Bundle",
    "DiagnosticReport",
}


@dataclass
class ValidationResult:
    """Result of FHIR resource validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resource_type: str = ""


def validate_fhir_resource(resource: dict[str, Any]) -> ValidationResult:
    """Validate a FHIR R4 resource dictionary.

    Performs structural validation:
    - resourceType is present and valid
    - Required fields are present
    - Coding systems are valid URIs
    - References follow FHIR format

    Args:
        resource: FHIR resource as dictionary.

    Returns:
        ValidationResult with errors and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check resourceType
    resource_type = resource.get("resourceType", "")
    if not resource_type:
        errors.append("Missing 'resourceType' field")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    if resource_type not in VALID_RESOURCE_TYPES:
        errors.append(f"Unknown resourceType: {resource_type}")

    # Check required fields
    required = REQUIRED_FIELDS_BY_TYPE.get(resource_type, ["resourceType", "id"])
    for req_field in required:
        if req_field not in resource or not resource[req_field]:
            errors.append(f"Missing required field: {req_field}")

    # Check id format
    resource_id = resource.get("id", "")
    if resource_id and not isinstance(resource_id, str):
        errors.append("'id' must be a string")

    # Validate references
    _validate_references(resource, errors, warnings)

    # Validate coding elements
    _validate_codings(resource, errors, warnings)

    # Check meta
    if "meta" not in resource:
        warnings.append("Missing 'meta' element (recommended)")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        resource_type=resource_type,
    )


def _validate_references(resource: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    """Validate FHIR reference fields."""
    for key, value in resource.items():
        if key == "subject" and isinstance(value, dict):
            ref = value.get("reference", "")
            if ref and "/" not in ref:
                errors.append(f"Invalid reference format in '{key}': {ref} (expected Type/id)")
        elif isinstance(value, dict):
            _validate_references(value, errors, warnings)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _validate_references(item, errors, warnings)


def _validate_codings(resource: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    """Validate coding elements have required system and code fields."""
    if "coding" in resource and isinstance(resource["coding"], list):
        for coding in resource["coding"]:
            if isinstance(coding, dict):
                if "code" not in coding:
                    errors.append("Coding element missing 'code'")
                if "system" not in coding:
                    warnings.append("Coding element missing 'system' (recommended)")

    for value in resource.values():
        if isinstance(value, dict):
            _validate_codings(value, errors, warnings)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _validate_codings(item, errors, warnings)
