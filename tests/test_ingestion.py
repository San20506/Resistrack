"""Tests for M1.4 FHIR Ingestion & PHI Tokenization."""

import pytest

from resistrack.ingestion.tokenizer import PHITokenizer, HIPAA_IDENTIFIERS
from resistrack.ingestion.validator import FHIRBundleValidator
from resistrack.ingestion.deduplicator import ResourceDeduplicator


# ── Test fixtures ──

def _patient_resource() -> dict:
    return {
        "resourceType": "Patient",
        "id": "patient-1",
        "identifier": [{"system": "http://hospital.org/mrn", "value": "MRN12345"}],
        "name": [{"family": "Smith", "given": ["John"]}],
        "birthDate": "1990-01-15",
        "telecom": [{"system": "phone", "value": "555-123-4567"}],
        "address": [{"city": "Springfield", "state": "IL"}],
    }


def _observation_resource() -> dict:
    return {
        "resourceType": "Observation",
        "id": "obs-1",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": "12345-6"}]},
        "subject": {"reference": "Patient/patient-1"},
        "identifier": [{"value": "OBS-001"}],
    }


def _valid_bundle() -> dict:
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {"resource": _patient_resource()},
            {"resource": _observation_resource()},
        ],
    }


# ── PHITokenizer tests ──

class TestPHITokenizer:
    def test_generate_token_deterministic(self) -> None:
        tok = PHITokenizer(secret_key="test-key", tenant_id="hospital-1")
        t1 = tok.generate_token("MRN12345")
        t2 = tok.generate_token("MRN12345")
        assert t1 == t2
        assert t1.startswith("PT_")

    def test_generate_token_different_patients(self) -> None:
        tok = PHITokenizer(secret_key="test-key", tenant_id="hospital-1")
        t1 = tok.generate_token("MRN12345")
        t2 = tok.generate_token("MRN67890")
        assert t1 != t2

    def test_generate_token_tenant_isolation(self) -> None:
        tok1 = PHITokenizer(secret_key="test-key", tenant_id="hospital-1")
        tok2 = PHITokenizer(secret_key="test-key", tenant_id="hospital-2")
        t1 = tok1.generate_token("MRN12345")
        t2 = tok2.generate_token("MRN12345")
        assert t1 != t2

    def test_tokenize_patient_resource(self) -> None:
        tok = PHITokenizer(secret_key="test-key", tenant_id="hospital-1")
        result = tok.tokenize_resource(_patient_resource())
        assert result.patient_token.startswith("PT_")
        assert result.original_resource_type == "Patient"
        assert result.phi_fields_tokenized > 0
        assert result.resource["name"] == [{"text": "[REDACTED]"}]
        assert result.resource["birthDate"] == "REDACTED"

    def test_tokenize_observation_replaces_subject(self) -> None:
        tok = PHITokenizer(secret_key="test-key", tenant_id="hospital-1")
        result = tok.tokenize_resource(_observation_resource())
        assert "Patient/" in result.resource["subject"]["reference"]
        assert result.resource["subject"]["reference"].startswith("Patient/PT_")

    def test_tokenize_adds_security_label(self) -> None:
        tok = PHITokenizer(secret_key="test-key", tenant_id="hospital-1")
        result = tok.tokenize_resource(_patient_resource())
        security = result.resource["meta"]["security"]
        assert any(s["code"] == "R" for s in security)

    def test_tokenize_telecom_redacted(self) -> None:
        tok = PHITokenizer(secret_key="test-key", tenant_id="hospital-1")
        result = tok.tokenize_resource(_patient_resource())
        assert result.resource["telecom"][0]["value"] == "[REDACTED]"

    def test_scan_for_phi_detects_ssn(self) -> None:
        findings = PHITokenizer.scan_for_phi("Patient SSN is 123-45-6789")
        assert "SSN" in findings

    def test_scan_for_phi_detects_email(self) -> None:
        findings = PHITokenizer.scan_for_phi("Contact: john@hospital.org")
        assert "Email" in findings

    def test_scan_for_phi_clean_text(self) -> None:
        findings = PHITokenizer.scan_for_phi("Patient has elevated WBC count")
        assert len(findings) == 0

    def test_hipaa_identifiers_count(self) -> None:
        assert len(HIPAA_IDENTIFIERS) == 18


# ── FHIRBundleValidator tests ──

class TestFHIRBundleValidator:
    def test_valid_bundle(self) -> None:
        validator = FHIRBundleValidator()
        result = validator.validate_bundle(_valid_bundle())
        assert result.is_valid is True
        assert result.resource_count == 2

    def test_invalid_not_bundle(self) -> None:
        validator = FHIRBundleValidator()
        result = validator.validate_bundle({"resourceType": "Patient"})
        assert result.is_valid is False
        assert "Bundle" in result.errors[0]

    def test_empty_bundle(self) -> None:
        validator = FHIRBundleValidator()
        result = validator.validate_bundle({"resourceType": "Bundle", "entry": []})
        assert result.is_valid is False

    def test_missing_required_field(self) -> None:
        validator = FHIRBundleValidator()
        bundle = {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Observation"}}],
        }
        result = validator.validate_bundle(bundle)
        assert result.is_valid is False
        assert any("status" in e for e in result.errors)

    def test_validate_single_resource(self) -> None:
        validator = FHIRBundleValidator()
        result = validator.validate_resource(_patient_resource())
        assert result.is_valid is True

    def test_resource_type_counts(self) -> None:
        validator = FHIRBundleValidator()
        result = validator.validate_bundle(_valid_bundle())
        assert result.resource_types.get("Patient", 0) == 1
        assert result.resource_types.get("Observation", 0) == 1


# ── ResourceDeduplicator tests ──

class TestResourceDeduplicator:
    def test_no_duplicates(self) -> None:
        dedup = ResourceDeduplicator()
        resources = [_patient_resource(), _observation_resource()]
        result = dedup.deduplicate(resources)
        assert result.duplicates_removed == 0
        assert len(result.unique_resources) == 2

    def test_removes_duplicates(self) -> None:
        dedup = ResourceDeduplicator()
        resources = [_patient_resource(), _patient_resource()]
        result = dedup.deduplicate(resources)
        assert result.duplicates_removed == 1
        assert len(result.unique_resources) == 1

    def test_reset_clears_state(self) -> None:
        dedup = ResourceDeduplicator()
        resources = [_patient_resource()]
        dedup.deduplicate(resources)
        dedup.reset()
        result = dedup.deduplicate(resources)
        assert result.duplicates_removed == 0

    def test_different_types_not_duplicates(self) -> None:
        dedup = ResourceDeduplicator()
        r1 = {"resourceType": "Patient", "id": "1"}
        r2 = {"resourceType": "Observation", "id": "1"}
        result = dedup.deduplicate([r1, r2])
        assert result.duplicates_removed == 0
        assert len(result.unique_resources) == 2
