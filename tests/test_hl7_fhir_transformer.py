"""Tests for HL7-FHIR Transformer module (M1.3)."""

import pytest

from resistrack.transformers.hl7_parser import HL7Message, parse_hl7_message
from resistrack.transformers.fhir_mapper import (
    map_adt_to_patient,
    map_adt_to_encounter,
    map_oru_to_observation,
    map_rde_to_medication_request,
)
from resistrack.transformers.fhir_validator import validate_fhir_resource


# --- Synthetic HL7 Messages ---

ADT_A01_MESSAGE = (
    "MSH|^~\\&|EPIC|HOSP|RESISTRACK|RT|20250101120000||ADT^A01|MSG001|P|2.5\r"
    "PID|||PAT001^^^HOSP^MR||DOE^JOHN^A||19800115|M|||123 MAIN ST^^CITY^ST^12345\r"
    "PV1||I|ICU^BED01^01||||ATT001^SMITH^JANE|||MED||||||||VIS001|||||||||||||||||||||||||20250101080000\r"
)

ORU_R01_MESSAGE = (
    "MSH|^~\\&|LAB|HOSP|RESISTRACK|RT|20250101140000||ORU^R01|MSG002|P|2.5\r"
    "PID|||PAT001^^^HOSP^MR||DOE^JOHN\r"
    "OBX|1|NM|2160-0^Creatinine^LN||1.2|mg/dL|0.7-1.3||||F|||20250101130000\r"
    "OBX|2|NM|6690-2^WBC^LN||15.5|10*3/uL|4.5-11.0|H|||F|||20250101130000\r"
    "OBX|3|ST|6463-4^Bacteria identified^LN||E.coli||||||F|||20250101130000\r"
)

RDE_O11_MESSAGE = (
    "MSH|^~\\&|EPIC|HOSP|RESISTRACK|RT|20250101150000||RDE^O11|MSG003|P|2.5\r"
    "PID|||PAT001^^^HOSP^MR||DOE^JOHN\r"
    "RXE|Q6H|1234^Vancomycin^RxNorm|1000|1000|mg^mg|IV|||||||||||||||||||\r"
)


# --- HL7 Parser Tests ---


class TestHL7Parser:
    def test_parse_adt_message(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        assert msg.message_type == "ADT"
        assert msg.trigger_event == "A01"

    def test_parse_oru_message(self) -> None:
        msg = parse_hl7_message(ORU_R01_MESSAGE)
        assert msg.message_type == "ORU"
        assert msg.trigger_event == "R01"

    def test_parse_segments(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        assert msg.get_segment("MSH") is not None
        assert msg.get_segment("PID") is not None
        assert msg.get_segment("PV1") is not None

    def test_parse_pid_fields(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        pid = msg.get_segment("PID")
        assert pid is not None
        assert pid.get_component(3, 1) == "PAT001"
        assert pid.get_component(5, 1) == "DOE"
        assert pid.get_component(5, 2) == "JOHN"

    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_hl7_message("")

    def test_missing_msh_raises(self) -> None:
        with pytest.raises(ValueError, match="MSH"):
            parse_hl7_message("PID|||PAT001")

    def test_get_all_obx_segments(self) -> None:
        msg = parse_hl7_message(ORU_R01_MESSAGE)
        obx_segments = msg.get_all_segments("OBX")
        assert len(obx_segments) == 3


# --- FHIR Mapper Tests ---


class TestADTToPatient:
    def test_patient_resource_type(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        patient = map_adt_to_patient(msg)
        assert patient["resourceType"] == "Patient"

    def test_patient_identifier(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        patient = map_adt_to_patient(msg)
        assert patient["identifier"][0]["value"] == "PAT001"

    def test_patient_name(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        patient = map_adt_to_patient(msg)
        assert patient["name"][0]["family"] == "DOE"
        assert "JOHN" in patient["name"][0]["given"]

    def test_patient_gender(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        patient = map_adt_to_patient(msg)
        assert patient["gender"] == "male"

    def test_patient_birth_date(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        patient = map_adt_to_patient(msg)
        assert patient["birthDate"] == "1980-01-15"

    def test_patient_validates(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        patient = map_adt_to_patient(msg)
        result = validate_fhir_resource(patient)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_missing_pid_raises(self) -> None:
        msg = HL7Message(raw="", segments=[])
        with pytest.raises(ValueError, match="PID"):
            map_adt_to_patient(msg)


class TestADTToEncounter:
    def test_encounter_resource_type(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        encounter = map_adt_to_encounter(msg)
        assert encounter["resourceType"] == "Encounter"

    def test_encounter_class_inpatient(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        encounter = map_adt_to_encounter(msg)
        assert encounter["class"]["code"] == "IMP"

    def test_encounter_validates(self) -> None:
        msg = parse_hl7_message(ADT_A01_MESSAGE)
        encounter = map_adt_to_encounter(msg)
        result = validate_fhir_resource(encounter)
        assert result.is_valid, f"Validation errors: {result.errors}"


class TestORUToObservation:
    def test_observation_count(self) -> None:
        msg = parse_hl7_message(ORU_R01_MESSAGE)
        observations = map_oru_to_observation(msg)
        assert len(observations) == 3

    def test_numeric_observation(self) -> None:
        msg = parse_hl7_message(ORU_R01_MESSAGE)
        observations = map_oru_to_observation(msg)
        creatinine = observations[0]
        assert creatinine["resourceType"] == "Observation"
        assert creatinine["valueQuantity"]["value"] == 1.2
        assert creatinine["valueQuantity"]["unit"] == "mg/dL"

    def test_high_flag_interpretation(self) -> None:
        msg = parse_hl7_message(ORU_R01_MESSAGE)
        observations = map_oru_to_observation(msg)
        wbc = observations[1]
        assert "interpretation" in wbc
        assert wbc["interpretation"][0]["coding"][0]["code"] == "H"

    def test_string_observation(self) -> None:
        msg = parse_hl7_message(ORU_R01_MESSAGE)
        observations = map_oru_to_observation(msg)
        bacteria = observations[2]
        assert bacteria["valueString"] == "E.coli"

    def test_observations_validate(self) -> None:
        msg = parse_hl7_message(ORU_R01_MESSAGE)
        observations = map_oru_to_observation(msg)
        for obs in observations:
            result = validate_fhir_resource(obs)
            assert result.is_valid, f"Validation errors: {result.errors}"


class TestRDEToMedicationRequest:
    def test_medication_resource_type(self) -> None:
        msg = parse_hl7_message(RDE_O11_MESSAGE)
        med = map_rde_to_medication_request(msg)
        assert med["resourceType"] == "MedicationRequest"

    def test_medication_code(self) -> None:
        msg = parse_hl7_message(RDE_O11_MESSAGE)
        med = map_rde_to_medication_request(msg)
        coding = med["medicationCodeableConcept"]["coding"][0]
        assert coding["code"] == "1234"
        assert coding["display"] == "Vancomycin"

    def test_medication_validates(self) -> None:
        msg = parse_hl7_message(RDE_O11_MESSAGE)
        med = map_rde_to_medication_request(msg)
        result = validate_fhir_resource(med)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_missing_rxe_raises(self) -> None:
        msg = HL7Message(raw="", segments=[])
        with pytest.raises(ValueError, match="RXE"):
            map_rde_to_medication_request(msg)


# --- FHIR Validator Tests ---


class TestFHIRValidator:
    def test_missing_resource_type(self) -> None:
        result = validate_fhir_resource({"id": "123"})
        assert not result.is_valid

    def test_unknown_resource_type(self) -> None:
        result = validate_fhir_resource({"resourceType": "Unicorn", "id": "123"})
        assert not result.is_valid

    def test_valid_patient(self) -> None:
        patient = {
            "resourceType": "Patient",
            "id": "test-123",
            "meta": {"lastUpdated": "2025-01-01T00:00:00Z"},
        }
        result = validate_fhir_resource(patient)
        assert result.is_valid

    def test_missing_required_field(self) -> None:
        encounter = {
            "resourceType": "Encounter",
            "id": "test-123",
            # Missing 'status' and 'class'
        }
        result = validate_fhir_resource(encounter)
        assert not result.is_valid
        assert any("status" in e for e in result.errors)
