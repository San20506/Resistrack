"""FHIR R4 resource mapping from HL7 v2.x messages."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from resistrack.transformers.hl7_parser import HL7Message


def _generate_id() -> str:
    """Generate a UUID for FHIR resource IDs."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return datetime.now(tz=timezone.utc).isoformat()


def map_adt_to_patient(message: HL7Message) -> dict[str, Any]:
    """Map HL7 ADT message to FHIR R4 Patient resource.

    Extracts patient demographics from PID segment.

    Args:
        message: Parsed HL7 ADT message.

    Returns:
        FHIR R4 Patient resource as dictionary.

    Raises:
        ValueError: If PID segment is missing.
    """
    pid = message.get_segment("PID")
    if pid is None:
        msg = "ADT message missing PID segment"
        raise ValueError(msg)

    # PID-3: Patient Identifier
    patient_id = pid.get_component(3, 1)
    # PID-5: Patient Name (Family^Given)
    family_name = pid.get_component(5, 1)
    given_name = pid.get_component(5, 2)
    # PID-7: Date of Birth
    birth_date = pid.get_field(7)
    # PID-8: Sex
    gender_map = {"M": "male", "F": "female", "O": "other", "U": "unknown"}
    gender = gender_map.get(pid.get_field(8), "unknown")

    return {
        "resourceType": "Patient",
        "id": _generate_id(),
        "identifier": [
            {
                "system": "urn:resistrack:patient",
                "value": patient_id,
            }
        ],
        "name": [
            {
                "use": "official",
                "family": family_name,
                "given": [given_name] if given_name else [],
            }
        ],
        "gender": gender,
        "birthDate": _format_hl7_date(birth_date),
        "meta": {
            "lastUpdated": _now_iso(),
            "profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"],
        },
    }


def map_adt_to_encounter(message: HL7Message) -> dict[str, Any]:
    """Map HL7 ADT message to FHIR R4 Encounter resource.

    Extracts visit information from PV1 segment.

    Args:
        message: Parsed HL7 ADT message.

    Returns:
        FHIR R4 Encounter resource as dictionary.

    Raises:
        ValueError: If PV1 segment is missing.
    """
    pv1 = message.get_segment("PV1")
    if pv1 is None:
        msg = "ADT message missing PV1 segment"
        raise ValueError(msg)

    pid = message.get_segment("PID")
    patient_id = pid.get_component(3, 1) if pid else "unknown"

    # PV1-2: Patient Class (I=inpatient, O=outpatient, E=emergency)
    patient_class = pv1.get_field(2)
    class_map = {
        "I": {"code": "IMP", "display": "inpatient encounter"},
        "O": {"code": "AMB", "display": "ambulatory"},
        "E": {"code": "EMER", "display": "emergency"},
    }
    encounter_class = class_map.get(patient_class, {"code": "IMP", "display": "inpatient encounter"})

    # PV1-3: Assigned Patient Location
    location = pv1.get_component(3, 1)
    # PV1-44: Admit Date/Time
    admit_date = pv1.get_field(44)

    return {
        "resourceType": "Encounter",
        "id": _generate_id(),
        "status": "in-progress",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            **encounter_class,
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "location": [
            {
                "location": {"display": location},
                "status": "active",
            }
        ]
        if location
        else [],
        "period": {"start": _format_hl7_datetime(admit_date)} if admit_date else {},
        "meta": {"lastUpdated": _now_iso()},
    }


def map_oru_to_observation(message: HL7Message) -> list[dict[str, Any]]:
    """Map HL7 ORU message to FHIR R4 Observation resources.

    Creates one Observation per OBX segment.

    Args:
        message: Parsed HL7 ORU message.

    Returns:
        List of FHIR R4 Observation resources.
    """
    observations: list[dict[str, Any]] = []
    pid = message.get_segment("PID")
    patient_id = pid.get_component(3, 1) if pid else "unknown"

    for obx in message.get_all_segments("OBX"):
        # OBX-2: Value Type
        value_type = obx.get_field(2)
        # OBX-3: Observation Identifier (code^display^system)
        obs_code = obx.get_component(3, 1)
        obs_display = obx.get_component(3, 2)
        obs_system = obx.get_component(3, 3) or "http://loinc.org"
        # OBX-5: Observation Value
        obs_value = obx.get_field(5)
        # OBX-6: Units
        obs_units = obx.get_component(6, 1)
        # OBX-8: Abnormal Flags
        abnormal_flag = obx.get_field(8)
        # OBX-11: Observation Result Status
        status_map = {"F": "final", "P": "preliminary", "C": "corrected"}
        status = status_map.get(obx.get_field(11), "final")
        # OBX-14: Date/Time of Observation
        obs_date = obx.get_field(14)

        observation: dict[str, Any] = {
            "resourceType": "Observation",
            "id": _generate_id(),
            "status": status,
            "code": {
                "coding": [
                    {
                        "system": obs_system,
                        "code": obs_code,
                        "display": obs_display,
                    }
                ]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": _format_hl7_datetime(obs_date) if obs_date else _now_iso(),
            "meta": {"lastUpdated": _now_iso()},
        }

        # Set value based on type
        if value_type == "NM" and obs_value:
            try:
                observation["valueQuantity"] = {
                    "value": float(obs_value),
                    "unit": obs_units,
                    "system": "http://unitsofmeasure.org",
                }
            except ValueError:
                observation["valueString"] = obs_value
        elif value_type == "ST" or value_type == "TX":
            observation["valueString"] = obs_value
        elif value_type == "CE" or value_type == "CWE":
            observation["valueCodeableConcept"] = {
                "coding": [{"code": obs_value}]
            }
        else:
            observation["valueString"] = obs_value

        # Interpretation from abnormal flags
        if abnormal_flag:
            flag_map = {
                "H": {"code": "H", "display": "High"},
                "L": {"code": "L", "display": "Low"},
                "HH": {"code": "HH", "display": "Critical high"},
                "LL": {"code": "LL", "display": "Critical low"},
                "N": {"code": "N", "display": "Normal"},
                "A": {"code": "A", "display": "Abnormal"},
                "R": {"code": "R", "display": "Resistant"},
                "S": {"code": "S", "display": "Susceptible"},
            }
            if abnormal_flag in flag_map:
                observation["interpretation"] = [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                **flag_map[abnormal_flag],
                            }
                        ]
                    }
                ]

        observations.append(observation)

    return observations


def map_rde_to_medication_request(message: HL7Message) -> dict[str, Any]:
    """Map HL7 RDE message to FHIR R4 MedicationRequest resource.

    Args:
        message: Parsed HL7 RDE message.

    Returns:
        FHIR R4 MedicationRequest resource.

    Raises:
        ValueError: If RXE segment is missing.
    """
    rxe = message.get_segment("RXE")
    if rxe is None:
        msg = "RDE message missing RXE segment"
        raise ValueError(msg)

    pid = message.get_segment("PID")
    patient_id = pid.get_component(3, 1) if pid else "unknown"

    # RXE-2: Give Code (code^display^system)
    med_code = rxe.get_component(2, 1)
    med_display = rxe.get_component(2, 2)
    med_system = rxe.get_component(2, 3) or "http://www.nlm.nih.gov/research/umls/rxnorm"
    # RXE-3: Give Amount - Minimum
    dose_value = rxe.get_field(3)
    # RXE-5: Give Units
    dose_unit = rxe.get_component(5, 1)
    # RXE-6: Give Dosage Form
    dosage_form = rxe.get_component(6, 1)
    # RXE-1: Quantity/Timing (frequency)
    frequency = rxe.get_component(1, 1)

    return {
        "resourceType": "MedicationRequest",
        "id": _generate_id(),
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [
                {
                    "system": med_system,
                    "code": med_code,
                    "display": med_display,
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "dosageInstruction": [
            {
                "text": f"{dose_value} {dose_unit} {dosage_form} {frequency}".strip(),
                "doseAndRate": [
                    {
                        "doseQuantity": {
                            "value": float(dose_value) if dose_value else 0,
                            "unit": dose_unit,
                        }
                    }
                ]
                if dose_value
                else [],
            }
        ],
        "meta": {"lastUpdated": _now_iso()},
    }


def _format_hl7_date(hl7_date: str) -> str:
    """Convert HL7 date (YYYYMMDD) to FHIR date (YYYY-MM-DD)."""
    if not hl7_date or len(hl7_date) < 8:
        return hl7_date
    return f"{hl7_date[:4]}-{hl7_date[4:6]}-{hl7_date[6:8]}"


def _format_hl7_datetime(hl7_dt: str) -> str:
    """Convert HL7 datetime (YYYYMMDDHHMMSS) to ISO 8601."""
    if not hl7_dt or len(hl7_dt) < 8:
        return hl7_dt
    date_part = f"{hl7_dt[:4]}-{hl7_dt[4:6]}-{hl7_dt[6:8]}"
    if len(hl7_dt) >= 12:
        time_part = f"{hl7_dt[8:10]}:{hl7_dt[10:12]}"
        if len(hl7_dt) >= 14:
            time_part += f":{hl7_dt[12:14]}"
        return f"{date_part}T{time_part}Z"
    return date_part
