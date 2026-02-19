# M1.3 -- HL7 v2 to FHIR R4 Transformer
**Phase:** 1 -- Foundation and Data Ingestion
**Duration:** ~3 days
**Status:** Not Started

## Objective
Transform legacy HL7 v2 clinical messages into modern FHIR R4 resources for downstream processing and storage.

## Scope
- Deploy Mirth Connect integration engine.
- Build transformation channels for:
    - HL7 ADT A01 -> FHIR Patient and Encounter.
    - HL7 ORU -> FHIR Observation.
    - HL7 RDE -> FHIR MedicationRequest.
- Validate FHIR R4 bundle schema conformance.

## Dependencies
- **Depends on:** None (independent, can be unit-tested with synthetic HL7 messages)
- **Depended on by:** M1.4

## Inputs
- Raw HL7 v2 messages (ADT, ORU, RDE).
- FHIR R4 schema definitions.

## Outputs
- Valid FHIR R4 JSON bundles.
- Transformation logs and error reports.

## Implementation Notes
- Use Mirth Connect JavaScript or Java transformers for mapping.
- Ensure all output bundles pass FHIR R4 validation.
- Maintain mapping documentation for each HL7 segment to FHIR element.

## Agent Rules
- Code Quality Standards: Unit tests with synthetic HL7 messages must pass.

## Done When
- [ ] Mirth Connect deployed.
- [ ] ADT A01 transforms to valid FHIR Patient+Encounter bundle.
- [ ] ORU transforms to FHIR Observation.
- [ ] RDE transforms to FHIR MedicationRequest.
- [ ] All outputs pass FHIR R4 validation.
- [ ] Unit tests with synthetic HL7 messages pass.
