# M1.4 -- FHIR Ingestion and PHI Tokenization
**Phase:** 1 -- Foundation and Data Ingestion
**Duration:** ~2 days
**Status:** Not Started

## Objective
Validate incoming FHIR bundles, de-duplicate records, and tokenize PHI to ensure data privacy before storage and analysis.

## Scope
- Lambda function for FHIR bundle validation and de-duplication.
- PHI tokenization: Replace patient identifiers (name, DOB, MRN) with an internal `patient_token`.
- Route validated and tokenized bundles to AWS HealthLake.
- Audit logging of all ingestion activities.

## Dependencies
- **Depends on:** M1.1, M1.3
- **Depended on by:** M1.5, M2.1

## Inputs
- FHIR R4 JSON bundles from M1.3.
- Tokenization keys from KMS.

## Outputs
- Tokenized FHIR resources in AWS HealthLake.
- Audit logs in RDS.
- De-duplication metadata.

## Implementation Notes
- Use AWS Lambda for serverless processing.
- Tokenize all 18 HIPAA identifiers.
- Ensure zero PHI is written to CloudWatch logs.
- Use RDS for maintaining an audit trail of tokenization events.

## Agent Rules
- RULE-DATA-01: PHI must remain within the VPC boundary.
- RULE-DATA-02: No raw patient identifiers as features; use tokenized IDs only.
- RULE-DATA-05: Audit logging for all data access and modifications.

## Done When
- [ ] Lambda validates FHIR bundles and rejects malformed ones.
- [ ] De-duplication by resource ID is implemented.
- [ ] All 18 HIPAA identifiers are tokenized to `patient_token`.
- [ ] Validated bundles are stored in HealthLake.
- [ ] Audit trail entry is written to RDS.
- [ ] Zero PHI exists in CloudWatch logs.
