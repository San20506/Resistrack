# M1.5 -- Data Storage Layer
**Phase:** 1 -- Foundation and Data Ingestion
**Duration:** ~2 days
**Status:** Not Started

## Objective
Provision and configure secure, encrypted storage for clinical data, model outputs, and audit logs.

## Scope
- AWS HealthLake (FHIR R4) for primary patient data storage with AES-256 encryption.
- Amazon RDS (PostgreSQL) for model outputs, alert records, and audit logs.
- S3 buckets for raw clinical notes, model artifacts, and reports.
- Implementation of a 7-year audit log retention policy.

## Dependencies
- **Depends on:** M1.1
- **Depended on by:** M1.6, M2.1

## Inputs
- VPC configuration from M1.1.
- KMS keys for encryption.

## Outputs
- HealthLake FHIR store endpoint.
- RDS PostgreSQL instance and connection strings.
- S3 bucket names and ARNs.

## Implementation Notes
- All storage must be encrypted at rest using KMS CMKs.
- Storage must only be accessible from within the VPC.
- Configure S3 lifecycle policies for cost-effective long-term storage.
- RDS must use a schema optimized for audit and model output tracking.

## Agent Rules
- RULE-DATA-01: PHI must remain within the VPC boundary.
- RULE-DATA-05: 7-year retention policy for audit logs.

## Done When
- [ ] HealthLake FHIR store is operational with AES-256/KMS encryption.
- [ ] RDS PostgreSQL is provisioned with the audit schema.
- [ ] S3 buckets are created with encryption and lifecycle policies.
- [ ] 7-year retention policy is configured for audit logs.
- [ ] All storage is accessible only from within the VPC.
