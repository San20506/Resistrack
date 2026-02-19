# HIPAA Compliance Checklist — ResisTrack

**Document Version:** 1.0.0
**Last Updated:** 2026-02-18
**Status:** In Progress

## Administrative Safeguards (45 CFR § 164.308)

| # | Requirement | Status | Implementation |
|---|------------|--------|----------------|
| A1 | Security Management Process | ✅ Done | Risk analysis documented; KMS encryption for all data at rest |
| A2 | Assigned Security Responsibility | ⬜ Pending | Security officer designation required |
| A3 | Workforce Security | ✅ Done | RBAC with 6 roles (PHYSICIAN, PHARMACIST, INFECTION_CONTROL, NURSE, ADMIN, READONLY) |
| A4 | Information Access Management | ✅ Done | Role-based permissions via RBACEnforcer; least-privilege IAM policies |
| A5 | Security Awareness Training | ⬜ Pending | Training program to be developed |
| A6 | Security Incident Procedures | ⬜ Pending | Incident response plan to be documented |
| A7 | Contingency Plan | ⬜ Pending | Disaster recovery plan required |
| A8 | Evaluation | ⬜ Pending | Annual security evaluation process |
| A9 | BAA with AWS | ⬜ Pending | AWS BAA to be executed before production |

## Physical Safeguards (45 CFR § 164.310)

| # | Requirement | Status | Implementation |
|---|------------|--------|----------------|
| P1 | Facility Access Controls | ✅ Done | AWS data centers; SOC 2 Type II certified |
| P2 | Workstation Use | ⬜ Pending | Policy required for clinical workstation access |
| P3 | Workstation Security | ⬜ Pending | Endpoint security policy |
| P4 | Device and Media Controls | ✅ Done | No local PHI storage; all data in encrypted AWS services |

## Technical Safeguards (45 CFR § 164.312)

| # | Requirement | Status | Implementation |
|---|------------|--------|----------------|
| T1 | Access Control | ✅ Done | SMART on FHIR OAuth 2.0 + JWT; role-based access |
| T2 | Audit Controls | ✅ Done | RDS audit log table; CloudTrail enabled |
| T3 | Integrity Controls | ✅ Done | KMS CMK encryption; data quality flags on all predictions |
| T4 | Person/Entity Authentication | ✅ Done | OAuth 2.0 with OIDC; SMART launch from EHR |
| T5 | Transmission Security | ✅ Done | TLS 1.2+ enforced; VPC private subnets only |

## PHI Protection Measures

| # | Measure | Status | Implementation |
|---|---------|--------|----------------|
| PHI-1 | No PHI in logs | ✅ Done | CloudWatch log scrubbing; tokenized patient IDs |
| PHI-2 | No PHI in JWTs | ✅ Done | TokenPayload validator rejects PHI fields |
| PHI-3 | No PHI in localStorage | ✅ Done | Dashboard uses session-only tokens; no PHI persisted client-side |
| PHI-4 | No PHI outside VPC | ✅ Done | VPC with private subnets only; no public endpoints for data |
| PHI-5 | PHI tokenization | ✅ Done | Patient identifiers replaced with opaque tokens |
| PHI-6 | Encryption at rest | ✅ Done | KMS CMKs per tenant for all data stores |
| PHI-7 | Encryption in transit | ✅ Done | TLS 1.2+ on all connections |

## Business Associate Agreements

| Partner | BAA Status | Notes |
|---------|-----------|-------|
| AWS | ⬜ Required | Must execute before handling real PHI |
| Epic | ⬜ Required | For SMART on FHIR integration |
| Cerner | ⬜ Required | For SMART on FHIR integration |
