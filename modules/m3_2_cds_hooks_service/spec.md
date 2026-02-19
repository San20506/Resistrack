# M3.2 -- CDS Hooks Service
**Phase:** 3 -- Clinical Integration
**Duration:** ~3 days
**Status:** Not Started

## Objective
Deploy a high-performance CDS Hooks service to provide real-time AMR risk assessments and clinical decision support within the EHR workflow.

## Scope
This module registers and manages CDS Hooks for patient-view, order-sign, and encounter-discharge events. It performs synchronous queries to the SageMaker inference endpoint with a strict 2-second SLA. The service generates CDS Cards containing Risk Tiers, top-3 SHAP factors in plain English, and stewardship recommendations. It provides three clinician response options: Acknowledged, Override (with reason), and Escalate. All interactions are logged to an RDS audit trail.

## Dependencies
- **Depends on:** M2.7, M3.1
- **Depended on by:** M3.3, M3.4

## Inputs
- CDS Hook trigger requests (FHIR resources, user context)
- Real-time inference results from SageMaker endpoint
- Clinician response data from CDS Cards

## Outputs
- CDS Cards with risk scores and recommendations
- Audit logs of clinician actions in RDS
- Trigger signals for notification dispatch (M3.4)

## Implementation Notes
The service must implement a timeout mechanism at 1500ms to ensure the 2000ms p95 SLA is met. If the timeout is reached, a cached score must be returned with the CACHED_RESULT=true flag. SHAP explanations must be translated from technical feature names to clinician-friendly language.

## Agent Rules
- RULE-CDS-01: Maintain p95 latency <= 2000ms.
- RULE-CDS-02: Include "Why this alert?" SHAP link for HIGH and CRITICAL tiers.
- RULE-CDS-03: Provide exactly three response options: Acknowledged, Override, Escalate.
- RULE-CDS-04: Monitor and log all override events.
- RULE-DATA-05: Log all CDS interactions to the RDS audit trail.

## Done When
- [ ] CDS Hooks are registered for patient-view, order-sign, and encounter-discharge.
- [ ] CDS Cards render correctly in the Epic sandbox within the 2s SLA.
- [ ] Cached fallback mechanism activates when latency exceeds 1.5s.
- [ ] All three clinician response options are functional and captured.
- [ ] Every response is successfully logged to the RDS audit trail.
- [ ] SHAP "Why this alert?" link is present and functional for HIGH/CRITICAL alerts.
