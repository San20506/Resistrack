# M3.4 -- Multi-Role Notification Dispatch
**Phase:** 3 -- Clinical Integration
**Duration:** ~2 days
**Status:** Not Started

## Objective
Route critical AMR risk alerts to the appropriate clinical teams via AWS SNS while maintaining strict HIPAA compliance.

## Scope
This module implements a notification routing system based on Risk Tiers. HIGH and CRITICAL alerts are dispatched to the Pharmacy team for immediate stewardship review. CRITICAL alerts with an MDRO flag are also routed to the Infection Control Officer. All notification payloads are strictly de-identified, containing only the patient_token and no Protected Health Information (PHI).

## Dependencies
- **Depends on:** M3.2
- **Depended on by:** None directly

## Inputs
- Risk assessment results (Risk Tier, MDRO flag, patient_token)
- Role-based subscription lists for SNS topics

## Outputs
- SNS notifications dispatched to Pharmacy and Infection Control teams
- Delivery confirmation logs

## Implementation Notes
The dispatch logic must ensure that no PHI leaves the VPC. SNS payloads must be audited to verify they only contain the patient_token. The system should support multiple delivery protocols (e.g., SMS, Email, PagerDuty) as configured in the SNS topics.

## Agent Rules
- RULE-DATA-01: No PHI outside VPC; SNS payloads must be de-identified.
- RULE-DATA-02: Ensure secure data handling within the VPC.

## Done When
- [ ] SNS topics are configured for each clinical role.
- [ ] HIGH/CRITICAL alerts successfully trigger Pharmacy notifications.
- [ ] CRITICAL alerts with MDRO flags trigger Infection Control notifications.
- [ ] All payloads are verified to contain zero PHI (patient_token only).
- [ ] Notification delivery is confirmed through CloudWatch logs.
- [ ] End-to-end test with a CRITICAL patient assessment passes.
