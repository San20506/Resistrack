# M3.3 -- Override Rate Monitoring
**Phase:** 3 -- Clinical Integration
**Duration:** ~1 day
**Status:** Not Started

## Objective
Monitor clinician override rates to identify potential alert fatigue and ensure the clinical relevance of AMR risk notifications.

## Scope
This module tracks the override rate for each clinician using a rolling 30-day window. It analyzes data from the RDS audit trail to calculate the percentage of alerts that were overridden versus acknowledged or escalated. If a clinician's override rate exceeds 60%, the system automatically generates a model feedback report for clinical informatics review.

## Dependencies
- **Depends on:** M3.2
- **Depended on by:** None directly (feeds into clinical informatics review)

## Inputs
- Clinician response logs from RDS audit trail
- Alert frequency data per clinician

## Outputs
- Per-clinician override rate metrics
- Automated model feedback reports for high-override scenarios

## Implementation Notes
The monitoring logic should be implemented as a scheduled task (e.g., AWS Lambda cron) that aggregates logs daily. The 60% threshold is a critical indicator of alert fatigue or model misalignment and must trigger an immediate notification to the clinical informatics team.

## Agent Rules
- RULE-CDS-04: Implement specific monitoring for clinician overrides.

## Done When
- [ ] Override rate is calculated per clinician over a rolling 30-day window.
- [ ] Automated report generation is triggered when the 60% threshold is exceeded.
- [ ] Reports include clinician ID, override count, total alerts, and calculated rate.
- [ ] Reports are successfully delivered to the clinical informatics team.
