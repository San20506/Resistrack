# M5.4 -- Production Monitoring and Alerting
**Phase:** 5 -- MLOps and Governance
**Duration:** ~1 day
**Status:** Not Started

## Objective
Implement comprehensive production monitoring and alerting to ensure the health, performance, and reliability of the ResisTrack platform.

## Scope
This module involves creating CloudWatch dashboards to monitor inference latency (p95), error rates, and SageMaker endpoint health. It configures CloudWatch alarms that trigger SNS notifications for critical events: latency > 1500ms, error rate > 1%, and PSI drift alerts. All monitoring and logging must use de-identified tokens to ensure no PHI is exposed in CloudWatch logs.

## Dependencies
- **Depends on:** M5.2
- **Depended on by:** None directly

## Inputs
- Real-time metrics from SageMaker and Lambda
- Error logs and health check results
- PSI drift metrics from the MLOps pipeline

## Outputs
- Production CloudWatch dashboards
- Configured CloudWatch alarms and SNS topics
- Monitoring access for the IT Admin role

## Implementation Notes
Dashboards should provide a clear, real-time view of system health. Alarms must be tuned to minimize false positives while ensuring critical issues are addressed immediately. The IT Admin role should have dedicated access to these monitoring tools.

## Agent Rules
- RULE-DATA-01: No PHI in CloudWatch logs; use de-identified tokens only.
- RULE-CDS-01: Monitor for p95 latency <= 2000ms.

## Done When
- [ ] CloudWatch dashboard displays latency p95, error rate, and endpoint health.
- [ ] Alarms successfully trigger SNS notifications when latency exceeds 1500ms.
- [ ] Alarms trigger on error rates greater than 1%.
- [ ] PSI drift alarm is configured and functional.
- [ ] All log entries are verified to be free of PHI.
- [ ] Dashboard is accessible and verified for the IT Admin role.
