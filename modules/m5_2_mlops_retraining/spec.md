# M5.2 -- SageMaker MLOps Retraining Pipeline
**Phase:** 5 -- MLOps and Governance
**Duration:** ~2 days
**Status:** Not Started

## Objective
Establish an automated MLOps pipeline for monthly model retraining, drift monitoring, and safe deployment.

## Scope
This module implements a SageMaker Pipeline for automated monthly retraining on new hospital data. It includes Population Stability Index (PSI) drift monitoring, which triggers emergency retraining if PSI exceeds 0.20. The pipeline uses a blue/green deployment strategy, where new models receive 10% of traffic initially and are auto-promoted if their AUC is within 0.02 of the previous model over 72 hours. It also maintains rollback capability for the previous two production versions.

## Dependencies
- **Depends on:** M5.1
- **Depended on by:** M5.3, M5.4

## Inputs
- New hospital data from HealthLake/RDS
- Current production model artifacts
- PSI drift thresholds and monitoring metrics

## Outputs
- Automated retraining pipeline in SageMaker
- New model versions with semantic versioning (MAJOR.MINOR.PATCH)
- Blue/green deployment configurations

## Implementation Notes
The retraining pipeline must be robust and handle data quality issues gracefully. Rollback capability is critical and must be achievable in less than 15 minutes. All prediction data used for monitoring must be de-identified before storage.

## Agent Rules
- RULE-MLOPS-01: Monthly retraining and PSI-triggered emergency retraining.
- RULE-MLOPS-02: Blue/green deployment with 10% canary traffic.
- RULE-MLOPS-03: Use semantic versioning for all model artifacts.
- RULE-MLOPS-04: Ensure rollback capability < 15 minutes.
- RULE-MLOPS-05: Store only de-identified prediction data for monitoring.

## Done When
- [ ] Monthly retraining pipeline executes successfully end-to-end.
- [ ] PSI monitoring is active and triggers retraining at the 0.20 threshold.
- [ ] Blue/green deployment with 10% canary traffic is operational.
- [ ] Auto-promotion logic correctly evaluates AUC over a 72-hour window.
- [ ] Rollback to the previous 2 versions is verified to take < 15 minutes.
- [ ] Model versioning strictly follows the MAJOR.MINOR.PATCH format.
