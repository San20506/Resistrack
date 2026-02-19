# M5.1 -- Clinical Validation Gate
**Phase:** 5 -- MLOps and Governance
**Duration:** ~2 days
**Status:** Not Started

## Objective
Perform rigorous clinical validation of the AMR risk model on a large, held-out dataset to ensure it meets all safety and performance thresholds.

## Scope
This module involves validating the model on at least 1000 held-out patient records from a partner hospital. It confirms that the model meets all six acceptance metrics: AUC-ROC >= 0.82, AUPRC >= 0.70, Sensitivity@80%Spec >= 0.80, FPR <= 0.20, Brier Score <= 0.15, and p95 latency <= 2000ms. The validation includes disaggregated analysis by age band (<18, 18-65, >65) and ICU status. Any subgroup AUC gap >= 10% must be flagged as a risk item.

## Dependencies
- **Depends on:** All Phase 2 modules
- **Depended on by:** M5.2

## Inputs
- Held-out test dataset (>= 1000 records)
- Trained model artifacts
- Subgroup metadata (age, ICU status)

## Outputs
- Comprehensive validation report with all 6 metrics
- Subgroup analysis results
- Finalized model card with performance sign-off

## Implementation Notes
Validation must be performed on a completely independent test set that was not used during training or hyperparameter tuning. The disaggregated analysis is critical for identifying potential biases or performance gaps in specific patient populations.

## Agent Rules
- RULE-SAFETY-05: Validate on >=1000 records; ensure sensitivity >= 0.80 and specificity >= 0.75.
- RULE-TRAIN-07: Perform subgroup disaggregation and flag significant performance gaps.

## Done When
- [ ] All 6 acceptance metrics meet their respective thresholds on the held-out test set.
- [ ] Subgroup analysis is completed for age bands and ICU status.
- [ ] Any AUC gap >= 10% between subgroups is documented and reviewed.
- [ ] Model card is drafted with full validation results and limitations.
- [ ] Clinical validation sign-off gate is officially passed.
