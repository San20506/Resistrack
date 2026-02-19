# M2.4 -- XGBoost Tabular Risk Model
**Phase:** 2 -- ML Development and Ensemble
**Duration:** ~2 days
**Status:** Not Started

## Objective
Train a high-performance XGBoost model on structured clinical features to predict AMR risk.

## Scope
- Train XGBoost on 47 structured features from M2.1.
- Implement SMOTE or class-weight balancing if the resistant class is <20%.
- Perform Bayesian Hyperparameter Optimization (HPO) via SageMaker (50+ trials).
- Hyperparameters to tune: `max_depth` (3-8), `learning_rate` (0.01-0.3), `n_estimators` (100-1000), `subsample` (0.6-1.0).
- Use a 70/15/15 stratified split by outcome and `hospital_tenant_id`.
- Log all training runs and metrics to SageMaker Experiments.

## Dependencies
- **Depends on:** M2.1
- **Depended on by:** M2.6

## Inputs
- 47 tabular features from SageMaker Feature Store.

## Outputs
- Trained XGBoost model artifact.
- Experiment logs including AUC-ROC, AUPRC, sensitivity, and confusion matrix.

## Implementation Notes
- Ensure no hospital leakage by stratifying the split by `hospital_tenant_id`.
- Use `random_state=42` for all operations to ensure reproducibility.
- SMOTE should only be applied to the training set, not the validation or test sets.

## Agent Rules
- RULE-TRAIN-01: Stratified split; no same-hospital train/test leakage.
- RULE-TRAIN-02: Use SMOTE or class-weighting for imbalanced data.
- RULE-TRAIN-03: Bayesian HPO with 50+ trials.
- RULE-TRAIN-05: Log all runs to SageMaker Experiments.
- RULE-TRAIN-07: Perform subgroup disaggregation analysis.

## Done When
- [ ] XGBoost is trained with Bayesian HPO (50+ trials).
- [ ] SMOTE or class-weighting is applied if class imbalance is present.
- [ ] 70/15/15 stratified split is implemented without hospital leakage.
- [ ] All runs are logged to SageMaker Experiments with full metrics.
- [ ] `random_state=42` is used throughout.
- [ ] Model artifact is saved and versioned.
