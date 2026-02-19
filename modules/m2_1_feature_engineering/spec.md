# M2.1 -- Feature Engineering Pipeline
**Phase:** 2 -- ML Development and Ensemble
**Duration:** ~3 days
**Status:** Not Started

## Objective
Extract and validate 47 structured features from FHIR data to create a tabular dataset for ML model training and inference.

## Scope
- SageMaker Processing Job for structured feature extraction.
- Implementation of 47 tabular features including:
    - Lab trends (wbc_trend_7d, crp_latest, creatinine_trend).
    - Medication history (prior_beta_lactam_count, prior_fluoroquinolone_count, prior_carbapenem_flag).
    - Clinical context (icu_admission_flag, age_years, charlson_comorbidity_index, admission_ward_code).
    - Hospitalization history (days_since_last_hospitalization, culture_positive_history_flag, isolation_flag_current).
    - Vitals (temperature_max_48h, heart_rate_max_48h).
- Physiological range validation and `DATA_QUALITY_FLAG` triggering.
- Forward-fill for missing time steps and `DATA_COMPLETENESS_SCORE` calculation.
- Feature registration in SageMaker Feature Store.

## Dependencies
- **Depends on:** None (can be built/tested with MIMIC-IV synthetic data)
- **Depended on by:** M2.2, M2.3, M2.4

## Inputs
- FHIR resources from HealthLake (Patient, Encounter, Observation, MedicationRequest).

## Outputs
- 47-feature tabular dataset in SageMaker Feature Store.
- Data quality metadata (flags and completeness scores).

## Implementation Notes
- Use `mypy` strict for type checking.
- Ensure `random_state=42` for reproducibility.
- Unit tests must achieve >=80% coverage.
- `DATA_COMPLETENESS_SCORE < 0.70` if >30% of features are missing.

## Agent Rules
- RULE-DATA-02: Use tokenized IDs only.
- RULE-SAFETY-03: Implement data quality flagging for out-of-range values.
- RULE-TRAIN-01: Use stratified splits for training/validation.
- Code Quality: mypy strict, random_state=42, unit tests >=80% coverage.

## Done When
- [ ] All 47 features are extracted and validated.
- [ ] Physiological range validation is implemented.
- [ ] `DATA_QUALITY_FLAG` triggers on out-of-range values.
- [ ] `DATA_COMPLETENESS_SCORE` is calculated correctly.
- [ ] Features are registered in SageMaker Feature Store.
- [ ] Unit tests pass with >=80% coverage.
- [ ] `mypy` strict passes.
