# M2.6 -- Ensemble, Calibration and SHAP Explainer
**Phase:** 2 -- ML Development and Ensemble
**Duration:** ~2 days
**Status:** Not Started

## Objective
Combine multiple model outputs into a calibrated ensemble prediction with human-readable explanations.

## Scope
- Learn ensemble weights for XGBoost, LSTM, and ClinicalBERT using a held-out validation meta-learner.
- Implement Platt scaling for probability calibration (target Brier Score <= 0.15).
- Use `TreeExplainer` (SHAP) to identify the top-5 contributing features per patient.
- Map calibrated scores to Risk Tiers:
    - LOW: 0-24
    - MEDIUM: 25-49
    - HIGH: 50-74
    - CRITICAL: 75-100
- Attach a `LOW_CONFIDENCE_FLAG` if the model confidence is < 0.60.

## Dependencies
- **Depends on:** M2.3, M2.4, M2.5
- **Depended on by:** M2.7

## Inputs
- XGBoost predictions.
- LSTM trend-risk vectors.
- ClinicalBERT 32-dimensional risk vectors.

## Outputs
- Calibrated AMR Risk Score (0-100).
- Risk Tier classification.
- SHAP top-5 features with explanations.
- Full JSON output payload (per Agent Rules section 5).

## Implementation Notes
- Do not use hardcoded weights for the ensemble; weights must be learned by the meta-learner.
- SHAP explanations must be mapped to human-readable clinical terms.
- The output JSON must strictly adhere to the schema defined in the Agent Rules.

## Agent Rules
- RULE-SAFETY-02: Implement `LOW_CONFIDENCE_FLAG`.
- RULE-TRAIN-06: Use learned ensemble weights.
- Agent Rules section 5: Output schema requirements.
- Agent Rules section 7: Acceptance metrics (Brier Score).

## Done When
- [ ] Meta-learner successfully combines all three model outputs.
- [ ] Platt scaling calibrates probabilities to a Brier Score <= 0.15.
- [ ] SHAP top-5 features are generated with human-readable explanations.
- [ ] Risk tier mapping is correctly implemented.
- [ ] `LOW_CONFIDENCE_FLAG` is set when confidence is < 0.60.
- [ ] Output matches the Agent Rules section 5 JSON schema exactly.
