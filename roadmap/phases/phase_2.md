# Phase 2: AI/ML Inference Engine

## Objective
Develop a three-model ensemble on SageMaker to produce a calibrated AMR Risk Score (0-100) with SHAP explainability.

## Duration
2 Weeks

## Modules

### M2.1 Feature Engineering Pipeline
*   **Duration:** 3 days
*   **Scope:** Define 47 tabular features, implement data quality flags, and setup SageMaker Feature Store.
*   **Dependencies:** Independent (utilizes MIMIC-IV synthetic data).

### M2.2 Temporal Feature Extraction
*   **Duration:** 2 days
*   **Scope:** Process 72-hour rolling windows into LSTM tensors (batch, 72, 13).
*   **Dependencies:** M2.1.

### M2.3 NLP Feature Extraction / ClinicalBERT
*   **Duration:** 2 days
*   **Scope:** Fine-tune Bio_ClinicalBERT to generate a 32-dimensional risk vector from clinical notes.
*   **Dependencies:** M2.1.

### M2.4 XGBoost Tabular Risk Model
*   **Duration:** 2 days
*   **Scope:** Bayesian Hyperparameter Optimization (HPO), SMOTE balancing, and SageMaker Experiments logging.
*   **Dependencies:** M2.1.

### M2.5 PyTorch LSTM Temporal Model
*   **Duration:** 2 days
*   **Scope:** Time-series training to produce a trend-risk vector.
*   **Dependencies:** M2.2.

### M2.6 Ensemble, Calibration and SHAP
*   **Duration:** 2 days
*   **Scope:** Meta-learner weighting, Platt scaling for calibration, and TreeExplainer for top-5 feature extraction.
*   **Dependencies:** M2.3, M2.4, M2.5.

### M2.7 SageMaker Inference Endpoint
*   **Duration:** 1 day
*   **Scope:** Deploy real-time endpoint with auto-scaling and a 24-hour response cache.
*   **Dependencies:** M2.6.

## Model Output Schema
The endpoint returns a JSON payload with the following structure:
```json
{
  "patient_token": "string",
  "risk_score": 0-100,
  "risk_tier": "LOW | MEDIUM | HIGH | CRITICAL",
  "confidence": 0.0-1.0,
  "low_confidence_flag": boolean,
  "shap_top_features": [
    {"feature": "string", "impact": float}
  ],
  "antibiotic_class_risk": [
    {"class": "string", "risk_index": float}
  ]
}
```

## Risk Tier Thresholds
*   **LOW:** 0 - 24
*   **MEDIUM:** 25 - 49
*   **HIGH:** 50 - 74
*   **CRITICAL:** 75 - 100

## Success Criteria
*   Inference endpoint returns valid JSON within 2000ms.
*   Performance metrics: AUC-ROC >= 0.82, AUPRC >= 0.70.
*   Sensitivity >= 0.80 at 80% Specificity.
*   SHAP top-5 features are present in the output.
*   LOW_CONFIDENCE_FLAG is correctly triggered for confidence scores < 0.60.
