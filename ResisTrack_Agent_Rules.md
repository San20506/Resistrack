# ResisTrack — Agent Rules
## Project-Specific Guidelines for AI Model Building

**Project:** ResisTrack · AI-Powered AMR Risk Prediction & Infection Control Platform  
**Team:** Curelytics · Impact-AI-Thon 2026  
**Version:** 1.0

---

## 1. Project Context & Mission

ResisTrack predicts antimicrobial resistance (AMR) risk in hospitalized patients within 6 hours of admission, before culture results are available (which take 48–120 hours). The platform ingests EHR data, lab results, vital signs, prior antibiotic history, and clinical notes to generate real-time AMR risk scores and stewardship recommendations.

> **Primary Goal:** Close the diagnostic gap. Reduce inappropriate antibiotic prescribing. Support — never replace — clinical judgment.

---

## 2. Core Ethical & Safety Rules

These rules are **non-negotiable** and override all other instructions:

```
RULE-SAFETY-01: The model MUST NEVER issue antibiotic prescriptions, 
modify medication orders, or make autonomous treatment decisions.
All outputs are decision support — final authority rests with the clinician.

RULE-SAFETY-02: The model MUST NOT surface predictions with a 
calibrated confidence score below 0.60 without attaching a 
LOW_CONFIDENCE_FLAG = true to the output payload.

RULE-SAFETY-03: The model MUST flag when input data quality is 
insufficient (e.g., < 3 lab values in prior 72 hours, missing vitals) 
and must communicate data completeness score alongside the risk output.

RULE-SAFETY-04: Model outputs must NEVER be used to deny treatment.
They are risk stratification tools only.

RULE-SAFETY-05: No model version may be promoted to production without 
passing clinical validation on >= 1,000 patient records with documented 
sensitivity >= 0.80 and specificity >= 0.75.
```

---

## 3. Data Handling Rules

```
RULE-DATA-01: ALL patient data must be treated as Protected Health 
Information (PHI) under HIPAA. No PHI may leave the AWS VPC boundary.

RULE-DATA-02: The model must NEVER receive raw patient identifiers 
(name, SSN, DOB, MRN) as direct input features. All patient references 
must use tokenized internal IDs only.

RULE-DATA-03: Clinical notes passed to ClinicalBERT/BioBERT must be 
processed inside the VPC only. No external API calls (e.g., OpenAI, 
Anthropic) with patient note content.

RULE-DATA-04: Model training data must originate only from hospitals 
that have signed a Business Associate Agreement (BAA) and data sharing 
consent. Training on non-consented data is prohibited.

RULE-DATA-05: All model inputs and outputs must be logged to the 
audit trail in RDS with timestamp, user role, hospital_tenant_id, 
and de-identified patient_token. Log retention: 7 years minimum.

RULE-DATA-06: Training datasets must be de-identified per HIPAA 
Safe Harbor (removing all 18 PHI identifiers) before use in 
any non-production environment.
```

---

## 4. Input Feature Specification

### 4.1 Accepted Structured Features (XGBoost / Tabular Model)

| Feature Name | Type | Source | Notes |
|---|---|---|---|
| `wbc_trend_7d` | float | LIS | White blood cell count — 7-day slope |
| `crp_latest` | float | LIS | C-Reactive Protein, most recent value |
| `creatinine_trend` | float | LIS | Creatinine 72h delta |
| `prior_beta_lactam_count` | int | Pharmacy | Count of prior beta-lactam Rx in past 90 days |
| `prior_fluoroquinolone_count` | int | Pharmacy | Count in past 90 days |
| `prior_carbapenem_flag` | bool | Pharmacy | Any carbapenem exposure in past 12 months |
| `icu_admission_flag` | bool | EHR | Is the current encounter ICU admission |
| `age_years` | int | EHR | Patient age — do NOT use DOB directly |
| `charlson_comorbidity_index` | int | Calculated | From ICD-10 codes in active problem list |
| `admission_ward_code` | categorical | EHR | Encoded ward ID (not ward name) |
| `days_since_last_hospitalization` | int | EHR | 0 if no prior admission in system |
| `culture_positive_history_flag` | bool | LIS | Any prior positive culture on record |
| `isolation_flag_current` | bool | EHR | Active contact/droplet isolation order |
| `temperature_max_48h` | float | Vitals | Max temp (°C) in past 48 hours |
| `heart_rate_max_48h` | float | Vitals | Max HR in past 48 hours |

> All feature values must be validated against acceptable ranges before inference. Values outside physiologically plausible ranges must trigger `DATA_QUALITY_FLAG`.

### 4.2 Temporal Features (PyTorch LSTM)

- Input shape: `(batch_size, 72, 13)` — 72 hourly timestamps, 13 channels (8 lab values + 5 vitals)
- Missing time steps: forward-fill with last known value; if >30% of timestamps are missing → set `DATA_COMPLETENESS_SCORE < 0.70` and attach warning
- Normalization: z-score per feature using hospital-cohort training statistics (not global statistics)

### 4.3 NLP Features (ClinicalBERT)

- Input: last 3 clinical notes (physician + nursing), max 512 tokens each after truncation
- Truncation strategy: keep first 128 tokens (contains chief complaint / assessment) + last 384 tokens
- Notes older than 72 hours: exclude unless no newer notes exist
- Do NOT pass radiology report image data — text reports only

---

## 5. Model Output Schema

Every inference call must return the following structured JSON payload:

```json
{
  "patient_token": "string (de-identified internal token)",
  "hospital_tenant_id": "string",
  "inference_timestamp": "ISO 8601 UTC",
  "amr_risk_score": 0.0,
  "risk_tier": "LOW | MEDIUM | HIGH | CRITICAL",
  "confidence_score": 0.0,
  "low_confidence_flag": false,
  "data_completeness_score": 0.0,
  "data_quality_flag": false,
  "antibiotic_class_risk": {
    "beta_lactam": 0.0,
    "carbapenem": 0.0,
    "fluoroquinolone": 0.0,
    "aminoglycoside": 0.0,
    "vancomycin": 0.0
  },
  "shap_top_features": [
    {
      "feature_name": "string",
      "shap_value": 0.0,
      "direction": "INCREASES_RISK | DECREASES_RISK",
      "human_readable": "string (plain English explanation for clinician)"
    }
  ],
  "recommended_action": "string (stewardship recommendation text)",
  "model_version": "string",
  "explanation_available": true
}
```

### Risk Tier Thresholds

| Score Range | Tier | Required Action |
|---|---|---|
| 0 – 24 | LOW | No immediate action required; monitor |
| 25 – 49 | MEDIUM | Flag for pharmacist review within 24h |
| 50 – 74 | HIGH | Trigger CDS Hook alert to attending physician and pharmacy |
| 75 – 100 | CRITICAL | Immediate CDS alert + infection control notification |

---

## 6. Model Training Rules

```
RULE-TRAIN-01: Train/validation/test split must be 70/15/15 with 
stratification on outcome label (resistant/sensitive) and hospital_tenant_id.
Do NOT train and test on data from the same hospital to avoid 
site-specific overfitting.

RULE-TRAIN-02: Class imbalance handling — apply SMOTE or class_weight 
balancing when positive (resistant) class prevalence < 20%.
Document imbalance ratio in the model card.

RULE-TRAIN-03: XGBoost hyperparameter search must use Bayesian 
optimization (not random search) with 50+ trials via SageMaker HPO.
Key parameters to tune: max_depth (3–8), learning_rate (0.01–0.3),
n_estimators (100–1000), subsample (0.6–1.0).

RULE-TRAIN-04: ClinicalBERT fine-tuning must use a clinical-domain 
pre-trained checkpoint (e.g., emilyalsentzer/Bio_ClinicalBERT).
Do NOT fine-tune general-domain BERT on clinical notes.

RULE-TRAIN-05: All training runs must be logged to SageMaker 
Experiments with: dataset version, feature set version, hyperparameters, 
AUC-ROC, AUPRC, sensitivity@80%specificity, and confusion matrix.

RULE-TRAIN-06: Ensemble weights (XGBoost vs LSTM vs NLP) must be 
learned via a held-out validation set meta-learner. Do NOT hardcode 
equal weights.

RULE-TRAIN-07: Model performance must be disaggregated by 
subgroup: age band (< 18, 18–65, > 65), ICU vs non-ICU, 
and primary organism if label is available.
Report any subgroup performance gaps >= 10% AUC as a risk item.
```

---

## 7. Evaluation Metrics & Acceptance Criteria

The following metrics must ALL be met before a model can be promoted to production:

| Metric | Minimum Threshold | Primary Model (XGBoost) |
|---|---|---|
| AUC-ROC | ≥ 0.82 | Primary evaluation metric |
| AUPRC | ≥ 0.70 | Required for imbalanced data fairness |
| Sensitivity @ 80% Specificity | ≥ 0.80 | Critical for patient safety — miss rate |
| False Positive Rate | ≤ 0.20 | Alert fatigue prevention |
| Calibration (Brier Score) | ≤ 0.15 | Probability reliability |
| Inference Latency (p95) | ≤ 2,000 ms | Real-time CDS requirement |

> **Mandatory:** All thresholds must be validated on a **held-out test set** (not validation set) before the model card is signed off.

---

## 8. CDS Hook Integration Rules

```
RULE-CDS-01: CDS Hook responses must be returned within 2 seconds 
(p95). If inference endpoint latency exceeds 1.5 seconds, return 
a cached score from the last inference run (max 24 hours old) 
and flag CACHED_RESULT = true in the response.

RULE-CDS-02: CDS Hook cards must include a "Why this alert?" 
link that opens the SHAP explainability panel — mandatory for 
High and Critical tier alerts.

RULE-CDS-03: Every CDS Hook alert must provide three response 
options to the clinician: 
  (a) "Acknowledged — will act" 
  (b) "Override — not applicable" (requires reason code selection)
  (c) "Escalate to ID specialist"
All responses must be logged.

RULE-CDS-04: Override rate per clinician must be monitored. 
If any clinician's override rate exceeds 60% over a 30-day period,
auto-generate a model feedback report for review by the clinical 
informatics team.
```

---

## 9. MLOps & Deployment Rules

```
RULE-MLOPS-01: Model retraining schedule — monthly automated 
SageMaker Pipeline run on new hospital data. Emergency retraining 
triggered if model drift score (PSI > 0.20) is detected in 
production monitoring.

RULE-MLOPS-02: Blue/green deployment required for all model 
updates. New model receives 10% traffic initially; auto-promote 
to 100% if AUC-ROC on production shadow traffic >= previous 
model - 0.02 over 72 hours.

RULE-MLOPS-03: Model versioning: semantic versioning (MAJOR.MINOR.PATCH).
MAJOR version bump required for changes to feature set.
MINOR for retrained weights on same feature set.
PATCH for calibration-only updates.

RULE-MLOPS-04: Model rollback capability must be maintained for 
the previous 2 production versions. Rollback execution time 
target: < 15 minutes.

RULE-MLOPS-05: All production model predictions must be stored 
(de-identified) for post-hoc analysis and ground truth 
comparison once culture results are available.
Model accuracy against culture ground truth must be reported monthly.
```

---

## 10. Prohibited Patterns

The following patterns are **strictly prohibited** in any code, model, or pipeline component:

```python
# ❌ NEVER DO: External API calls with patient data
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": patient_note}]  # PROHIBITED
)

# ❌ NEVER DO: Raw PHI as model features
features["patient_name"] = row["patient_name"]  # PROHIBITED
features["date_of_birth"] = row["dob"]           # PROHIBITED
features["social_security"] = row["ssn"]         # PROHIBITED

# ❌ NEVER DO: Log PHI to CloudWatch or stdout
print(f"Processing patient {patient_mrn}")        # PROHIBITED
logger.info(f"Patient name: {patient_name}")       # PROHIBITED

# ❌ NEVER DO: Hardcode thresholds for clinical decisions
if amr_score > 50:
    prescribe_vancomycin()  # PROHIBITED — model never prescribes

# ❌ NEVER DO: Deploy model without validation gate
model.deploy(validation_passed=False)             # PROHIBITED
```

---

## 11. Code Quality Standards

- All Python code must pass `mypy --strict` type checking
- All ML pipelines must be reproducible: set `random_state=42` or equivalent for all stochastic operations
- Feature engineering functions must have unit tests with ≥80% line coverage
- SageMaker Processing scripts must be containerized (Docker) and version-pinned for reproducibility
- Secrets (API keys, DB credentials) must NEVER appear in code — use AWS Secrets Manager exclusively
- All infrastructure must be provisioned via AWS CDK (TypeScript) — no console-created resources in production

---

## 12. Glossary

| Term | Definition |
|---|---|
| AMR | Antimicrobial Resistance — resistance of microorganisms to antimicrobial medicines |
| MDRO | Multi-Drug Resistant Organism |
| SHAP | SHapley Additive exPlanations — model explainability method |
| CDS Hooks | Clinical Decision Support Hooks — standard for EHR-integrated alerts |
| SMART on FHIR | Substitutable Medical Applications, Reusable Technologies on FHIR |
| HL7 v2 | Health Level 7 version 2 — legacy healthcare messaging standard |
| FHIR R4 | Fast Healthcare Interoperability Resources Release 4 — modern healthcare data standard |
| BAA | Business Associate Agreement — HIPAA-required contract for PHI handling |
| PHI | Protected Health Information |
| PSI | Population Stability Index — metric for detecting model/data drift |
| AUPRC | Area Under the Precision-Recall Curve |

---

*ResisTrack Agent Rules v1.0 — Team Curelytics — Impact-AI-Thon 2026*  
*These rules must be reviewed and updated with each MAJOR model version release.*
