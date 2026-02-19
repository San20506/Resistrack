# Model Card — ResisTrack AMR Risk Prediction

**Version:** 1.0.0
**Last Updated:** 2026-02-18
**Framework:** Based on Mitchell et al., "Model Cards for Model Reporting" (2019)

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | ResisTrack AMR Risk Predictor |
| **Version** | 1.0.0 |
| **Type** | Ensemble (XGBoost + LSTM + ClinicalBERT) |
| **Architecture** | Learned-weight ensemble with three specialist models |
| **Training Framework** | XGBoost 2.x, PyTorch 2.x, HuggingFace Transformers |
| **License** | Proprietary — Team Curelytics |

## Intended Use

**Primary Use:** Clinical decision support for predicting antimicrobial resistance risk in hospitalized patients.

**Primary Users:** Physicians, pharmacists, infection control specialists, nurses.

**Out-of-Scope Uses:**
- This model does NOT prescribe treatment or antibiotics
- Not validated for outpatient/community settings
- Not intended for pediatric populations (unless validated)
- Not a replacement for culture and sensitivity testing

## Training Data

| Attribute | Value |
|-----------|-------|
| **Source** | MIMIC-IV (de-identified ICU data) |
| **Size** | ≥ 1,000 patient records (validation set) |
| **Split** | 70/15/15 (train/val/test), stratified |
| **Features** | 47 tabular features across 5 groups |
| **Class Balance** | SMOTE when positive class < 20% |
| **Random State** | 42 |

### Feature Groups (47 total)
1. **Lab Trends (12):** WBC, CRP, procalcitonin, lactate, etc.
2. **Medication History (8):** Prior antibiotic use, duration, classes
3. **Clinical Context (10):** Comorbidities, procedures, cultures
4. **Hospitalization (8):** LOS, ICU days, transfers, devices
5. **Vital Signs (9):** Temperature, HR, BP, RR, SpO2 trends

## Performance Metrics

| Metric | Threshold | Achieved |
|--------|-----------|----------|
| AUC-ROC | ≥ 0.82 | TBD |
| AUPRC | ≥ 0.70 | TBD |
| Sensitivity @ 80% Specificity | ≥ 0.80 | TBD |
| False Positive Rate | ≤ 0.20 | TBD |
| Brier Score | ≤ 0.15 | TBD |
| P95 Inference Latency | ≤ 2,000 ms | TBD |

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `patient_token` | string | De-identified patient identifier |
| `amr_risk_score` | int (0-100) | Composite AMR risk score |
| `risk_tier` | enum | LOW (0-24), MEDIUM (25-49), HIGH (50-74), CRITICAL (75-100) |
| `confidence_score` | float (0-1) | Model confidence in prediction |
| `low_confidence_flag` | bool | True when confidence < 0.60 |
| `data_completeness_score` | float (0-1) | Proportion of available input features |
| `data_quality_flag` | bool | True when data quality is acceptable |
| `antibiotic_class_risk` | object | Per-class risk for 5 antibiotic classes |
| `shap_top_features` | array | Top-5 SHAP feature explanations |
| `recommended_action` | string | Advisory action (not prescriptive) |
| `model_version` | string | Deployed model version |

## Ethical Considerations

- **Bias Monitoring:** Subgroup disaggregation by age, sex, race/ethnicity
- **Fairness Constraint:** Performance must not degrade > 5% for any protected subgroup
- **Transparency:** SHAP explanations provided for every prediction
- **Human Oversight:** All predictions require clinician review; model never prescribes
- **Uncertainty Communication:** LOW_CONFIDENCE_FLAG explicitly marks uncertain predictions

## Limitations

- Trained on MIMIC-IV data (single US academic medical center)
- May not generalize to non-ICU settings without revalidation
- Temporal patterns may shift with evolving resistance patterns
- ClinicalBERT performance depends on clinical note quality and completeness
- Real-time performance subject to EHR data availability and latency

## Monitoring and Updates

- Continuous PSI drift monitoring per feature
- Quarterly retraining with blue/green deployment
- Annual model card update with real-world performance data
- Override rate tracking per risk tier and user role
