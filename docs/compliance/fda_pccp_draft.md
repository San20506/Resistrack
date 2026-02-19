# FDA Predetermined Change Control Plan (PCCP) — ResisTrack

**Document Version:** 1.0.0
**Last Updated:** 2026-02-18
**Regulatory Framework:** FDA Guidance on PCCP for AI/ML-Based Software as a Medical Device (SaMD)

## 1. Device Description

**Product Name:** ResisTrack AMR Risk Prediction System
**Intended Use:** Clinical decision support for antimicrobial resistance (AMR) risk assessment. The system predicts the likelihood of antimicrobial resistance in hospitalized patients and provides risk-stratified alerts to clinical staff.

**SaMD Category:** Class II (Non-significant risk CDS)

**Important Limitation:** ResisTrack does NOT prescribe treatment. All outputs are advisory and require clinician review. The system includes a LOW_CONFIDENCE_FLAG (threshold: 0.60) to explicitly mark uncertain predictions.

## 2. Initial Algorithm Description

### 2.1 Model Architecture
- **XGBoost Ensemble:** Tabular feature processing (47 features)
- **PyTorch LSTM:** Temporal sequence modeling for lab trends
- **ClinicalBERT (Bio_ClinicalBERT):** Clinical note NLP analysis
- **Learned Ensemble:** Weighted combination of all three models

### 2.2 Training Data
- **Source:** MIMIC-IV clinical database (de-identified)
- **Split:** 70% training / 15% validation / 15% test (stratified)
- **Minimum Validation:** 1,000+ patient records
- **Class Imbalance:** SMOTE applied when positive class < 20%
- **Random State:** 42 (all operations)

### 2.3 Performance Specifications (Locked)
| Metric | Threshold |
|--------|-----------|
| AUC-ROC | ≥ 0.82 |
| AUPRC | ≥ 0.70 |
| Sensitivity | ≥ 0.80 at 80% Specificity |
| False Positive Rate | ≤ 0.20 |
| Brier Score | ≤ 0.15 |
| P95 Latency | ≤ 2,000 ms |

## 3. Predetermined Changes

### 3.1 Allowed Changes (No New 510(k) Required)

| Change Category | Description | Monitoring Trigger | Validation Requirement |
|----------------|-------------|-------------------|----------------------|
| Model Retraining | Retrain on new data from same distribution | PSI drift score > threshold | AUC-ROC ≥ 0.82 maintained on holdout set |
| Feature Weight Updates | Ensemble weight re-optimization | Monthly performance review | All metrics meet Section 2.3 thresholds |
| Threshold Adjustment | Risk tier boundary recalibration | Clinician feedback > 5% override rate | Subgroup fairness maintained |
| Hyperparameter Tuning | Bayesian optimization (50+ trials) | Scheduled quarterly | Cross-validated performance stable |

### 3.2 Changes Requiring Submission

| Change Type | Trigger |
|------------|---------|
| New feature categories | Adding non-tabular/NLP/temporal features |
| Architecture change | Replacing XGBoost/LSTM/ClinicalBERT |
| Intended use expansion | New clinical populations or settings |
| Risk tier restructuring | Changing the 4-tier risk classification |

## 4. Performance Monitoring Plan

### 4.1 Continuous Monitoring
- **Population Stability Index (PSI):** Monitored per feature, alert on drift
- **Model Performance:** AUC-ROC, AUPRC tracked weekly on production data
- **Override Rate:** Clinical override tracking per risk tier
- **Subgroup Analysis:** Performance disaggregated by demographics

### 4.2 Retraining Protocol
- **Trigger:** PSI drift detected OR quarterly schedule
- **Method:** Blue/green deployment with shadow scoring
- **Validation:** Champion/challenger comparison on holdout data
- **Rollback:** Automated if new model degrades below thresholds

## 5. Real-World Performance Reporting

Annual performance reports will include:
- Model accuracy metrics vs. locked thresholds
- Subgroup performance analysis
- Override rate analysis by role and risk tier
- Adverse event correlation analysis
- Data drift summary and retraining log
