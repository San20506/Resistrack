# ResisTrack — Modular Development Roadmap
**Project:** ResisTrack · AI-Powered AMR Risk Prediction & Infection Control Platform
**Team:** Curelytics · Impact-AI-Thon 2026
**Version:** 1.0 · Date: February 18, 2026
**Context:** Starting from scratch. All documentation (PRD v1.0, Tech Arch v1.0, Agent Rules v1.0) is finalized. No code exists yet.

---

## Context Summary

| Field | Detail |
|---|---|
| **Goal** | Predict AMR risk within 6 hours of patient admission; reduce inappropriate antibiotic prescribing by ≥30% |
| **Stack** | AWS (SageMaker, HealthLake, RDS, Lambda, CDK) · Python 3.11 · XGBoost · PyTorch LSTM · ClinicalBERT · React.js · FHIR R4 · CDS Hooks |
| **Domain** | Healthcare AI / Clinical Decision Support |
| **Constraints** | Hackathon timeline (Impact-AI-Thon 2026); small team; HIPAA compliance non-negotiable; no PHI outside AWS VPC |
| **Current State** | Starting from scratch — documentation complete, no code written |

---

## Roadmap Overview

| Phase | Title | Duration | Focus |
|---|---|---|---|
| 1 | Secure Data Foundation | 2 weeks | Infrastructure, ingestion, storage |
| 2 | AI/ML Inference Engine | 2 weeks | Feature engineering, model training, ensemble |
| 3 | Clinical Integration & Alerts | 1.5 weeks | CDS Hooks, EHR integration, notifications |
| 4 | Dashboard, Reporting & UX | 1 week | React dashboard, reports, role-based views |
| 5 | Validation, MLOps & Hardening | 1 week | Clinical validation, CI/CD, monitoring, compliance |

**Total Estimated Duration:** ~7.5 weeks

---

## Phase 1 — Secure Data Foundation

**Objective:** Build the HIPAA-compliant AWS infrastructure and data ingestion pipeline that receives, transforms, validates, and stores patient data from hospital EHR/LIS/Pharmacy systems — before any ML work begins.

### Modules

#### M1.1 — AWS Infrastructure Baseline *(~3 days)*
- Provision private AWS VPC with subnets, Security Groups, NACLs
- Set up AWS CDK (TypeScript) project for all IaC — no console-created resources
- Configure AWS KMS with per-tenant customer-managed keys
- Set up AWS IAM roles with least-privilege RBAC policies
- Configure AWS Secrets Manager for all credentials
- **Independent:** Can be built and validated before any other module

#### M1.2 — Hospital Connectivity Layer *(~2 days)*
- Configure VPN / AWS Direct Connect tunnel from hospital DMZ to VPC
- Implement TLS 1.3 + mutual certificate authentication for all data feeds
- Set up API Gateway with JWT validation and per-tenant rate limiting
- **Depends on:** M1.1

#### M1.3 — HL7 v2 → FHIR R4 Transformation *(~3 days)*
- Deploy Mirth Connect integration engine
- Build transformation channels: HL7 ADT A01 → FHIR Patient/Encounter, HL7 ORU → FHIR Observation, HL7 RDE → FHIR MedicationRequest
- Validate FHIR R4 bundle schema conformance
- **Independent:** Can be built and unit-tested with synthetic HL7 messages

#### M1.4 — FHIR Ingestion & PHI Tokenization Lambda *(~2 days)*
- Build Lambda function for FHIR bundle validation, de-duplication, and PHI tokenization
- Replace patient identifiers (name, DOB, MRN) with internal `patient_token`
- Route validated bundles to AWS HealthLake
- **Depends on:** M1.1, M1.3

#### M1.5 — Data Storage Layer *(~2 days)*
- Configure AWS HealthLake (FHIR R4) as primary patient data store — AES-256 encrypted
- Provision Amazon RDS (PostgreSQL) for model outputs, alert records, audit logs
- Set up S3 buckets for raw clinical notes, model artifacts, reports
- Implement 7-year audit log retention policy
- **Depends on:** M1.1

#### M1.6 — Audit Logging & Compliance Baseline *(~1 day)*
- Enable AWS CloudTrail for all API activity
- Enable RDS audit logging for all data access events
- Verify HIPAA-eligible service configuration (HealthLake, RDS, S3, KMS)
- **Depends on:** M1.5

### ✅ Phase 1 Success Criterion
> A synthetic HL7 ADT A01 admission message flows end-to-end: hospital DMZ → Mirth Connect → FHIR R4 Bundle → API Gateway → Lambda (PHI tokenized) → HealthLake (stored, encrypted). Audit log entry confirmed in RDS. Zero PHI visible in CloudWatch logs.

---

## Phase 2 — AI/ML Inference Engine

**Objective:** Build the three-model ensemble (XGBoost + LSTM + ClinicalBERT) on AWS SageMaker, with a feature engineering pipeline that extracts structured, temporal, and NLP features from HealthLake data, producing a calibrated AMR Risk Score (0–100) with SHAP explainability.

### Modules

#### M2.1 — Feature Engineering Pipeline *(~3 days)*
- Build SageMaker Processing Job for structured feature extraction (47 tabular features per Agent Rules §4.1)
  - WBC trend (7d), CRP, creatinine delta, prior antibiotic counts, ICU flag, age, Charlson Comorbidity Index, admission ward, culture history, vitals
- Implement physiological range validation → `DATA_QUALITY_FLAG` on out-of-range values
- Implement forward-fill for missing time steps; set `DATA_COMPLETENESS_SCORE < 0.70` if >30% missing
- Register features in SageMaker Feature Store
- **Independent:** Can be built and tested with MIMIC-IV synthetic data

#### M2.2 — Temporal Feature Extraction (LSTM Input) *(~2 days)*
- Build 72-hour rolling window extractor: 8 lab values + 5 vital signs → tensor shape `(batch, 72, 13)`
- Implement z-score normalization per feature using hospital-cohort training statistics
- **Depends on:** M2.1

#### M2.3 — NLP Feature Extraction (ClinicalBERT) *(~2 days)*
- Load `emilyalsentzer/Bio_ClinicalBERT` checkpoint from HuggingFace (no external API calls with PHI)
- Implement note selection: last 3 clinical notes, max 512 tokens, truncation: first 128 + last 384 tokens
- Exclude notes older than 72 hours unless no newer notes exist
- Fine-tune classification head for AMR risk embedding (768-dim → 32-dim risk vector)
- **Depends on:** M2.1; **Independent of** M2.2

#### M2.4 — XGBoost Tabular Risk Model *(~2 days)*
- Train XGBoost on structured features from MIMIC-IV + partner hospital data
- Apply SMOTE / class_weight balancing if resistant class prevalence < 20%
- Bayesian HPO via SageMaker (50+ trials): `max_depth`, `learning_rate`, `n_estimators`, `subsample`
- Train/val/test split: 70/15/15, stratified by outcome + `hospital_tenant_id`
- Log all runs to SageMaker Experiments (AUC-ROC, AUPRC, sensitivity@80%spec, confusion matrix)
- **Depends on:** M2.1

#### M2.5 — PyTorch LSTM Temporal Model *(~2 days)*
- Build and train PyTorch LSTM on 72-hour time-series tensors
- Output: trend-risk vector fed as additional features into XGBoost ensemble
- Reproducibility: `random_state=42` for all stochastic operations
- **Depends on:** M2.2

#### M2.6 — Ensemble, Calibration & SHAP Explainer *(~2 days)*
- Learn ensemble weights (XGBoost + LSTM + ClinicalBERT) via held-out validation meta-learner — no hardcoded weights
- Apply Platt scaling for probability calibration (Brier Score target ≤ 0.15)
- Apply TreeExplainer (SHAP) to final XGBoost predictions → top-5 contributing features per patient
- Map score to Risk Tier: LOW (0–24), MEDIUM (25–49), HIGH (50–74), CRITICAL (75–100)
- Attach `LOW_CONFIDENCE_FLAG = true` if confidence score < 0.60
- **Depends on:** M2.4, M2.5, M2.3

#### M2.7 — SageMaker Inference Endpoint *(~1 day)*
- Deploy ensemble as SageMaker real-time endpoint
- Configure auto-scaling: 1–20 instances based on invocation rate and latency
- Validate p95 inference latency ≤ 2,000 ms
- Implement response caching (24-hour TTL) for graceful degradation
- **Depends on:** M2.6

### ✅ Phase 2 Success Criterion
> Given a synthetic patient record (structured features + 72h vitals/labs + 2 clinical notes), the SageMaker endpoint returns a valid JSON payload (per Agent Rules §5 schema) within 2,000ms. AUC-ROC ≥ 0.82, AUPRC ≥ 0.70, Sensitivity@80%Spec ≥ 0.80 on held-out test set. SHAP top-5 features present in response. `LOW_CONFIDENCE_FLAG` correctly set for scores < 0.60.

---

## Phase 3 — Clinical Integration & Alerts

**Objective:** Surface AMR risk scores directly into the EHR clinician workflow via CDS Hooks, implement multi-role notification dispatch, and enforce SMART on FHIR authorization — without requiring clinicians to leave the EHR.

### Modules

#### M3.1 — SMART on FHIR Authorization *(~2 days)*
- Implement OAuth 2.0 / OpenID Connect authorization server
- Configure SMART on FHIR app launch for Epic and Cerner
- Implement JWT token validation on every CDS Hook request
- Set up IAM RBAC: Physician, ID Specialist, Pharmacist, Infection Control, Admin, IT Admin roles
- **Independent:** Can be built and tested with a SMART sandbox

#### M3.2 — CDS Hooks Service *(~3 days)*
- Register CDS Hooks with Epic/Cerner for: `patient-view` (admission), `order-sign` (antibiotic orders), `encounter-discharge`
- On trigger: query SageMaker endpoint synchronously; enforce 2-second response SLA
- If latency > 1,500ms: return cached score with `CACHED_RESULT = true`
- Build CDS Card response: Risk Tier, top-3 SHAP factors (plain English), stewardship recommendation, "Why this alert?" link
- Implement three clinician response options: Acknowledged / Override (reason code required) / Escalate to ID Specialist
- Log all clinician responses to RDS audit trail
- **Depends on:** M2.7, M3.1

#### M3.3 — Override Rate Monitoring *(~1 day)*
- Track per-clinician override rate over rolling 30-day window
- Auto-generate model feedback report if override rate > 60% for any clinician
- **Depends on:** M3.2

#### M3.4 — Multi-Role Notification Dispatch *(~2 days)*
- Implement AWS SNS notification routing by Risk Tier:
  - HIGH/CRITICAL → Pharmacy team alert (stewardship review)
  - CRITICAL → Infection Control Officer alert (MDRO flag)
- Ensure all notification payloads contain only de-identified `patient_token` — no PHI
- **Depends on:** M3.2

### ✅ Phase 3 Success Criterion
> A simulated patient admission in an Epic SMART sandbox triggers a CDS Hook. A CDS Card appears in the EHR UI within 2 seconds showing Risk Tier, top-3 SHAP factors, and stewardship recommendation. Clinician override is logged to RDS. A CRITICAL-tier patient triggers SNS notifications to Pharmacy and Infection Control with no PHI in the payload.

---

## Phase 4 — Dashboard, Reporting & UX

**Objective:** Build the React.js web dashboard for authorized clinical users, providing ward-level AMR heatmaps, patient risk timelines, infection trend analytics, and automated stewardship report generation — with strict role-based views.

### Modules

#### M4.1 — React Dashboard Shell & Auth *(~2 days)*
- Scaffold React.js app served via AWS CloudFront
- Implement SMART launch and direct OAuth 2.0 login
- Build role-based routing: each user type sees only their authorized scope
- **Independent:** Can be built with mock data before backend is connected

#### M4.2 — Ward-Level AMR Risk Heatmap *(~2 days)*
- Build interactive ward/unit heatmap showing real-time AMR risk distribution
- Color-coded by Risk Tier (LOW → CRITICAL)
- Drill-down to individual patient risk timeline
- **Depends on:** M4.1

#### M4.3 — Patient Risk Timeline & SHAP Panel *(~1 day)*
- Display per-patient AMR risk score history over admission
- SHAP explainability panel: top-5 contributing factors with direction (INCREASES/DECREASES risk) and plain English explanation
- Mandatory for HIGH and CRITICAL tier alerts (per Agent Rules §8)
- **Depends on:** M4.1

#### M4.4 — Pharmacy & Infection Control Views *(~1 day)*
- Pharmacy view: High/Critical patient list with recommended de-escalation options
- Infection Control view: hospital-wide outbreak trend analytics, MDRO cluster alerts
- **Depends on:** M4.1

#### M4.5 — Automated Stewardship Reports *(~1 day)*
- Build Python (ReportLab) report generator: weekly PDF/CSV stewardship summary
- Auto-schedule via SageMaker Pipelines or Lambda cron
- Store generated reports in S3; deliver to Infection Control Officers
- **Depends on:** M4.1

### ✅ Phase 4 Success Criterion
> An Infection Control Officer logs in via SMART launch and views the ward heatmap, drills into a CRITICAL patient's SHAP panel, and downloads a weekly PDF stewardship report. A Pharmacist sees only their authorized High/Critical patient list. A Hospital Admin sees only financial impact metrics — no PHI.

---

## Phase 5 — Validation, MLOps & Hardening

**Objective:** Complete clinical validation against the acceptance criteria, establish the automated MLOps retraining pipeline, implement production monitoring, and finalize all compliance documentation before any production deployment.

### Modules

#### M5.1 — Clinical Validation Gate *(~2 days)*
- Validate model on ≥ 1,000 held-out patient records from partner hospital
- Confirm all acceptance thresholds (per Agent Rules §7):
  - AUC-ROC ≥ 0.82, AUPRC ≥ 0.70, Sensitivity@80%Spec ≥ 0.80, FPR ≤ 0.20, Brier Score ≤ 0.15
- Disaggregate performance by subgroup: age band (<18, 18–65, >65), ICU vs non-ICU
- Flag any subgroup AUC gap ≥ 10% as a risk item in the model card
- Sign off model card — no deployment without this gate
- **Depends on:** All Phase 2 modules

#### M5.2 — SageMaker MLOps Retraining Pipeline *(~2 days)*
- Build SageMaker Pipeline for automated monthly retraining on new hospital data
- Implement Population Stability Index (PSI) monitoring — trigger emergency retraining if PSI > 0.20
- Configure blue/green deployment: new model receives 10% traffic; auto-promote if AUC ≥ previous − 0.02 over 72 hours
- Maintain rollback capability for previous 2 production versions; rollback target < 15 minutes
- **Depends on:** M5.1

#### M5.3 — CI/CD Pipeline *(~1 day)*
- Build AWS CodePipeline + CodeBuild pipeline: automated build, test, and deployment
- Enforce `mypy --strict` type checking on all Python code
- Enforce ≥ 80% unit test coverage on all feature engineering functions
- Gate: no model deployment without passing automated regression tests on held-out validation set
- **Depends on:** M5.2

#### M5.4 — Production Monitoring & Alerting *(~1 day)*
- Configure Amazon CloudWatch dashboards: inference latency (p95), error rates, SageMaker endpoint health
- Set up CloudWatch alarms → SNS for: latency > 1,500ms, error rate > 1%, PSI drift alert
- Verify all monitoring uses de-identified tokens — no PHI in CloudWatch logs
- **Depends on:** M5.2

#### M5.5 — Compliance Documentation & Model Card *(~1 day)*
- Complete HIPAA Security Rule compliance checklist
- Draft FDA PCCP (Predetermined Change Control Plan) documentation for ML model updates
- Finalize model card: dataset version, feature set, hyperparameters, all evaluation metrics, subgroup analysis, known limitations
- Document BAA status for all hospital data partners
- **Independent:** Can be drafted in parallel with M5.1–M5.4

### ✅ Phase 5 Success Criterion
> Model card is signed off with all 6 acceptance metrics met on ≥ 1,000 patient held-out test set. CI/CD pipeline deploys a new model version via blue/green with 10% canary traffic. CloudWatch dashboard shows p95 latency ≤ 2,000ms. A simulated PSI drift event triggers the emergency retraining pipeline. Rollback to previous version completes in < 15 minutes.

---

## Module Dependency Map

```
Phase 1 (Infrastructure)
  M1.1 → M1.2, M1.4, M1.5
  M1.3 → M1.4
  M1.5 → M1.6

Phase 2 (ML Engine)
  M2.1 → M2.2, M2.3, M2.4
  M2.2 → M2.5
  M2.4 + M2.5 + M2.3 → M2.6
  M2.6 → M2.7

Phase 3 (CDS Integration)
  M2.7 + M3.1 → M3.2
  M3.2 → M3.3, M3.4

Phase 4 (Dashboard)
  M4.1 → M4.2, M4.3, M4.4, M4.5

Phase 5 (Validation & MLOps)
  Phase 2 → M5.1
  M5.1 → M5.2 → M5.3, M5.4
  M5.5 (parallel)
```

---

## Non-Negotiable Constraints (All Phases)

| Rule | Constraint |
|---|---|
| **RULE-SAFETY-01** | Model NEVER issues prescriptions or modifies orders — decision support only |
| **RULE-SAFETY-02** | `LOW_CONFIDENCE_FLAG = true` on all outputs with confidence < 0.60 |
| **RULE-SAFETY-05** | No production deployment without clinical validation on ≥ 1,000 records |
| **RULE-DATA-01** | All PHI stays within AWS VPC — no external API calls with patient data |
| **RULE-DATA-02** | No raw patient identifiers as model features — tokenized IDs only |
| **RULE-DATA-03** | ClinicalBERT runs inside VPC only — no OpenAI/Anthropic API calls |
| **RULE-MLOPS-02** | Blue/green deployment required for all model updates |
| **RULE-MLOPS-04** | Rollback capability for previous 2 production versions |

---

*ResisTrack Modular Development Roadmap v1.0 — Team Curelytics — Impact-AI-Thon 2026*
*Generated: February 18, 2026*
