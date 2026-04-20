# ResisTrack

AI-Powered AMR Risk Prediction and Infection Control Platform by Team Curelytics for Impact-AI-Thon 2026.

ResisTrack is a clinical decision support system designed to predict Antimicrobial Resistance (AMR) risk within 6 hours of patient admission. By integrating real-time EHR data with advanced machine learning models, the platform aims to reduce inappropriate antibiotic prescribing by at least 30%. It provides clinicians with actionable insights and ward-level visualizations to improve infection control and patient outcomes.

## Tech Stack

*   **Cloud Infrastructure:** AWS (SageMaker, HealthLake, RDS, Lambda, CDK)
*   **Backend & ML:** Python 3.11, XGBoost, PyTorch LSTM, ClinicalBERT
*   **Frontend:** React.js
*   **Interoperability:** FHIR R4, CDS Hooks

## Directory Structure

```
project-root/
|-- roadmap/
|   |-- 00_context.md          # Project goal, stack, constraints
|   |-- 01_roadmap.md          # Full phased roadmap
|   |-- phases/
|       |-- phase_1.md         # Secure Data Foundation
|       |-- phase_2.md         # AI/ML Inference Engine
|       |-- phase_3.md         # Clinical Integration & Alerts
|       |-- phase_4.md         # Dashboard, Reporting & UX
|       |-- phase_5.md         # Validation, MLOps & Hardening
|-- modules/
|   |-- m1_1_aws_infra_baseline/    # VPC, CDK, KMS, IAM
|   |-- m1_2_hospital_connectivity/ # VPN, TLS, API Gateway
|   |-- m1_3_hl7_fhir_transformer/  # Mirth Connect, HL7->FHIR
|   |-- m1_4_fhir_ingestion_tokenization/ # Lambda, PHI tokenization
|   |-- m1_5_data_storage_layer/    # HealthLake, RDS, S3
|   |-- m1_6_audit_logging/         # CloudTrail, compliance
|   |-- m2_1_feature_engineering/   # 47 tabular features
|   |-- m2_2_temporal_features/     # LSTM tensors
|   |-- m2_3_nlp_clinicalbert/      # ClinicalBERT fine-tuning
|   |-- m2_4_xgboost_model/        # XGBoost training
|   |-- m2_5_lstm_model/           # PyTorch LSTM
|   |-- m2_6_ensemble_calibration/ # Ensemble + SHAP
|   |-- m2_7_sagemaker_endpoint/   # Real-time inference
|   |-- m3_1_smart_fhir_auth/      # OAuth, SMART launch
|   |-- m3_2_cds_hooks_service/    # CDS Cards, EHR integration
|   |-- m3_3_override_monitoring/  # Override rate tracking
|   |-- m3_4_notification_dispatch/ # SNS notifications
|   |-- m4_1_dashboard_shell/      # React app, auth
|   |-- m4_2_ward_heatmap/         # Ward risk visualization
|   |-- m4_3_patient_timeline/     # Patient risk + SHAP panel
|   |-- m4_4_pharmacy_ic_views/    # Role-specific views
|   |-- m4_5_stewardship_reports/  # PDF/CSV reports
|   |-- m5_1_clinical_validation/  # Model validation gate
|   |-- m5_2_mlops_retraining/     # Auto retraining pipeline
|   |-- m5_3_cicd_pipeline/        # CodePipeline + CodeBuild
|   |-- m5_4_monitoring_alerting/  # CloudWatch, alarms
|   |-- m5_5_compliance_docs/      # HIPAA, FDA PCCP, model card
|-- prompts/                       # Reusable prompt templates
|-- logs/                          # Decision records
|-- ResisTrack_Agent_Rules.md      # Agent rules for AI models
|-- ResisTrack_Roadmap.md          # Source roadmap document
```

## Navigation Guide

Start with `roadmap/00_context.md` for project context, then `roadmap/01_roadmap.md` for the full plan. Each module in `modules/` contains a `spec.md` with scope, dependencies, and a done-when checklist. Implementation code is located in `impl/` and tests in `tests/` within each module directory.

## Phase Overview

| Phase | Title | Duration | Modules |
|-------|-------|----------|---------|
| 1 | Secure Data Foundation | 3 Weeks | 6 |
| 2 | AI/ML Inference Engine | 4 Weeks | 7 |
| 3 | Clinical Integration & Alerts | 2 Weeks | 4 |
| 4 | Dashboard, Reporting & UX | 3 Weeks | 5 |
| 5 | Validation, MLOps & Hardening | 3 Weeks | 5 |

## Key Constraints

*   **HIPAA Compliance:** All data handling must adhere to HIPAA regulations.
*   **Data Privacy:** No PHI is permitted outside the VPC.
*   **Clinical Safety:** The model provides decision support and never prescribes medications directly.
*   **Validation:** Clinical validation is required before any model deployment.

## Team

Curelytics, Impact-AI-Thon 2026

## Getting Started (Local Development)

The codebase contains a functional Python backend logic (simulated) and a React frontend. The AWS infrastructure is defined in CDK but not deployed in this local demo.

### Prerequisites

*   Python 3.11+
*   Node.js 18+ (for frontend)
*   Unix-like environment (Linux/macOS) recommended for shell scripts.

### Installation

1.  **Clone the repository** (if you haven't already).
2.  **Run the setup script**:
    ```bash
    ./setup.sh
    ```
    This will:
    *   Create a Python virtual environment (`.venv`).
    *   Install Python dependencies from `requirements.txt`.
    *   Install the `resistrack` package in editable mode.
    *   Install frontend dependencies in `dashboard/`.

### Running the Demo (Backend Logic)

To verify the core logic (SageMaker Endpoint simulation and CDS Hooks service) without deploying to AWS:

1.  **Activate the virtual environment**:
    ```bash
    source .venv/bin/activate
    ```
2.  **Run the demo script**:
    ```bash
    python demo.py
    ```
    You should see output demonstrating a simulated AMR risk prediction and a Clinical Decision Support hook response.

### Running the Frontend

To start the React dashboard locally:

1.  Navigate to the dashboard directory:
    ```bash
    cd dashboard
    ```
2.  Start the development server:
    ```bash
    npm run dev
    ```
3.  Open your browser at the URL provided (usually `http://localhost:5173`).
    *   *Note:* The frontend is configured to point to `/api/v1`, but no backend server is running in this demo mode. The frontend will load but API calls will fail or need to be mocked.

### Project Status

*   **Backend:** Core logic for inference and hooks is implemented in `src/resistrack`. Real ML models (XGBoost/LSTM) are not included in the repo; the demo uses a mock predictor.
*   **Frontend:** React dashboard shell is implemented.
*   **Infrastructure:** AWS CDK constructs are in `infra/` but require AWS credentials and deployment to function.
