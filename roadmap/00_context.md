# Project Context: ResisTrack

## Overview
*   **Project Name:** ResisTrack
*   **Team:** Team Curelytics
*   **Event:** Impact-AI-Thon 2026
*   **Version:** 1.0.0

## Problem Statement
Antimicrobial Resistance (AMR) presents a critical diagnostic gap in modern healthcare. Standard culture results typically take 48 to 120 hours to return, forcing clinicians to prescribe broad-spectrum antibiotics empirically. This delay often leads to inappropriate prescribing, which accelerates resistance and worsens patient outcomes. There is an urgent need for accurate AMR risk prediction within the first 6 hours of hospital admission.

## Primary Goals
1.  Close the diagnostic gap by providing early AMR risk assessments.
2.  Reduce inappropriate antibiotic prescribing by at least 30%.
3.  Support clinical judgment through explainable AI without replacing the clinician's final decision.

## Technical Stack
*   **Cloud Infrastructure:** AWS (SageMaker, HealthLake, RDS, Lambda, CDK)
*   **Machine Learning:** Python 3.11, XGBoost, PyTorch LSTM, ClinicalBERT
*   **Frontend:** React.js
*   **Interoperability:** FHIR R4, CDS Hooks

## Constraints
*   **Compliance:** HIPAA-compliant architecture.
*   **Security:** No Protected Health Information (PHI) allowed outside the AWS VPC.
*   **Timeline:** Hackathon schedule.
*   **Resources:** Small development team.

## Non-Negotiable Rules
*   **Clinical Autonomy:** The model never prescribes medication or treatments. It only provides risk scores and supporting data.
*   **Confidence Threshold:** Any risk score with a confidence value below 0.60 must be marked with a LOW_CONFIDENCE_FLAG.
*   **Data Privacy:** PHI must remain within the secure VPC boundary at all times.
*   **Validation:** Clinical validation on a minimum of 1000 records is required before any production use.
