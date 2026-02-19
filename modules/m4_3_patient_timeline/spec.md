# M4.3 -- Patient Risk Timeline and SHAP Panel
**Phase:** 4 -- Clinical Dashboard
**Duration:** ~1 day
**Status:** Not Started

## Objective
Visualize individual patient risk progression and provide transparent, explainable AI insights to clinicians.

## Scope
This module displays a historical timeline of a patient's AMR risk scores throughout their admission. It also includes a SHAP explainability panel that highlights the top-5 factors contributing to the current risk score. Each factor is shown with its direction (INCREASES or DECREASES risk) and a plain English explanation. This panel is mandatory for all HIGH and CRITICAL tier alerts.

## Dependencies
- **Depends on:** M4.1
- **Depended on by:** None directly

## Inputs
- Historical risk scores for a specific patient_token
- SHAP feature importance values and metadata
- Human-readable feature mappings

## Outputs
- Interactive patient risk timeline chart
- SHAP explainability panel with top-5 factors

## Implementation Notes
The timeline chart should allow clinicians to see how risk has evolved in response to clinical events (e.g., new antibiotic orders). SHAP explanations must be clear and actionable, avoiding technical jargon. The panel must automatically expand or be prominently displayed when a patient is in the HIGH or CRITICAL tier.

## Agent Rules
- RULE-CDS-02: SHAP explainability is mandatory for HIGH and CRITICAL alerts.
- RULE-DATA-01: Ensure only de-identified patient_token is used for data fetching.

## Done When
- [ ] Timeline chart accurately displays risk score progression over the admission.
- [ ] SHAP panel shows the top-5 features with correct direction and human-readable text.
- [ ] SHAP panel automatically displays for patients in HIGH or CRITICAL tiers.
- [ ] Data refreshes in real-time as new assessments are available.
- [ ] Visualization is clear and intuitive for clinical users.
