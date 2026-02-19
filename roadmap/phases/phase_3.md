# Phase 3: Clinical Integration and Alerts

## Objective
Surface risk scores into the EHR workflow using CDS Hooks, SMART on FHIR authentication, and multi-role notification systems.

## Duration
1.5 Weeks

## Modules

### M3.1 SMART on FHIR Authorization
*   **Duration:** 2 days
*   **Scope:** Implement OAuth 2.0 and JWT validation. Configure IAM Role-Based Access Control (RBAC) for six user roles: Physician, ID Specialist, Pharmacist, Infection Control, Admin, and IT Admin.
*   **Dependencies:** Independent.

### M3.2 CDS Hooks Service
*   **Duration:** 3 days
*   **Scope:** Develop hooks for patient-view, order-sign, and encounter-discharge events. Create CDS Cards displaying SHAP factors and providing three clinician response options: Acknowledged, Override, and Escalate.
*   **Dependencies:** M2.7, M3.1.

### M3.3 Override Rate Monitoring
*   **Duration:** 1 day
*   **Scope:** Build a 30-day rolling override rate tracker. Implement automated reporting if the override rate exceeds 60%.
*   **Dependencies:** M3.2.

### M3.4 Multi-Role Notification Dispatch
*   **Duration:** 2 days
*   **Scope:** Configure SNS routing based on Risk Tier to alert Pharmacy and Infection Control teams. Ensure no PHI is included in notification payloads.
*   **Dependencies:** M3.2.

## Success Criteria
*   CDS Cards appear in the Epic SMART sandbox within 2 seconds of the trigger event.
*   Clinician override actions are successfully logged to the RDS database.
*   CRITICAL-tier risk scores trigger SNS notifications to Pharmacy and Infection Control.
*   Verification that notification payloads contain zero PHI.
