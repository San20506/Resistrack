# Phase 4: Dashboard, Reporting and UX

## Objective
Build a React.js clinical dashboard featuring ward heatmaps, patient timelines, and automated stewardship reports.

## Duration
1 Week

## Modules

### M4.1 React Dashboard Shell and Auth
*   **Duration:** 2 days
*   **Scope:** Deploy CloudFront application, implement SMART launch sequence, and configure role-based routing.
*   **Dependencies:** Independent (initially uses mock data).

### M4.2 Ward-Level AMR Risk Heatmap
*   **Duration:** 2 days
*   **Scope:** Create an interactive ward heatmap color-coded by Risk Tier with drill-down capabilities.
*   **Dependencies:** M4.1.

### M4.3 Patient Risk Timeline and SHAP Panel
*   **Duration:** 1 day
*   **Scope:** Develop per-patient risk history visualizations and a SHAP explainability panel (mandatory for HIGH and CRITICAL tiers).
*   **Dependencies:** M4.1.

### M4.4 Pharmacy and Infection Control Views
*   **Duration:** 1 day
*   **Scope:** Build specialized views including pharmacy priority lists and Infection Control outbreak trend analytics.
*   **Dependencies:** M4.1.

### M4.5 Automated Stewardship Reports
*   **Duration:** 1 day
*   **Scope:** Implement ReportLab PDF/CSV generator via Lambda cron jobs with S3 delivery.
*   **Dependencies:** M4.1.

## Success Criteria
*   Infection Control Officers can view ward heatmaps and drill into CRITICAL patient SHAP panels.
*   Weekly stewardship reports are automatically generated and delivered as PDFs.
*   Pharmacists see only authorized views relevant to their role.
*   Administrators have access to system metrics with zero PHI visibility.
