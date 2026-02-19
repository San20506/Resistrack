# Project Roadmap: ResisTrack

## Phase Overview
The project is structured into five distinct phases over a total duration of approximately 7.5 weeks.

| Phase | Duration | Focus Area |
| :--- | :--- | :--- |
| Phase 1 | 2 Weeks | Secure Data Foundation |
| Phase 2 | 2 Weeks | AI/ML Inference Engine |
| Phase 3 | 1.5 Weeks | Clinical Integration and Alerts |
| Phase 4 | 1 Week | Dashboard, Reporting and UX |
| Phase 5 | 1 Week | Validation, MLOps and Hardening |

## Module Dependency Map

### Phase 1: Data Foundation
*   M1.1 -> M1.2, M1.4, M1.5
*   M1.3 -> M1.4
*   M1.5 -> M1.6

### Phase 2: ML Engine
*   M2.1 -> M2.2, M2.3, M2.4
*   M2.2 -> M2.5
*   M2.4 + M2.5 + M2.3 -> M2.6
*   M2.6 -> M2.7

### Phase 3: Clinical Integration
*   M2.7 + M3.1 -> M3.2
*   M3.2 -> M3.3, M3.4

### Phase 4: UX and Reporting
*   M4.1 -> M4.2, M4.3, M4.4, M4.5

### Phase 5: Hardening
*   Phase 2 -> M5.1
*   M5.1 -> M5.2
*   M5.2 -> M5.3, M5.4
*   M5.5 runs in parallel to other Phase 5 tasks

## Non-Negotiable Constraints

| Constraint | Requirement |
| :--- | :--- |
| Data Privacy | No PHI outside AWS VPC |
| Model Safety | Model never prescribes; clinical judgment only |
| Reliability | LOW_CONFIDENCE_FLAG for scores < 0.60 |
| Compliance | HIPAA-compliant architecture and logging |
| Validation | Minimum 1000 records for clinical validation |
