# Phase 5: Validation, MLOps and Hardening

## Objective
Complete clinical validation, establish automated retraining pipelines, and finalize compliance documentation for production readiness.

## Duration
1 Week

## Modules

### M5.1 Clinical Validation Gate
*   **Duration:** 2 days
*   **Scope:** Perform evaluation on a held-out test set of at least 1000 patients. Conduct subgroup analysis and obtain model card sign-off.
*   **Dependencies:** All Phase 2 modules.

### M5.2 SageMaker MLOps Retraining Pipeline
*   **Duration:** 2 days
*   **Scope:** Implement monthly automated retraining, Population Stability Index (PSI) drift monitoring, and blue/green deployment with rollback capabilities.
*   **Dependencies:** M5.1.

### M5.3 CI/CD Pipeline
*   **Duration:** 1 day
*   **Scope:** Setup CodePipeline and CodeBuild with strict mypy type checking, minimum 80% test coverage, and manual deployment gates.
*   **Dependencies:** M5.2.

### M5.4 Production Monitoring and Alerting
*   **Duration:** 1 day
*   **Scope:** Configure CloudWatch dashboards with alarms for latency and error rates. Ensure all logs are de-identified.
*   **Dependencies:** M5.2.

### M5.5 Compliance Documentation and Model Card
*   **Duration:** 1 day
*   **Scope:** Complete HIPAA checklist, FDA PCCP documentation, and Business Associate Agreement (BAA) status. Finalize the model card.
*   **Dependencies:** Independent (runs in parallel with other Phase 5 tasks).

## Acceptance Metrics

| Metric | Target |
| :--- | :--- |
| AUC-ROC | >= 0.82 |
| AUPRC | >= 0.70 |
| Sensitivity @ 80% Spec | >= 0.80 |
| False Positive Rate (FPR) | <= 0.20 |
| Brier Score | <= 0.15 |
| Inference Latency (p95) | <= 2000ms |

## Success Criteria
*   Model card is signed off with all acceptance metrics met.
*   Blue/green canary deployment process is verified.
*   CloudWatch confirms p95 latency is within the 2000ms limit.
*   PSI drift detection successfully triggers the retraining pipeline.
*   System rollback completes in under 15 minutes.
