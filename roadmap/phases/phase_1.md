# Phase 1: Secure Data Foundation

## Objective
Establish a HIPAA-compliant AWS infrastructure and an HL7 to FHIR ingestion pipeline to support subsequent machine learning work.

## Duration
2 Weeks

## Modules

### M1.1 AWS Infrastructure Baseline
*   **Duration:** 3 days
*   **Scope:** Setup VPC, CDK Infrastructure as Code (IaC), KMS encryption keys, IAM roles, and Secrets Manager.
*   **Dependencies:** Independent.

### M1.2 Hospital Connectivity Layer
*   **Duration:** 2 days
*   **Scope:** Establish VPN or Direct Connect, implement TLS 1.3 mutual authentication, and configure API Gateway.
*   **Dependencies:** M1.1.

### M1.3 HL7 v2 to FHIR R4 Transformer
*   **Duration:** 3 days
*   **Scope:** Configure Mirth Connect channels to transform ADT, ORU, and RDE messages into FHIR bundles.
*   **Dependencies:** Independent.

### M1.4 FHIR Ingestion and PHI Tokenization
*   **Duration:** 2 days
*   **Scope:** Develop Lambda functions to validate, de-duplicate, and tokenize PHI before routing data to HealthLake.
*   **Dependencies:** M1.1, M1.3.

### M1.5 Data Storage Layer
*   **Duration:** 2 days
*   **Scope:** Deploy AWS HealthLake, RDS PostgreSQL, and S3 buckets with AES-256/KMS encryption.
*   **Dependencies:** M1.1.

### M1.6 Audit Logging and Compliance Baseline
*   **Duration:** 1 day
*   **Scope:** Enable CloudTrail, configure RDS audit logs, and finalize HIPAA service configurations.
*   **Dependencies:** M1.5.

## Dependency Diagram
M1.1 (Infra)
 ├── M1.2 (Connectivity)
 ├── M1.5 (Storage) ──> M1.6 (Audit)
 └── M1.4 (Ingestion) <── M1.3 (Transformer)

## Parallelization Strategy
*   **Stream A:** Infrastructure and Connectivity (M1.1, M1.2, M1.5, M1.6).
*   **Stream B:** Data Transformation and Ingestion (M1.3, M1.4).
*   Stream B can begin M1.3 immediately while Stream A builds the foundation required for M1.4.

## Success Criteria
*   Synthetic HL7 ADT A01 messages flow end-to-end into HealthLake.
*   All PHI is tokenized and encrypted at rest.
*   Audit logs are successfully captured in RDS.
*   Verification that zero PHI appears in CloudWatch logs.
