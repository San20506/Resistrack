# M1.1 -- AWS Infrastructure Baseline
**Phase:** 1 -- Foundation and Data Ingestion
**Duration:** ~3 days
**Status:** Not Started

## Objective
Establish a secure, compliant AWS foundation using Infrastructure as Code (CDK) to host all Resistrack services and data.

## Scope
- Private VPC with isolated subnets, Security Groups, and NACLs.
- AWS CDK (TypeScript) project for all infrastructure management.
- AWS KMS with per-tenant customer-managed keys (CMKs).
- IAM roles with strict least-privilege RBAC.
- AWS Secrets Manager for secure credential storage.

## Dependencies
- **Depends on:** None (independent)
- **Depended on by:** M1.2, M1.4, M1.5

## Inputs
- AWS Account credentials (via environment variables/CLI).
- Tenant configuration metadata (tenant IDs for KMS key creation).

## Outputs
- VPC ID and Subnet IDs.
- KMS Key ARNs (per tenant).
- IAM Role ARNs for application services.
- Secrets Manager secret ARNs.

## Implementation Notes
- All infrastructure must be defined in TypeScript CDK.
- No resources should be created manually via the AWS Console.
- VPC must not have public subnets; use NAT Gateways or VPC Endpoints for external access.
- Follow AWS Well-Architected Framework security pillars.

## Agent Rules
- RULE-DATA-01: PHI must remain within the VPC boundary.
- Code Quality Standards: All infrastructure must be managed via CDK.

## Done When
- [ ] CDK deploys VPC with private subnets.
- [ ] KMS key created per tenant.
- [ ] IAM roles provisioned with least-privilege.
- [ ] Secrets Manager configured.
- [ ] No console-created resources exist.
- [ ] CDK synth passes without errors.
