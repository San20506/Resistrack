# M5.3 -- CI/CD Pipeline
**Phase:** 5 -- MLOps and Governance
**Duration:** ~1 day
**Status:** Not Started

## Objective
Implement a robust CI/CD pipeline to automate the building, testing, and deployment of the ResisTrack platform while enforcing high code quality standards.

## Scope
This module uses AWS CodePipeline and CodeBuild to create an automated CI/CD workflow. The pipeline enforces `mypy --strict` on all Python code and requires at least 80% unit test coverage on feature engineering scripts. It includes a deployment gate that prevents any model or code deployment if regression tests fail.

## Dependencies
- **Depends on:** M5.2
- **Depended on by:** None directly

## Inputs
- Source code from the repository
- Unit and regression test suites
- Configuration for CodePipeline and CodeBuild

## Outputs
- Automated CI/CD pipeline in AWS
- Build artifacts and test reports
- Deployed services in staging/production environments

## Implementation Notes
The pipeline should be designed to provide fast feedback to developers. All processing scripts must be containerized to ensure consistency across environments. Secrets must never be stored in the code and should be managed via AWS Secrets Manager.

## Agent Rules
- Code Quality Standards: Enforce `mypy --strict` and >= 80% unit test coverage.
- Containerization: All processing scripts must be containerized.
- Security: No secrets in code; use AWS Secrets Manager.

## Done When
- [ ] CodePipeline is configured with source, build, test, and deploy stages.
- [ ] `mypy --strict` passes for all Python modules in the pipeline.
- [ ] Unit test coverage of >= 80% is enforced and verified.
- [ ] Deployment gate successfully blocks deployments on regression test failure.
- [ ] The pipeline runs end-to-end, deploying a verified build to production.
