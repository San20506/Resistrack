# ResisTrack Learnings

## Project Context
- AMR Risk Prediction Platform for Impact-AI-Thon 2026
- Tech: CDK TypeScript (infra), Python 3.11 (ML/backend), React.js (frontend)
- AWS: SageMaker, HealthLake, RDS, Lambda, CDK
- HIPAA compliant, no PHI outside VPC
- Model NEVER prescribes, LOW_CONFIDENCE_FLAG < 0.60
- mypy strict, random_state=42, >=80% test coverage
