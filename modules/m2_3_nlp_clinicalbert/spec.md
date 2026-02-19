# M2.3 -- NLP Feature Extraction / ClinicalBERT
**Phase:** 2 -- ML Development and Ensemble
**Duration:** ~2 days
**Status:** Not Started

## Objective
Extract risk vectors from unstructured clinical notes using a domain-specific transformer model.

## Scope
- Load `emilyalsentzer/Bio_ClinicalBERT` checkpoint from HuggingFace.
- Implement note selection logic: last 3 clinical notes (physician + nursing).
- Apply truncation strategy: first 128 tokens + last 384 tokens (max 512 tokens total).
- Exclude notes older than 72 hours unless no newer notes exist.
- Fine-tune a classification head to produce a 32-dimensional AMR risk vector.

## Dependencies
- **Depends on:** M2.1
- **Depended on by:** M2.6

## Inputs
- Clinical notes from HealthLake or S3.

## Outputs
- 32-dimensional risk vector per patient for ensemble input.

## Implementation Notes
- ClinicalBERT must run inside the VPC; no external API calls with PHI are permitted.
- Cache the model checkpoint within the VPC to avoid external dependencies during runtime.
- Use SageMaker training jobs for fine-tuning the classification head.

## Agent Rules
- RULE-DATA-03: ClinicalBERT runs inside VPC only; no external API calls.
- RULE-TRAIN-04: Must use a clinical-domain pre-trained checkpoint.
- Agent Rules section 4.3: NLP feature specification.

## Done When
- [ ] `Bio_ClinicalBERT` is loaded and cached within the VPC.
- [ ] Note selection logic correctly identifies the last 3 notes within the 72h window.
- [ ] Truncation strategy (128 first + 384 last) is implemented.
- [ ] Fine-tuning produces a valid 32-dimensional output vector.
- [ ] No PHI leaves the VPC during processing.
- [ ] Processing runs successfully on a SageMaker training job.
