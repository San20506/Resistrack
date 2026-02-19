# M2.5 -- PyTorch LSTM Temporal Model
**Phase:** 2 -- ML Development and Ensemble
**Duration:** ~2 days
**Status:** Not Started

## Objective
Train a PyTorch LSTM model on time-series clinical data to capture temporal trends in patient risk.

## Scope
- Build and train a PyTorch LSTM model using 72-hour time-series tensors from M2.2.
- Output a trend-risk vector to be used as an additional feature for the final ensemble.
- Ensure full reproducibility with `random_state=42`.

## Dependencies
- **Depends on:** M2.2
- **Depended on by:** M2.6

## Inputs
- Tensors of shape (batch, 72, 13) from M2.2.

## Outputs
- Trend-risk vector per patient for ensemble input.

## Implementation Notes
- Use SageMaker training jobs for model execution.
- Implement early stopping based on validation loss to prevent overfitting.
- Log all training metrics and model versions to SageMaker Experiments.
- Follow `mypy` strict type checking for the model implementation.

## Agent Rules
- RULE-TRAIN-05: Experiment logging and tracking.
- Code Quality: Reproducibility (random_state=42), mypy strict.

## Done When
- [ ] LSTM is trained on temporal tensors.
- [ ] Model produces a valid trend-risk vector output.
- [ ] Training and validation metrics are logged to SageMaker Experiments.
- [ ] `random_state=42` is set and verified.
- [ ] Reproducible results are confirmed across multiple runs.
