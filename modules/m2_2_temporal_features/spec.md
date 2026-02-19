# M2.2 -- Temporal Feature Extraction
**Phase:** 2 -- ML Development and Ensemble
**Duration:** ~2 days
**Status:** Not Started

## Objective
Transform time-series clinical data into normalized tensors for temporal modeling with LSTM.

## Scope
- Build a 72-hour rolling window extractor.
- Process 8 lab values and 5 vital signs into a tensor of shape (batch, 72, 13).
- Implement Z-score normalization using hospital-cohort training statistics.
- Handle missing time steps via forward-fill logic.

## Dependencies
- **Depends on:** M2.1
- **Depended on by:** M2.5

## Inputs
- Structured features and raw time-series data from M2.1 Feature Store.

## Outputs
- Tensors of shape (batch, 72, 13) for LSTM training and inference.

## Implementation Notes
- Normalization must use cohort-specific statistics, not global ones, to account for hospital-level variance.
- Forward-fill should be used for missing timestamps within the 72-hour window.
- Ensure the extractor can handle varying sampling frequencies by resampling to a fixed hourly grid if necessary.

## Agent Rules
- Agent Rules section 4.2: Temporal features specification.
- RULE-SAFETY-03: Data completeness monitoring.

## Done When
- [ ] 72h rolling window extractor produces the correct tensor shape (batch, 72, 13).
- [ ] Z-score normalization uses cohort-specific statistics.
- [ ] Forward-fill correctly handles missing timestamps.
- [ ] `DATA_COMPLETENESS_SCORE` flags cases with >30% missing data.
- [ ] Unit tests pass for various edge cases (e.g., all missing, single data point).
