# M4.2 -- Ward-Level AMR Risk Heatmap
**Phase:** 4 -- Clinical Dashboard
**Duration:** ~2 days
**Status:** Not Started

## Objective
Provide a real-time, ward-level visualization of AMR risk distribution across the hospital to facilitate proactive infection control.

## Scope
This module implements an interactive heatmap showing the distribution of AMR risk across different hospital wards or units. Wards are color-coded based on their aggregate Risk Tier: LOW (green), MEDIUM (yellow), HIGH (orange), and CRITICAL (red). Users can drill down from a ward view to see individual patient risk timelines.

## Dependencies
- **Depends on:** M4.1
- **Depended on by:** None directly

## Inputs
- Real-time aggregate risk scores per ward
- Ward/Unit metadata and layout information
- Individual patient risk scores and locations

## Outputs
- Interactive ward-level heatmap visualization
- Navigation links to individual patient timelines (M4.3)

## Implementation Notes
The heatmap must update in real-time as new risk scores are generated. Use an accessible color palette that is colorblind-friendly. The visualization should be optimized for desktop viewing in clinical environments.

## Agent Rules
- RULE-DATA-01: Ensure ward-level aggregates do not inadvertently expose PHI.
- RULE-DATA-02: Securely fetch real-time risk data from the backend.

## Done When
- [ ] Heatmap renders all hospital wards with correct color coding.
- [ ] Real-time updates accurately reflect the current risk distribution.
- [ ] Drill-down functionality successfully navigates to the patient timeline.
- [ ] UI is responsive and performs well on desktop browsers.
- [ ] Color palette is verified as accessible and colorblind-friendly.
