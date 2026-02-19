# M4.4 -- Pharmacy and Infection Control Views
**Phase:** 4 -- Clinical Dashboard
**Duration:** ~1 day
**Status:** Not Started

## Objective
Provide specialized, role-based views for Pharmacists and Infection Control Officers to manage stewardship and outbreak risks.

## Scope
This module develops two distinct dashboard views. The Pharmacy view provides a sortable list of HIGH and CRITICAL risk patients with specific, actionable de-escalation recommendations. The Infection Control view offers hospital-wide outbreak trend analytics, MDRO cluster alerts, and temporal trend charts. Access to these views is strictly controlled based on the user's authenticated role.

## Dependencies
- **Depends on:** M4.1
- **Depended on by:** None directly

## Inputs
- Filtered patient risk data (HIGH/CRITICAL for Pharmacy)
- Aggregate MDRO and outbreak trend data
- User role context from JWT

## Outputs
- Specialized Pharmacy stewardship dashboard
- Specialized Infection Control outbreak dashboard

## Implementation Notes
The Pharmacy view must prioritize patients based on risk severity and the potential for stewardship intervention. The Infection Control view should use advanced data visualization to highlight emerging clusters or trends. Ensure that the Hospital Admin role cannot see any PHI or de-identified patient-level data in these views.

## Agent Rules
- RULE-DATA-01: No PHI exposure in any dashboard view.
- RULE-DATA-02: Maintain strict role-based access control.

## Done When
- [ ] Pharmacy view displays only HIGH/CRITICAL patients with actionable recommendations.
- [ ] Infection Control view shows outbreak trends and MDRO clusters accurately.
- [ ] Each view is restricted to its authorized role via the routing system.
- [ ] Hospital Admin role is verified to have no access to patient-level data.
- [ ] UI components are optimized for the specific workflows of each role.
