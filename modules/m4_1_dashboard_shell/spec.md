# M4.1 -- React Dashboard Shell and Auth
**Phase:** 4 -- Clinical Dashboard
**Duration:** ~2 days
**Status:** Not Started

## Objective
Scaffold the React.js dashboard application and implement secure authentication and role-based routing.

## Scope
This module involves building the core React.js application shell, which will be served via AWS CloudFront. It implements the SMART on FHIR launch sequence and direct OAuth 2.0 login for standalone access. The shell includes a robust role-based routing system that ensures each of the six user roles (Physician, ID Specialist, Pharmacist, Infection Control Officer, Hospital Admin, IT Admin) only sees their authorized views and data.

## Dependencies
- **Depends on:** None (independent, uses mock data initially). Uses M3.1 auth when integrated.
- **Depended on by:** M4.2, M4.3, M4.4, M4.5

## Inputs
- OAuth 2.0 / SMART on FHIR auth tokens
- User role and tenant context from JWT
- Mock data for initial UI development

## Outputs
- Deployed React.js application shell on CloudFront
- Authenticated user sessions with role-specific routing
- Base layout components for specialized views

## Implementation Notes
The dashboard must be designed for high availability and low latency. Use a modern state management library (e.g., React Context or Redux) to handle user sessions and global application state. Ensure the UI is responsive and follows clinical usability standards.

## Agent Rules
- RULE-DATA-01: Ensure no PHI is stored in local storage or browser state.
- RULE-DATA-02: Maintain secure communication with backend services.

## Done When
- [ ] React application is scaffolded and successfully deployed to CloudFront.
- [ ] SMART on FHIR launch flow is fully operational.
- [ ] OAuth 2.0 login works for standalone dashboard access.
- [ ] Role-based routing correctly restricts access per user role.
- [ ] All 6 roles render their respective dashboard layouts.
- [ ] Mock data renders correctly across all initial views.
