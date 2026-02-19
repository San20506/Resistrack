# M3.1 -- SMART on FHIR Authorization
**Phase:** 3 -- Clinical Integration
**Duration:** ~2 days
**Status:** Not Started

## Objective
Implement a secure OAuth 2.0 and OpenID Connect authorization server to manage SMART on FHIR app launches and validate requests.

## Scope
This module establishes the security perimeter for the ResisTrack platform. It handles the SMART on FHIR launch sequence for major EHR vendors like Epic and Cerner. The implementation includes JWT token issuance and validation for every CDS Hook request. It also configures IAM Role-Based Access Control (RBAC) for the six defined user roles: Physician, ID Specialist, Pharmacist, Infection Control Officer, Hospital Admin, and IT Admin.

## Dependencies
- **Depends on:** None (independent)
- **Depended on by:** M3.2, M4.1

## Inputs
- SMART launch parameters from EHR (client_id, iss, launch)
- User credentials and role assignments from hospital identity provider
- Authorization codes for token exchange

## Outputs
- Validated JWT access tokens with appropriate scopes
- Refresh tokens for session persistence
- User identity and role context for downstream services

## Implementation Notes
The authorization server must strictly adhere to the SMART on FHIR App Launch Framework. JWTs should contain custom claims for hospital_tenant_id and user_role to facilitate multi-tenancy and RBAC. All cryptographic operations must use industry-standard libraries and secure key management via AWS KMS.

## Agent Rules
- RULE-DATA-01: Ensure no PHI is included in JWT claims or logged during the auth flow.
- RULE-DATA-02: Maintain strict VPC isolation for the authorization service.

## Done When
- [ ] OAuth 2.0 authorization server is fully deployed and accessible.
- [ ] SMART launch flow successfully completes in the Epic sandbox environment.
- [ ] JWT validation middleware is operational and protecting CDS Hook endpoints.
- [ ] All 6 user roles are configured with specific scopes and verified.
- [ ] Token refresh mechanism is working as expected.
