# M1.2 -- Hospital Connectivity Layer
**Phase:** 1 -- Foundation and Data Ingestion
**Duration:** ~2 days
**Status:** Not Started

## Objective
Securely connect hospital data sources to the AWS VPC using encrypted tunnels and mutual authentication.

## Scope
- VPN or Direct Connect tunnel from hospital DMZ to the AWS VPC.
- TLS 1.3 with mutual certificate authentication (mTLS) for all data feeds.
- API Gateway deployment with JWT validation.
- Per-tenant rate limiting and throttling.

## Dependencies
- **Depends on:** M1.1
- **Depended on by:** M1.3, M1.4 (indirectly via data flow)

## Inputs
- Hospital network configuration (IP ranges, gateway details).
- Client certificates for mTLS.
- JWT signing keys/provider details.

## Outputs
- Established VPN/Direct Connect tunnel.
- API Gateway endpoint URL.
- mTLS configuration and trust store.

## Implementation Notes
- Use AWS Client VPN or Site-to-Site VPN for the tunnel.
- API Gateway must be private or restricted to specific source IPs.
- Ensure all data in transit uses TLS 1.3.

## Agent Rules
- RULE-DATA-01: PHI must remain within the VPC boundary.
- RULE-DATA-03: All data in transit must be encrypted.

## Done When
- [ ] VPN tunnel established (or simulated for development).
- [ ] TLS 1.3 mutual auth configured and verified.
- [ ] API Gateway deployed with JWT validation active.
- [ ] Rate limiting per tenant is operational.
