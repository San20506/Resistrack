# Architecture Decision Log

Purpose: Record architectural and design decisions with rationale.

## ADR-001: Use modular directory structure with per-module specs
**Date:** 2026-02-18
**Status:** Accepted
**Module(s):** All

### Context
The project requires a highly organized structure to manage 27 distinct modules across 5 phases. We need a way to ensure each module has clear boundaries, documented requirements, and isolated implementation and testing.

### Decision
We will use a modular directory structure where each module is contained within its own directory under `modules/`. Each module directory will include:
*   `spec.md`: A specification file defining scope, dependencies, and success criteria.
*   `impl/`: A directory for implementation code.
*   `tests/`: A directory for unit and integration tests.

### Rationale
This structure promotes separation of concerns and allows for parallel development. The inclusion of a `spec.md` in every module ensures that requirements are locked before implementation begins, and the `tests/` directory ensures that every module is verified against its spec.

### Consequences
*   Developers must create and follow a `spec.md` for every new module.
*   Dependencies between modules are explicitly tracked.
*   The project remains navigable despite its large number of components.
