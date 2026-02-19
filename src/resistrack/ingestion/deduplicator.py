"""FHIR resource deduplication by resource ID."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeduplicationResult:
    """Result of deduplication pass over FHIR resources."""

    unique_resources: list[dict[str, Any]]
    duplicates_removed: int
    total_processed: int


class ResourceDeduplicator:
    """Deduplicate FHIR resources by resource type + ID."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def deduplicate(self, resources: list[dict[str, Any]]) -> DeduplicationResult:
        """Remove duplicate resources from a list."""
        unique: list[dict[str, Any]] = []
        dupes = 0

        for resource in resources:
            key = self._resource_key(resource)
            if key in self._seen:
                dupes += 1
            else:
                self._seen.add(key)
                unique.append(resource)

        return DeduplicationResult(
            unique_resources=unique,
            duplicates_removed=dupes,
            total_processed=len(resources),
        )

    def reset(self) -> None:
        """Clear deduplication state."""
        self._seen.clear()

    @staticmethod
    def _resource_key(resource: dict[str, Any]) -> str:
        """Generate a unique key for a resource."""
        rt = resource.get("resourceType", "Unknown")
        rid = resource.get("id", "")

        if not rid:
            identifiers = resource.get("identifier", [])
            if identifiers and isinstance(identifiers, list):
                first = identifiers[0]
                system = first.get("system", "")
                value = first.get("value", "")
                rid = f"{system}|{value}"

        return f"{rt}/{rid}"
