"""Constants and enums for ResisTrack."""

from enum import StrEnum
from typing import Final


class RiskTier(StrEnum):
    """AMR Risk tiers based on score ranges."""

    LOW = "LOW"  # 0-24
    MEDIUM = "MEDIUM"  # 25-49
    HIGH = "HIGH"  # 50-74
    CRITICAL = "CRITICAL"  # 75-100


# Mapping of risk tiers to score ranges (inclusive)
RISK_TIER_RANGES: Final[dict[RiskTier, tuple[int, int]]] = {
    RiskTier.LOW: (0, 24),
    RiskTier.MEDIUM: (25, 49),
    RiskTier.HIGH: (50, 74),
    RiskTier.CRITICAL: (75, 100),
}

CONFIDENCE_THRESHOLD: Final[float] = 0.60
RANDOM_STATE: Final[int] = 42

ANTIBIOTIC_CLASSES: Final[list[str]] = [
    "penicillins",
    "cephalosporins",
    "carbapenems",
    "fluoroquinolones",
    "aminoglycosides",
]

__all__ = [
    "RiskTier",
    "RISK_TIER_RANGES",
    "CONFIDENCE_THRESHOLD",
    "RANDOM_STATE",
    "ANTIBIOTIC_CLASSES",
]
