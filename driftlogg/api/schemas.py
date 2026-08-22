"""Request and response models for the scoring API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RiskBand(str, Enum):
    """Coarse risk buckets for display.

    Thresholds are deliberate: `operating_points.csv` from training shows the
    precision/recall trade at each cutoff, and these bands should be set from
    that curve rather than picked round.
    """

    HIGH = "high"
    ELEVATED = "elevated"
    LOW = "low"
    UNKNOWN = "unknown"

    @classmethod
    def from_score(cls, score: float | None) -> RiskBand:
        """Bucket a probability into a display band."""
        if score is None:
            return cls.UNKNOWN
        if score >= 0.60:
            return cls.HIGH
        if score >= 0.30:
            return cls.ELEVATED
        return cls.LOW


class FeatureContribution(BaseModel):
    """One feature's contribution to a package's score."""

    feature: str
    value: float | None
    label: str = Field(description="Human-readable explanation of this signal.")


class PackageRisk(BaseModel):
    """Scored risk for a single dependency."""

    package: str
    repo: str | None = Field(default=None, description="owner/name on GitHub.")
    score: float | None = Field(default=None, description="P(abandoned within horizon).")
    band: RiskBand = RiskBand.UNKNOWN
    reasons: list[FeatureContribution] = Field(default_factory=list)
    error: str | None = Field(default=None, description="Why scoring failed, if it did.")

    @property
    def scored(self) -> bool:
        return self.score is not None


class ScoreRequest(BaseModel):
    """Score an explicit list of package names."""

    packages: list[str] = Field(min_length=1, max_length=200)


class ScoreResponse(BaseModel):
    """Result of scoring a set of dependencies."""

    generated_at: datetime
    model_kind: str = Field(description="Which model produced the scores.")
    horizon_days: int
    total: int
    scored: int
    results: list[PackageRisk]

    @property
    def high_risk_count(self) -> int:
        return sum(1 for r in self.results if r.band is RiskBand.HIGH)


class HealthResponse(BaseModel):
    """Service health and model availability."""

    status: str
    model_loaded: bool
    model_kind: str
    github_token_present: bool
