"""Response schemas for the runtime dashboard."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DomainWeightInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    similarity: float = Field(description="Similarity weight.")
    success_rate: float = Field(description="Success rate weight.")
    recency: float = Field(description="Recency weight.")
    cost: float = Field(description="Cost weight.")
    confidence: float = Field(description="Confidence weight.")


class LearningStatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_patterns: int = Field(description="Number of cached patterns.")
    per_domain_weights: dict[str, DomainWeightInfo] = Field(
        description="Ranking weights per domain."
    )
    anti_pattern_count: int = Field(0, description="Number of recorded anti-patterns.")


class ProviderStatsEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    model_id: str
    total_calls: int
    success_rate: float
    avg_latency_ms: float


class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning: LearningStatsResponse = Field(description="Learning system stats.")
    providers: list[ProviderStatsEntry] = Field(
        default_factory=list, description="Provider performance stats."
    )
