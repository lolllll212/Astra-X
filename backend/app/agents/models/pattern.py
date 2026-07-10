from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AntiPattern(BaseModel):
    """A negative execution pattern — what *not* to do.

    Captures failure modes, tool-specific pitfalls, and approaches that
    consistently underperform.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID primary key.",
    )
    goal_pattern: str = Field(
        min_length=1,
        description="Normalized goal this anti-pattern applies to.",
    )
    warning: str = Field(
        description="What to avoid and why (e.g. 'Avoid Tool X for large repos — timeouts').",
    )
    failure_reason: str = Field(
        default="",
        description="Root cause (e.g. 'tool_timeout', 'low_confidence', 'bad_strategy').",
    )
    suggestion: str | None = Field(
        default=None,
        description="Alternative approach that might work better.",
    )
    tags: list[str] = Field(default_factory=list)
    occurrence_count: int = Field(default=1, ge=1)
    last_seen_at: datetime | None = Field(default=None)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


class ExecutionPattern(BaseModel):
    """An execution strategy learned from past successful agent runs.

    Each pattern captures what capabilities/tools worked well for a
    type of goal, how they were sequenced, and what overall approach
    produced good results.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="UUID primary key.",
    )
    goal_pattern: str = Field(
        min_length=1,
        description=(
            "Normalized goal pattern this applies to, "
            "e.g. 'research {topic}' or 'write code for {task}'."
        ),
    )
    capability: str | None = Field(
        default=None,
        description="Primary capability used (e.g. 'search_web', 'execute_python').",
    )
    strategy_summary: str = Field(
        default="",
        description="Human-readable description of the strategy that worked.",
    )
    plan_template: str | None = Field(
        default=None,
        description="The task sequence as a newline-separated list of descriptions.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Keywords for matching against new goals.",
    )
    success_count: int = Field(
        default=1,
        ge=1,
        description="Number of times this pattern led to a successful outcome.",
    )
    total_count: int = Field(
        default=1,
        ge=1,
        description="Number of times this pattern has been applied.",
    )
    avg_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average reflection confidence across all uses.",
    )
    avg_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average feedback importance across all uses.",
    )
    avg_execution_cost_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average execution cost in milliseconds across all uses.",
    )
    last_reflection_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reflection confidence from the most recent application.",
    )
    last_success_at: datetime | None = Field(
        default=None,
        description="When this pattern was last successfully applied.",
    )
    created_at: datetime | None = Field(
        default=None,
    )
    updated_at: datetime | None = Field(
        default=None,
    )
