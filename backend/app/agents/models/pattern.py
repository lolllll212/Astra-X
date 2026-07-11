from __future__ import annotations

from datetime import UTC, datetime
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
    preferred_provider: str | None = Field(
        default=None,
        description="Provider that worked best for this pattern (e.g. 'ollama', 'openai').",
    )
    preferred_model: str | None = Field(
        default=None,
        description="Model that worked best for this pattern (e.g. 'llama3.1', 'gpt-4').",
    )
    tool_sequence: list[str] = Field(
        default_factory=list,
        description="Ordered list of tools used, e.g. ['search_web', 'scrape_web', 'execute_python'].",
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

    def apply_confidence_decay(self, decay_factor: float = 0.9) -> ExecutionPattern:
        """Return a new pattern with decayed confidence when pattern fails or stops being used.

        Args:
            decay_factor: Multiplier for confidence decay (default 0.9 = 10% decay).

        Returns:
            A new ExecutionPattern with decayed confidence values.
        """
        new_avg_conf = max(0.0, self.avg_confidence * decay_factor)
        new_last_ref_conf = max(0.0, self.last_reflection_confidence * decay_factor)
        return self.model_copy(update={
            "avg_confidence": new_avg_conf,
            "last_reflection_confidence": new_last_ref_conf,
            "updated_at": datetime.now(UTC),
        })

    def record_failure(self, decay_factor: float = 0.95) -> ExecutionPattern:
        """Record a failed application and return a new pattern with decayed confidence.

        Args:
            decay_factor: Multiplier for confidence decay (default 0.95 = 5% decay per failure).

        Returns:
            A new ExecutionPattern with incremented total_count and decayed confidence.
        """
        new_total = self.total_count + 1
        new_avg_conf = max(0.0, self.avg_confidence * decay_factor)
        new_last_ref_conf = max(0.0, self.last_reflection_confidence * decay_factor)
        return self.model_copy(update={
            "total_count": new_total,
            "avg_confidence": new_avg_conf,
            "last_reflection_confidence": new_last_ref_conf,
            "updated_at": datetime.now(UTC),
        })
