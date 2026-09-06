"""Goal hierarchy — Mission → Objective → Goal → Tasks → Actions.

This enables long-running autonomous work by persisting goals across
sessions.  A single *Mission* contains multiple *Objectives*, each
objective is broken into *Goals* (one per agent run), and each goal
produces *Tasks* and *Actions*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


class MissionStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ObjectiveStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class GoalStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_PRIORITY_BASE: dict[Priority, float] = {
    Priority.CRITICAL: 1.0,
    Priority.HIGH: 0.75,
    Priority.MEDIUM: 0.5,
    Priority.LOW: 0.25,
}


def default_priority_score(
    priority: Priority,
    urgency: datetime | None = None,
    estimated_cost_ms: float = 0.0,
    success_probability: float = 0.0,
    now: datetime | None = None,
) -> float:
    """Compute a composite 0–1 priority score from a goal's fields.

    Weighting (adds to 1.0):

    * **35 %** — base priority (CRITICAL=1.0 … LOW=0.25)
    * **30 %** — urgency (overdue=1.0, linear decay over 7 days)
    * **20 %** — success probability (higher = better)
    * **15 %** — inverse cost (cheaper = better, capped at 60 s)
    """
    p_base = _PRIORITY_BASE.get(priority, 0.5)

    # Urgency: days until deadline (closer = higher)
    urgency_score = 0.5
    if urgency is not None:
        now = now or datetime.now(timezone.utc)
        remaining = (urgency - now).total_seconds()
        if remaining <= 0:
            urgency_score = 1.0
        else:
            urgency_score = 1.0 - min(remaining / (7 * 86400), 1.0)

    # Cost: cheaper is better
    cost_score = (
        1.0 - min(estimated_cost_ms / 60000.0, 1.0)
        if estimated_cost_ms > 0
        else 0.5
    )

    # Success: higher is better
    success_score = success_probability if success_probability > 0 else 0.5

    return (
        p_base * 0.35
        + urgency_score * 0.30
        + success_score * 0.20
        + cost_score * 0.15
    )


@dataclass
class PrioritizedGoal:
    """A goal with its computed priority score and a human-readable reason."""

    goal: GoalNode
    score: float
    reason: str


@dataclass
class Mission:
    """A long-running, high-level endeavor.

    Persists across sessions.  Example: *"Build an e-commerce platform"*.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    status: MissionStatus = MissionStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Objective:
    """A major milestone within a mission.

    Example: *"Implement user authentication"*.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    mission_id: str = ""
    title: str = ""
    description: str = ""
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    order: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalNode:
    """A single session-scoped unit of work within an objective.

    Maps to one coordinator ``run()`` call.  Example: *"Create login
    API endpoint"*.

    Extended with prioritization fields: *priority*, *urgency*,
    *estimated_cost_ms*, *success_probability*, and *dependencies*
    so the Goal Manager can rank goals and choose what to work on next.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    objective_id: str = ""
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: Priority = Priority.MEDIUM
    urgency: datetime | None = None
    estimated_cost_ms: float = 0.0
    success_probability: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    plan_id: str | None = None
    result_summary: str = ""
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """An atomic logged step — a tool invocation or LLM call.

    Captured by the coordinator after each task execution.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    goal_id: str = ""
    task_id: str = ""
    tool_name: str | None = None
    description: str = ""
    input_summary: str = ""
    output_summary: str = ""
    status: str = "completed"
    latency_ms: float = 0.0
    reflection_confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
