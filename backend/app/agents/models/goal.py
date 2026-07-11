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
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    objective_id: str = ""
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
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
