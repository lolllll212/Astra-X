"""Task model for the agent framework.

A task is a single unit of work that an agent plans and executes. Tasks
are composed into plans and tracked through their lifecycle by the task
graph and executor.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(StrEnum):
    """Lifecycle states for a single task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    """A single unit of work within an agent plan.

    Attributes:
        id: Unique task identifier.
        description: Human-readable description of what to do.
        status: Current lifecycle status.
        capability: Abstract capability needed (e.g. ``"search_web"``).
        tool_name: Resolved concrete tool name, set at execution time.
        tool_args: Arguments to pass to the tool, if any.
        result: Text output produced by executing this task.
        error: Error message if the task failed.
        dependencies: Task IDs that must complete before this one.
        metadata: Arbitrary structured metadata for extensibility.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique task identifier.")
    description: str = Field(min_length=1, description="What to do.")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status.")
    capability: str | None = Field(default=None, description="Capability needed (e.g. 'search_web').")
    profile: dict[str, object] | None = Field(
        default=None,
        description=(
            "Capability profile for adaptive model selection. "
            "Keys: ``requires_coding``, ``reasoning`` (none/low/medium/deep), "
            "``prefers_speed``, ``prefers_large_context``, ``requires_vision``."
        ),
    )
    tool_name: str | None = Field(default=None, description="Resolved tool name (set at execution time).")
    tool_args: dict[str, object] = Field(default_factory=dict, description="Tool arguments.")
    result: str | None = Field(default=None, description="Execution output.")
    error: str | None = Field(default=None, description="Error message.")
    dependencies: list[str] = Field(default_factory=list, description="Dependency task IDs.")
    metadata: dict[str, object] = Field(default_factory=dict, description="Arbitrary metadata.")
