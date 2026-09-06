"""Plan model for the agent framework.

A plan is the output of the planner — a sequence (or graph) of tasks
that together achieve a user's goal. The coordinator executes the plan
through the executor and may revise it based on reflection.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.models.task import Task


class Plan(BaseModel):
    """A plan consisting of an ordered sequence of tasks.

    Attributes:
        goal: The original user goal that this plan was created for.
        tasks: The tasks composing this plan, in execution order.
        created_at: When the plan was generated.
        completed_at: When execution finished, if it has.
        metadata: Arbitrary metadata for extensibility.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str = Field(min_length=1, description="Original user goal.")
    tasks: list[Task] = Field(min_length=1, description="Tasks in execution order.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Creation timestamp.")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp.")
    metadata: dict[str, object] = Field(default_factory=dict, description="Arbitrary metadata.")

    @property
    def task_count(self) -> int:
        """Number of tasks in this plan."""
        return len(self.tasks)

    @property
    def completed_count(self) -> int:
        """Number of tasks with status ``completed``."""
        return sum(1 for t in self.tasks if t.status.value == "completed")

    @property
    def is_complete(self) -> bool:
        """Whether all tasks have reached a terminal status."""
        terminal = {"completed", "failed", "skipped"}
        return all(t.status.value in terminal for t in self.tasks)
