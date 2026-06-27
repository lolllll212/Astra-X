"""Execution result models for the agent framework.

These models capture the outcome of executing a single task
(:class:`ExecutionResult`) and the outcome of reflecting on a result
(:class:`ReflectionResult`).
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.agents.models.task import TaskStatus


class ExecutionResult(BaseModel):
    """The outcome of executing a single task.

    Attributes:
        task_id: The task that was executed.
        status: Whether execution succeeded, failed, etc.
        output: Text output produced by execution.
        error: Error message if execution failed.
        tool_name: The tool that was invoked, if any.
        tool_call_id: Correlation ID for the tool call.
        started_at: When execution began.
        completed_at: When execution finished.
        metadata: Arbitrary execution metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(description="Executed task identifier.")
    status: TaskStatus = Field(description="Outcome status.")
    output: str | None = Field(default=None, description="Execution output.")
    error: str | None = Field(default=None, description="Error message.")
    tool_name: str | None = Field(default=None, description="Tool invoked.")
    tool_call_id: str | None = Field(default=None, description="Tool call correlation ID.")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Start timestamp.")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Finish timestamp.")
    metadata: dict[str, object] = Field(default_factory=dict, description="Arbitrary metadata.")


class ReflectionResult(BaseModel):
    """The outcome of reflecting on an execution result.

    Reflection decides whether more work is needed, provides feedback
    on the quality of the result, and optionally suggests follow-up
    tasks.

    Attributes:
        needs_more_work: Whether additional tasks should be executed.
        feedback: Qualitative feedback on the result.
        reason: Explanation of why more work is (or is not) needed.
        next_tasks: Suggested follow-up tasks, if any.
        confidence: Self-assessed confidence in the result (0.0-1.0).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    needs_more_work: bool = Field(description="Whether additional tasks are needed.")
    feedback: str | None = Field(default=None, description="Qualitative feedback.")
    reason: str = Field(description="Explanation of the decision.")
    next_tasks: list[str] = Field(default_factory=list, description="Suggested follow-up tasks.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the result.")
