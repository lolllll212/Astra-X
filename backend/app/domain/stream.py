"""Streaming event domain models.

Defines the structured event types emitted by providers during streaming
responses. These events are consumed by the service layer and serialised
as server-sent events (SSE) for the API.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.domain.enums import StreamEventType
from app.domain.message import ToolCallBlock

__all__ = [
    "ArtifactEvent",
    "PlanEvent",
    "PlannedTaskSchema",
    "ReflectionEvent",
    "StreamCitationEvent",
    "StreamDoneEvent",
    "StreamErrorEvent",
    "StreamEvent",
    "StreamMetadataEvent",
    "StreamStartEvent",
    "StreamUsageEvent",
    "TaskProgressEvent",
    "TextDeltaEvent",
    "ThinkingEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
    "ToolProgressEvent",
    "ToolResultStreamEvent",
]


class StreamStartEvent(BaseModel):
    """Emitted before any other event to signal the stream has begun."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.START


class ThinkingEvent(BaseModel):
    """Emitted when the provider signals it is performing internal reasoning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.THINKING
    thought: str = Field(
        min_length=1,
        description="A fragment of the model's internal reasoning.",
    )


class TextDeltaEvent(BaseModel):
    """Emitted when a new text fragment arrives from the provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.TEXT_DELTA
    delta: str = Field(
        description="The incremental text fragment.",
    )
    index: int = Field(
        default=0,
        ge=0,
        description="Content block index this delta belongs to.",
    )


class ToolCallStartEvent(BaseModel):
    """Emitted when the provider signals the beginning of a tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.TOOL_CALL_START
    tool_call_id: str = Field(
        min_length=1,
        description="Unique identifier for this tool call.",
    )
    tool_name: str = Field(
        min_length=1,
        description="Name of the tool being invoked.",
    )


class ToolCallDeltaEvent(BaseModel):
    """Emitted when incremental tool-call arguments arrive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.TOOL_CALL_DELTA
    tool_call_id: str = Field(
        min_length=1,
        description="Correlates this delta to a previously started tool call.",
    )
    arguments_delta: str = Field(
        description="Incremental JSON fragment of the tool arguments.",
    )


class ToolCallEndEvent(BaseModel):
    """Emitted when a tool call finishes and its arguments are complete."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.TOOL_CALL_END
    tool_call_id: str = Field(
        min_length=1,
        description="Correlates this event to the completed tool call.",
    )
    tool_name: str = Field(
        min_length=1,
        description="Name of the tool that was invoked.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete, parsed tool arguments.",
    )

    def to_block(self) -> ToolCallBlock:
        """Convert this end event into a permanent content block.

        Returns:
            An immutable ToolCallBlock suitable for storage.
        """
        return ToolCallBlock(
            tool_call_id=self.tool_call_id,
            tool_name=self.tool_name,
            arguments=self.arguments,
        )


class ToolProgressEvent(BaseModel):
    """Emitted while a tool is executing (status updates for the frontend)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.TOOL_PROGRESS
    tool_name: str = Field(
        min_length=1,
        description="Name of the tool being executed.",
    )
    status: str = Field(
        description="Execution status: 'running', 'completed', 'failed'.",
    )
    message: str = Field(
        default="",
        description="Human-readable progress message.",
    )


class ToolResultStreamEvent(BaseModel):
    """Emitted when a tool completes execution with its output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.TOOL_RESULT
    tool_name: str = Field(
        min_length=1,
        description="Name of the tool that was executed.",
    )
    tool_call_id: str = Field(
        min_length=1,
        description="Correlates this result to the originating tool call.",
    )
    output: str = Field(
        default="",
        description="Text output produced by the tool.",
    )
    is_error: bool = Field(
        default=False,
        description="Whether the tool execution failed.",
    )
    duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Execution time in milliseconds.",
    )


class StreamCitationEvent(BaseModel):
    """Emitted when the provider includes a source citation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.CITATION
    citation_index: int = Field(
        ge=0,
        description="Index of the cited source.",
    )
    source: str = Field(
        min_length=1,
        description="Source identifier or URL.",
    )
    text: str = Field(
        default="",
        description="Referenced text snippet.",
    )


class StreamUsageEvent(BaseModel):
    """Emitted with token usage after generation completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.USAGE
    prompt_tokens: int = Field(ge=0, description="Tokens in the prompt.")
    completion_tokens: int = Field(ge=0, description="Tokens in the completion.")
    total_tokens: int = Field(ge=0, description="Total tokens consumed.")


class StreamErrorEvent(BaseModel):
    """Emitted when an error occurs during streaming."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.ERROR
    error_code: str = Field(
        description="Stable machine-readable error code.",
    )
    message: str = Field(
        description="Human-readable error description.",
    )


class StreamDoneEvent(BaseModel):
    """Emitted when the stream completes successfully."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.DONE
    finish_reason: str | None = Field(
        default=None,
        description="Provider-specific reason for finishing, if available.",
    )


class StreamMetadataEvent(BaseModel):
    """Emitted once at stream start with operational metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.METADATA
    conversation_id: str = Field(
        description="The conversation this stream belongs to.",
    )
    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="The message ID being streamed (auto-generated if omitted).",
    )
    model: str = Field(
        validation_alias=AliasChoices("model", "model_id"),
        description="Model serving the response.",
    )
    provider: str = Field(
        validation_alias=AliasChoices("provider", "provider_id"),
        description="Provider serving the response.",
    )


class PlannedTaskSchema(BaseModel):
    """Summary of a single task within a plan, for frontend display."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique task identifier.")
    description: str = Field(min_length=1, description="What the task does.")
    status: str = Field(default="pending", description="Task status for display.")
    dependencies: list[str] = Field(default_factory=list, description="Dependency task IDs.")
    tool_name: str | None = Field(default=None, description="Tool to invoke.")


class PlanEvent(BaseModel):
    """Emitted when the planner completes and a plan is ready."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.PLAN
    goal: str = Field(min_length=1, description="The original user goal.")
    tasks: list[PlannedTaskSchema] = Field(description="All planned tasks.")
    iteration: int = Field(default=0, description="Which plan→execute→reflect cycle this belongs to.")


class TaskProgressEvent(BaseModel):
    """Emitted when a task's lifecycle status changes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.TASK_PROGRESS
    task_id: str = Field(description="The task whose status changed.")
    description: str = Field(min_length=1, description="Task description.")
    status: str = Field(description="New status: 'running', 'completed', 'failed', 'skipped'.")
    result: str | None = Field(default=None, description="Output if completed.")
    error: str | None = Field(default=None, description="Error message if failed.")


class ReflectionEvent(BaseModel):
    """Emitted after the reflection step evaluates execution quality."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.REFLECTION
    needs_more_work: bool = Field(description="Whether more tasks are needed.")
    feedback: str | None = Field(default=None, description="Qualitative assessment.")
    reason: str = Field(description="Explanation of the decision.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the result (0-1).")
    iteration: int = Field(default=0, description="Which cycle this reflection belongs to.")


class ArtifactEvent(BaseModel):
    """Emitted when a tool produces a structured artifact alongside text output.

    Artifacts allow tools to return rich data (code blocks, tables, charts,
    file references) that the frontend can render natively.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.ARTIFACT
    task_id: str = Field(description="The task that produced this artifact.")
    label: str = Field(default="", description="Human-readable label for the artifact.")
    artifact_type: str = Field(
        description="Type: 'code', 'table', 'image', 'file', 'chart', 'json', 'markdown'.",
    )
    data: object = Field(description="Structured artifact data.")
    metadata: dict[str, object] = Field(default_factory=dict, description="Arbitrary metadata.")


StreamEvent = (
    StreamStartEvent
    | ThinkingEvent
    | TextDeltaEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | ToolProgressEvent
    | ToolResultStreamEvent
    | StreamCitationEvent
    | StreamUsageEvent
    | StreamErrorEvent
    | StreamDoneEvent
    | StreamMetadataEvent
    | PlanEvent
    | TaskProgressEvent
    | ReflectionEvent
    | ArtifactEvent
)
"""Union of all possible streaming event types."""
