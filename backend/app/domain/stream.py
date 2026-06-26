"""Streaming event domain models.

Defines the structured event types emitted by providers during streaming
responses. These events are consumed by the service layer and serialised
as server-sent events (SSE) for the API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import StreamEventType
from app.domain.message import ToolCallBlock

__all__ = [
    "StreamDoneEvent",
    "StreamErrorEvent",
    "StreamEvent",
    "StreamMetadataEvent",
    "TextDeltaEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
]


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
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage summary for the completed generation.",
    )


class StreamMetadataEvent(BaseModel):
    """Emitted once at stream start with operational metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: StreamEventType = StreamEventType.METADATA
    conversation_id: str = Field(
        description="The conversation this stream belongs to.",
    )
    message_id: str = Field(
        description="The message ID being streamed.",
    )
    model_id: str = Field(
        description="Model serving the response.",
    )
    provider_id: str = Field(
        description="Provider serving the response.",
    )


StreamEvent = (
    TextDeltaEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | StreamErrorEvent
    | StreamDoneEvent
    | StreamMetadataEvent
)
"""Union of all possible streaming event types."""
