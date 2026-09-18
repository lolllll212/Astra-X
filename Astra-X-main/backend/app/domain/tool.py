"""Tool domain models.

Defines the entities and value objects for the tool system — from
tool specification (how a tool is described) through invocation (a
specific call) to result (the output of that call).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ToolType

__all__ = [
    "ToolCall",
    "ToolParameter",
    "ToolResult",
    "ToolSpec",
]


class ToolParameter(BaseModel):
    """Describes a single parameter accepted by a tool function."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        min_length=1,
        description="Parameter name, as expected by the tool implementation.",
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this parameter does.",
    )
    type_: str = Field(
        alias="type",
        default="string",
        description="JSON Schema type (string, integer, boolean, array, object, etc.).",
    )
    required: bool = Field(
        default=False,
        description="Whether the caller must supply this parameter.",
    )
    enum: list[str] | None = Field(
        default=None,
        description="If set, the parameter must be one of these values.",
    )
    default: Any = Field(
        default=None,
        description="Default value when the parameter is omitted.",
    )


class ToolSpec(BaseModel):
    """Specification of an executable tool.

    A ToolSpec describes *what* a tool does and *how* to call it, in a
    provider-agnostic format. Provider adapters convert this spec into
    the provider-specific tool format (e.g. OpenAI function definition).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        min_length=1,
        description="Unique tool name used to invoke the tool.",
    )
    description: str = Field(
        default="",
        description="Human-readable description shown to the model and the user.",
    )
    tool_type: ToolType = Field(
        default=ToolType.FUNCTION,
        description="Category of tool this spec describes.",
    )
    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Ordered list of accepted parameters.",
    )


class ToolCall(BaseModel):
    """A concrete invocation of a tool.

    Created when the model decides to call a tool during generation.
    Carries all the context needed to execute the tool and correlate the
    result back to the conversation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="Unique identifier for this tool call invocation.",
    )
    conversation_id: str = Field(
        description="Conversation that triggered this tool call.",
    )
    message_id: str = Field(
        description="Message that contains this tool call.",
    )
    tool_name: str = Field(
        min_length=1,
        description="Name of the tool to execute.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="The parsed arguments passed to the tool.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the tool call was created.",
    )


class ToolResult(BaseModel):
    """The output of a single tool execution.

    Linked to the originating :class:`ToolCall` by ``tool_call_id``.
    The result is either a success with ``output`` or a failure with
    ``is_error`` set to ``True``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_id: str = Field(
        description="Correlates this result to the originating tool call.",
    )
    tool_name: str = Field(
        min_length=1,
        description="Name of the tool that produced this result.",
    )
    output: str = Field(
        default="",
        description="Text output produced by the tool.",
    )
    is_error: bool = Field(
        default=False,
        description="Whether the tool execution raised an error.",
    )
    duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Wall-clock time the tool took to execute, in milliseconds.",
    )
