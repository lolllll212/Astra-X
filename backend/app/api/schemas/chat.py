"""Request and response schemas for the chat API.

Schemas in this module are pure DTOs. Routes map between API schemas and
domain objects; no business logic lives here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_discriminators import Discriminator  # type: ignore[import-not-found]

from app.api.schemas.conversation import ConversationResponse
from app.domain.enums import ContentBlockType, MessageRole

# ------------------------------------------------------------------ #
# Content block schemas — discriminated union matching domain types
# ------------------------------------------------------------------ #


class TextBlockSchema(BaseModel):
    """A plain-text content block within a message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ContentBlockType = ContentBlockType.TEXT
    text: str = Field(min_length=1, description="The text content.")


class ImageBlockSchema(BaseModel):
    """An image content block (base64 data URI)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ContentBlockType = ContentBlockType.IMAGE
    data_uri: str = Field(min_length=1, description="Base64-encoded data URI.")
    mime_type: str = Field(default="image/png", description="MIME type.")


class ToolCallBlockSchema(BaseModel):
    """A tool invocation block emitted by the assistant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ContentBlockType = ContentBlockType.TOOL_CALL
    tool_call_id: str = Field(min_length=1, description="Unique tool call identifier.")
    tool_name: str = Field(min_length=1, description="Tool being invoked.")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments.")


class ToolResultBlockSchema(BaseModel):
    """The result of a previous tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ContentBlockType = ContentBlockType.TOOL_RESULT
    tool_call_id: str = Field(min_length=1, description="Original tool call identifier.")
    tool_name: str = Field(min_length=1, description="Tool that produced this result.")
    output: str = Field(default="", description="Tool output text.")
    is_error: bool = Field(default=False, description="Whether the tool execution failed.")


ContentBlockSchema = Annotated[
    TextBlockSchema | ImageBlockSchema | ToolCallBlockSchema | ToolResultBlockSchema,
    Discriminator("type"),
]
"""Discriminated union of all supported content block types."""


# ------------------------------------------------------------------ #
# Generation parameters
# ------------------------------------------------------------------ #


class GenerationParamsSchema(BaseModel):
    """Generation parameter overrides for a single request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None


# ------------------------------------------------------------------ #
# Chat request / response
# ------------------------------------------------------------------ #


class ChatRequest(BaseModel):
    """Request body for the non-streaming and streaming chat endpoints.

    Attributes:
        conversation_id: The conversation to continue.
        content: The user's message content blocks.
        model: Optional model override.
        provider: Optional provider override.
        params: Optional generation parameter overrides.
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(
        min_length=1,
        description="Conversation identifier.",
    )
    content: list[ContentBlockSchema] = Field(
        min_length=1,
        description="Message content blocks.",
    )
    model: str | None = Field(default=None, description="Model override.")
    provider: str | None = Field(default=None, description="Provider override.")
    params: GenerationParamsSchema | None = Field(
        default=None,
        description="Generation parameter overrides.",
    )


class ContinueRequest(BaseModel):
    """Request to continue an assistant response (e.g. after a tool result)."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(
        min_length=1,
        description="Conversation identifier.",
    )
    tool_results: list[ToolResultBlockSchema] = Field(
        min_length=1,
        description="Results from tool calls to feed back to the model.",
    )
    model: str | None = Field(default=None, description="Model override.")
    provider: str | None = Field(default=None, description="Provider override.")
    params: GenerationParamsSchema | None = Field(
        default=None,
        description="Generation parameter overrides.",
    )


class UsageResponse(BaseModel):
    """Token usage summary for a generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(ge=0, description="Tokens in the prompt.")
    completion_tokens: int = Field(ge=0, description="Tokens in the completion.")
    total_tokens: int = Field(ge=0, description="Total tokens consumed.")


class MessageResponse(BaseModel):
    """A single message in API responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Message identifier.")
    conversation_id: str = Field(description="Conversation identifier.")
    role: MessageRole = Field(description="Message role.")
    content: list[ContentBlockSchema] = Field(description="Content blocks.")
    created_at: datetime = Field(description="Creation timestamp.")
    parent_id: str | None = Field(default=None, description="Parent message identifier.")


class ChatResponse(BaseModel):
    """Response from a non-streaming chat request.

    Attributes:
        message: The assistant's response message.
        conversation: The updated conversation.
        usage: Token usage if available.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: MessageResponse = Field(description="Assistant response message.")
    conversation: ConversationResponse = Field(description="Updated conversation.")
    usage: UsageResponse | None = Field(default=None, description="Token usage.")


# ------------------------------------------------------------------ #
# Streaming event schemas — each maps to a domain StreamEvent type
# ------------------------------------------------------------------ #


class TextDeltaEvent(BaseModel):
    """Stream event carrying a partial text delta."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str = "text_delta"
    text: str = Field(description="Partial text content.")


class StreamErrorEvent(BaseModel):
    """Stream event indicating an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str = "error"
    message: str = Field(description="Error description.")


class StreamDoneEvent(BaseModel):
    """Final stream event indicating completion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str = "done"
    finish_reason: str = Field(default="stop", description="Why generation stopped.")
    usage: UsageResponse | None = Field(default=None, description="Token usage.")


StreamEventSchema = TextDeltaEvent | StreamErrorEvent | StreamDoneEvent
"""Union of all SSE event types emitted during streaming chat."""
