"""Message domain models.

Defines the Message aggregate — the core unit of exchange between a user
and an AI assistant. Messages are composed of typed content blocks and
carry role, metadata, and optional tool-call references.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ContentBlockType, MessageRole

__all__ = [
    "ContentBlock",
    "ImageBlock",
    "Message",
    "TextBlock",
    "ToolCallBlock",
    "ToolResultBlock",
]


class ContentBlock(BaseModel):
    """Base content block within a message part.

    A message may contain multiple content blocks (e.g. alternating text
    and image blocks). Subtypes carry additional fields specific to their
    content kind.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ContentBlockType = Field(
        description="Discriminator identifying the concrete block type.",
    )


class TextBlock(ContentBlock):
    """A plain-text content block."""

    type: ContentBlockType = ContentBlockType.TEXT
    text: str = Field(
        min_length=1,
        description="The text content.",
    )


class ImageBlock(ContentBlock):
    """An image content block, supplied as a base64-encoded data URI."""

    type: ContentBlockType = ContentBlockType.IMAGE
    data_uri: str = Field(
        min_length=1,
        description="Base64-encoded data URI of the image.",
    )
    mime_type: str = Field(
        default="image/png",
        description="MIME type of the image (e.g. image/png, image/jpeg).",
    )


class ToolCallBlock(ContentBlock):
    """A content block representing a tool invocation by the assistant."""

    type: ContentBlockType = ContentBlockType.TOOL_CALL
    tool_call_id: str = Field(
        min_length=1,
        description="Unique identifier for this tool call, used to correlate result.",
    )
    tool_name: str = Field(
        min_length=1,
        description="Name of the tool being invoked.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Positional and keyword arguments passed to the tool.",
    )


class ToolResultBlock(ContentBlock):
    """A content block carrying the result of a previous tool call."""

    type: ContentBlockType = ContentBlockType.TOOL_RESULT
    tool_call_id: str = Field(
        min_length=1,
        description="Correlates this result to the original tool call.",
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
        description="Whether the tool execution failed.",
    )


class Message(BaseModel):
    """A single message in a conversation.

    Messages are the immutable, time-ordered records of a conversation.
    Each message belongs to exactly one conversation and has a single
    role. The content is represented as a list of typed content blocks
    for maximum flexibility across providers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="Unique message identifier (UUID).",
    )
    conversation_id: str = Field(
        description="Conversation this message belongs to.",
    )
    role: MessageRole = Field(
        description="Who sent the message.",
    )
    content: list[ContentBlock] = Field(
        min_length=1,
        description="Ordered list of content blocks composing the message.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the message was created.",
    )
    parent_id: str | None = Field(
        default=None,
        description="ID of the message this message replies to, if any.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to this message.",
    )
