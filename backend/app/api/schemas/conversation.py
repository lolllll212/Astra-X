"""Request and response schemas for the conversation CRUD API.

Conversation schemas define the API contract; they are converted to and
from domain objects by the route handlers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas.common import PaginatedResponse
from app.domain.enums import ConversationStatus


class ConversationCreate(BaseModel):
    """Request body for creating a new conversation.

    Attributes:
        title: Optional human-readable title.
        system_prompt: Optional system-level instruction.
        model: Optional default model for this conversation.
        provider: Optional default provider for this conversation.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, description="Conversation title.")
    system_prompt: str | None = Field(
        default=None,
        description="System-level instruction.",
    )
    model: str | None = Field(default=None, description="Default model.")
    provider: str | None = Field(default=None, description="Default provider.")


class ConversationUpdate(BaseModel):
    """Request body for updating an existing conversation.

    All fields are optional — only provided fields are updated.

    Attributes:
        title: New conversation title.
        system_prompt: New system-level instruction.
        model: New default model.
        provider: New default provider.
        status: New conversation status.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, description="Conversation title.")
    system_prompt: str | None = Field(default=None, description="System-level instruction.")
    model: str | None = Field(default=None, description="Default model.")
    provider: str | None = Field(default=None, description="Default provider.")
    status: ConversationStatus | None = Field(default=None, description="Conversation status.")


class ConversationResponse(BaseModel):
    """Complete conversation representation returned by the API.

    Attributes:
        id: Unique conversation identifier.
        title: Human-readable title, if set.
        status: Current lifecycle status.
        participants: Participants in this conversation.
        metadata: Conversation metadata (system prompt, model, provider).
        message_count: Number of messages in the conversation.
        created_at: When the conversation was created.
        updated_at: When the conversation was last modified.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Unique conversation identifier.")
    title: str | None = Field(default=None, description="Conversation title.")
    status: ConversationStatus = Field(description="Current lifecycle status.")
    participants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation participants.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation metadata.",
    )
    message_count: int = Field(ge=0, description="Number of messages.")
    created_at: datetime = Field(description="Creation timestamp.")
    updated_at: datetime = Field(description="Last modification timestamp.")


class ConversationListResponse(PaginatedResponse):
    """Paginated list of conversations.

    Attributes:
        items: The page of conversation responses.
    """

    items: list[ConversationResponse] = Field(description="Page of conversations.")  # type: ignore[assignment]
