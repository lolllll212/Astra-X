"""Conversation domain models.

Defines the Conversation aggregate root and its associated value objects.
A conversation is the top-level organising unit for a sequence of messages
between a user and one or more AI participants.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.domain.enums import ConversationStatus

__all__ = [
    "Conversation",
    "ConversationMetadata",
    "ConversationParticipant",
]


class ConversationParticipant(BaseModel):
    """A participant in a conversation.

    Participants are identified by a stable ID and carry a human-readable
    display label. The participant type (user, agent, system) is inferred
    from context rather than stored as an enum, keeping this model
    open for future participant kinds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        description="Unique participant identifier.",
    )
    display_name: str = Field(
        min_length=1,
        description="Human-readable name shown in UI and logs.",
    )


class ConversationMetadata(BaseModel):
    """Mutable metadata attached to a conversation.

    Stored alongside the conversation but treated as mutable state that
    can be updated without creating a new conversation revision.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        default=None,
        description="User-visible title. Auto-generated from first message if unset.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for organisation and search.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System-level instruction active for this conversation.",
    )
    model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model", "model_id"),
        description="Model identifier used for the last (or default) response.",
    )
    provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("provider", "provider_id"),
        description="Provider identifier used for the last (or default) response.",
    )


class Conversation(BaseModel):
    """Aggregate root for a conversation.

    A Conversation owns a sequence of Messages and carries metadata,
    participant information, and lifecycle state. It is identified by a
    stable UUID and is immutable except for its metadata and status.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Unique conversation identifier (UUID).",
    )
    title: str | None = Field(
        default=None,
        description="Short, user-visible title. Auto-generated if unset.",
    )
    status: ConversationStatus = Field(
        default=ConversationStatus.ACTIVE,
        description="Current lifecycle state of the conversation.",
    )
    participants: list[ConversationParticipant] = Field(
        default_factory=list,
        description="Participants in this conversation. Populated by the service layer on creation.",
    )
    metadata: ConversationMetadata = Field(
        default_factory=ConversationMetadata,
        description="Mutable metadata associated with the conversation.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the conversation was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the conversation was last modified.",
    )
    message_count: int = Field(
        default=0,
        ge=0,
        description="Number of messages in this conversation (denormalised).",
    )
