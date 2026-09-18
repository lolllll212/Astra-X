"""Conversation domain-to-ORM mapper."""

from __future__ import annotations

from typing import Any

from app.database.models.conversation import ConversationModel
from app.domain.conversation import Conversation, ConversationMetadata, ConversationParticipant

__all__ = [
    "conversation_from_model",
    "conversation_to_model",
]


def conversation_to_model(domain: Conversation) -> ConversationModel:
    participants_data: list[dict[str, Any]] = [
        {"id": p.id, "display_name": p.display_name}
        for p in domain.participants
    ]
    metadata_data: dict[str, Any] = domain.metadata.model_dump(exclude_none=True) if domain.metadata else {}

    return ConversationModel(
        id=domain.id,
        title=domain.title,
        status=domain.status.value,
        participants=participants_data,
        metadata_=metadata_data,
        created_at=domain.created_at,
        updated_at=domain.updated_at,
        message_count=domain.message_count,
    )


def conversation_from_model(model: ConversationModel) -> Conversation:
    participants: list[ConversationParticipant] = [
        ConversationParticipant(id=p["id"], display_name=p["display_name"])
        for p in (model.participants or [])
    ]
    metadata_obj = ConversationMetadata(**(model.metadata_ or {}))

    return Conversation(
        id=model.id,
        title=model.title,
        status=model.status,
        participants=participants,
        metadata=metadata_obj,
        created_at=model.created_at,
        updated_at=model.updated_at,
        message_count=model.message_count,
    )
