"""Conversation management service.

Handles all conversation lifecycle operations — create, rename, delete,
archive, and search. This service contains no AI logic; it is a pure
domain-coordination layer over the conversation repository.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.core.exceptions import ResourceNotFoundError
from app.core.logging import get_logger
from app.database.repositories.conversation_repository import ConversationRepository
from app.domain.conversation import (
    Conversation,
    ConversationMetadata,
    ConversationParticipant,
)
from app.domain.enums import ConversationStatus

logger = get_logger(__name__)


class ConversationService:
    """Manages conversation lifecycle.

    Usage::

        service = ConversationService(repo)
        conv = await service.create("New chat")
        await service.rename(conv.id, "Updated title")
    """

    def __init__(self, repository: ConversationRepository) -> None:
        self._repo = repository

    async def create(
        self,
        title: str | None = None,
        *,
        participants: list[ConversationParticipant] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> Conversation:
        """Create a new conversation.

        Args:
            title: Optional human-readable title.
            participants: Optional list of participants. Defaults to
                empty — the service layer adds participants as needed.
            system_prompt: Optional system-level instruction.
            model_id: Optional default model for this conversation.
            provider_id: Optional default provider for this conversation.

        Returns:
            The newly created conversation.
        """
        now = datetime.now(UTC)
        conversation = Conversation(
            id=str(uuid4()),
            title=title,
            status=ConversationStatus.ACTIVE,
            participants=participants or [],
            metadata=ConversationMetadata(
                system_prompt=system_prompt,
                model=model,
                provider=provider,
            ),
            created_at=now,
            updated_at=now,
            message_count=0,
        )
        result = await self._repo.add(conversation)
        logger.info("conversation.created", conversation_id=result.id)
        return result

    async def get(self, conversation_id: str) -> Conversation:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: The conversation identifier.

        Returns:
            The conversation.

        Raises:
            ResourceNotFoundError: If the conversation does not exist.
        """
        conversation = await self._repo.get(conversation_id)
        if conversation is None:
            raise ResourceNotFoundError(
                message=f"Conversation '{conversation_id}' not found.",
            )
        return conversation

    async def rename(self, conversation_id: str, title: str) -> Conversation:
        """Rename a conversation.

        Args:
            conversation_id: The conversation identifier.
            title: The new title.

        Returns:
            The updated conversation.
        """
        conversation = await self.get(conversation_id)
        conversation.title = title
        conversation.updated_at = datetime.now(UTC)
        result = await self._repo.update(conversation)
        logger.info("conversation.renamed", conversation_id=conversation_id)
        return result

    async def archive(self, conversation_id: str) -> Conversation:
        """Move a conversation to archived status.

        Args:
            conversation_id: The conversation identifier.

        Returns:
            The updated conversation.
        """
        conversation = await self.get(conversation_id)
        conversation.status = ConversationStatus.ARCHIVED
        conversation.updated_at = datetime.now(UTC)
        result = await self._repo.update(conversation)
        logger.info("conversation.archived", conversation_id=conversation_id)
        return result

    async def delete(self, conversation_id: str) -> None:
        """Permanently delete a conversation.

        Args:
            conversation_id: The conversation identifier.

        Raises:
            ResourceNotFoundError: If the conversation does not exist.
        """
        deleted = await self._repo.delete(conversation_id)
        if not deleted:
            raise ResourceNotFoundError(
                message=f"Conversation '{conversation_id}' not found.",
            )
        logger.info("conversation.deleted", conversation_id=conversation_id)

    async def list_active(self) -> list[Conversation]:
        """List all active conversations, ordered by most recently updated.

        Returns:
            A list of active conversations.
        """
        return await self._repo.list_all(status=ConversationStatus.ACTIVE.value)

    async def update_metadata(
        self,
        conversation_id: str,
        **fields: Any,
    ) -> Conversation:
        """Update specific metadata fields on a conversation.

        Args:
            conversation_id: The conversation identifier.
            **fields: Metadata fields to update (e.g. ``system_prompt=...``).

        Returns:
            The updated conversation.
        """
        conversation = await self.get(conversation_id)
        metadata = conversation.metadata
        for key, value in fields.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        conversation.metadata = metadata
        conversation.updated_at = datetime.now(UTC)
        result = await self._repo.update(conversation)
        logger.info(
            "conversation.metadata_updated",
            conversation_id=conversation_id,
            fields=list(fields),
        )
        return result
