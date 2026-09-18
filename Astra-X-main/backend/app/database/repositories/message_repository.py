"""Message repository."""

from __future__ import annotations

from sqlalchemy import select

from app.database.converters.message_mapper import (
    message_from_model,
    message_to_model,
)
from app.database.models.message import MessageModel
from app.database.repositories.base import BaseRepository
from app.domain.message import Message


class MessageRepository(BaseRepository[Message, MessageModel]):
    """Repository for :class:`Message` entities."""

    @property
    def _model_cls(self) -> type[MessageModel]:
        return MessageModel

    def _to_domain(self, model: MessageModel) -> Message:
        return message_from_model(model)

    def _to_model(self, domain: Message) -> MessageModel:
        return message_to_model(domain)

    async def list_by_conversation(
        self,
        conversation_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Message]:
        """Retrieve messages for a conversation, ordered by creation time.

        Args:
            conversation_id: The conversation to fetch messages for.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip (for pagination).

        Returns:
            An ordered list of messages (oldest first).
        """
        stmt = (
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
            .order_by(MessageModel.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def count_by_conversation(self, conversation_id: str) -> int:
        """Count messages in a conversation.

        Args:
            conversation_id: The conversation to count messages for.

        Returns:
            The total number of messages.
        """
        return await self.count(conversation_id=conversation_id)
