"""Conversation repository."""

from __future__ import annotations

from app.database.converters.conversation_mapper import (
    conversation_from_model,
    conversation_to_model,
)
from app.database.models.conversation import ConversationModel
from app.database.repositories.base import BaseRepository
from app.domain.conversation import Conversation


class ConversationRepository(BaseRepository[Conversation, ConversationModel]):
    """Repository for :class:`Conversation` entities."""

    @property
    def _model_cls(self) -> type[ConversationModel]:
        return ConversationModel

    def _to_domain(self, model: ConversationModel) -> Conversation:
        return conversation_from_model(model)

    def _to_model(self, domain: Conversation) -> ConversationModel:
        return conversation_to_model(domain)
