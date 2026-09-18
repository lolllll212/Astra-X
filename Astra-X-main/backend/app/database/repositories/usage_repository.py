"""Usage repository."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import select

from app.database.models.usage import UsageModel
from app.domain.usage import Usage


class UsageRepository:
    """Repository for :class:`Usage` records.

    Usage is a value object embedded in a Message. This repository
    provides message-scoped read/write operations rather than the
    generic CRUD, because Usage has no independent identity in the
    domain — the database identity (``UsageModel.id``) is an internal
    detail.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    async def add_for_message(self, message_id: str, usage: Usage) -> Usage:
        """Persist a usage record linked to a message.

        Args:
            message_id: The message this usage record belongs to.
            usage: The usage value object.

        Returns:
            The same usage value object (unchanged, since it is
            immutable and has no domain-side identity).
        """
        model = UsageModel(
            id=str(uuid4()),
            message_id=message_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            prompt_tokens_details=usage.prompt_tokens_details,
            completion_tokens_details=usage.completion_tokens_details,
            cost_usd=usage.cost_usd,
        )
        self._session.add(model)
        await self._session.flush()
        return usage

    async def get_by_message(self, message_id: str) -> Usage | None:
        """Retrieve usage for a given message.

        Args:
            message_id: The message identifier.

        Returns:
            The usage value object if found, otherwise ``None``.
        """
        stmt = select(UsageModel).where(UsageModel.message_id == message_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return Usage(
            prompt_tokens=model.prompt_tokens,
            completion_tokens=model.completion_tokens,
            total_tokens=model.total_tokens,
            prompt_tokens_details=model.prompt_tokens_details,
            completion_tokens_details=model.completion_tokens_details,
            cost_usd=model.cost_usd,
        )

    async def delete_by_message(self, message_id: str) -> bool:
        """Delete usage records for a given message.

        Args:
            message_id: The message identifier.

        Returns:
            ``True`` if at least one record was deleted.
        """
        stmt = select(UsageModel).where(UsageModel.message_id == message_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True
