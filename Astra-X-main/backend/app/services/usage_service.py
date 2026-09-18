"""Usage tracking service.

Provides aggregated token usage statistics and cost tracking across
messages and conversations.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.database.repositories.usage_repository import UsageRepository
from app.domain.usage import Usage

logger = get_logger(__name__)


class UsageService:
    """Tracks and reports token usage."""

    def __init__(self, repository: UsageRepository) -> None:
        self._repo = repository

    async def get_for_message(self, message_id: str) -> Usage | None:
        """Retrieve usage for a specific message.

        Args:
            message_id: The message identifier.

        Returns:
            Usage record if available.
        """
        return await self._repo.get_by_message(message_id)

    async def record(
        self,
        message_id: str,
        usage: Usage,
    ) -> Usage:
        """Record token usage for a message.

        Args:
            message_id: The message that was generated.
            usage: The usage value object.

        Returns:
            The recorded usage.
        """
        result = await self._repo.add_for_message(message_id, usage)
        logger.info(
            "usage.recorded",
            message_id=message_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return result
