"""Memory service.

Provides relevant context from past interactions to enrich LLM prompts.

The initial implementation uses conversation history as memory. The
interface is designed to be backed by a vector store (semantic search,
reflection, summarization) in a future iteration without changing
callers.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.database.repositories.message_repository import MessageRepository
from app.domain.message import ContentBlock

logger = get_logger(__name__)

_MAX_RECENT_MESSAGES = 20
"""Number of recent messages to include as conversational memory."""


class MemoryService:
    """Retrieves relevant context for a given conversation and user input.

    Usage::

        context = await memory_service.get_relevant_context(
            conversation_id="...",
            user_content=[TextBlock(text="What were we talking about?")],
        )
    """

    def __init__(self, message_repository: MessageRepository) -> None:
        self._message_repo = message_repository

    async def get_relevant_context(
        self,
        conversation_id: str,
        user_content: list[ContentBlock] | None = None,
        *,
        limit: int = _MAX_RECENT_MESSAGES,
    ) -> str | None:
        """Return relevant memory context for the given conversation.

        The current implementation returns the most recent conversation
        history as formatted text. Future versions will incorporate
        semantic search, vector embeddings, and reflection.

        Args:
            conversation_id: The conversation to load memory for.
            user_content: The current user input, used in future
                semantic-search implementations to find related past
                messages.
            limit: Maximum number of recent messages to include.

        Returns:
            A formatted string of relevant context, or ``None`` if no
            relevant memory exists (e.g. first message in a conversation).
        """
        messages = await self._message_repo.list_by_conversation(
            conversation_id,
            limit=limit,
        )

        if not messages:
            return None

        lines: list[str] = []
        for msg in messages:
            role = msg.role.value.upper()
            text_parts: list[str] = []
            for block in msg.content:
                if hasattr(block, "text") and block.text:
                    text_parts.append(block.text)
            text = " ".join(text_parts) if text_parts else "[non-text content]"
            lines.append(f"{role}: {text}")

        return "\n".join(lines)

    async def store_memory(
        self,
        conversation_id: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Store a new memory entry.

        The initial implementation is a no-op — memory is derived from
        conversation history. Future versions will persist entries to a
        vector store for semantic retrieval.

        Args:
            conversation_id: The conversation this memory belongs to.
            content: The memory content (e.g. a summary or extracted fact).
            metadata: Optional structured metadata for filtering.
        """
        logger.debug(
            "memory.store_called",
            conversation_id=conversation_id,
            content_length=len(content),
        )
