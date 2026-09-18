"""Memory service.

Provides relevant context from past interactions to enrich LLM prompts,
backed by the semantic memory pipeline (vector embeddings, retrieval,
and consolidation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.logging import get_logger
from app.database.repositories.message_repository import MessageRepository
from app.domain.message import ContentBlock

if TYPE_CHECKING:
    from app.memory.manager import MemoryManager as CoreMemoryManager

logger = get_logger(__name__)

_MAX_RECENT_MESSAGES = 20
"""Number of recent messages to include as conversational memory."""


class MemoryService:
    """Retrieves relevant context for a given conversation and user input.

    Uses the core semantic memory manager for retrieval and storage,
    falling back to recent conversation history when no semantic
    matches exist.

    Usage::

        context = await memory_service.get_relevant_context(
            conversation_id="...",
            user_content=[TextBlock(text="What were we talking about?")],
        )
    """

    def __init__(
        self,
        message_repository: MessageRepository,
        core_memory_manager: CoreMemoryManager | None = None,
    ) -> None:
        self._message_repo = message_repository
        self._core_memory_manager = core_memory_manager

    @property
    def core_memory_manager(self) -> CoreMemoryManager | None:
        """Expose the core semantic memory manager, if available."""
        return self._core_memory_manager

    async def get_relevant_context(
        self,
        conversation_id: str,
        user_content: list[ContentBlock] | None = None,
        *,
        limit: int = _MAX_RECENT_MESSAGES,
    ) -> str | None:
        """Return relevant memory context for the given conversation.

        First attempts semantic retrieval via the core memory manager.
        If that yields no results, falls back to recent conversation
        history.

        Args:
            conversation_id: The conversation to load memory for.
            user_content: The current user input, used for semantic
                search queries.
            limit: Maximum number of recent messages to include as
                fallback.

        Returns:
            A formatted string of relevant context, or ``None`` if no
            relevant memory exists.
        """
        # Try semantic retrieval first
        if self._core_memory_manager is not None and user_content:
            query_text = " ".join(
                b.text for b in user_content if hasattr(b, "text") and b.text
            )
            if query_text:
                try:
                    results = await self._core_memory_manager.retrieve(
                        query=query_text,
                        conversation_id=conversation_id,
                        limit=5,
                    )
                    if results:
                        lines: list[str] = []
                        for r in results:
                            lines.append(f"[memory score={r.score:.3f}] {r.memory.content}")
                        return "\n".join(lines)
                except Exception:
                    logger.warning("memory.semantic_retrieval_failed", exc_info=True)

        # Fall back to recent conversation history
        messages = await self._message_repo.list_by_conversation(
            conversation_id,
            limit=limit,
        )

        if not messages:
            return None

        lines = []
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
        """Store a new memory entry via the core memory manager.

        Args:
            conversation_id: The conversation this memory belongs to.
            content: The memory content (e.g. a summary or extracted fact).
            metadata: Optional structured metadata for filtering.
        """
        if self._core_memory_manager is not None:
            try:
                await self._core_memory_manager.store(
                    content=content,
                    conversation_id=conversation_id,
                    metadata=metadata or {},
                )
                logger.debug(
                    "memory.stored",
                    conversation_id=conversation_id,
                    content_length=len(content),
                )
                return
            except Exception:
                logger.warning("memory.store_failed", exc_info=True)

        logger.debug(
            "memory.store_skipped",
            conversation_id=conversation_id,
            reason="no core memory manager available",
        )
