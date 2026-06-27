"""Memory manager for the agent framework.

The :class:`MemoryManager` bridges the existing
:class:`app.services.memory_service.MemoryService` into the agent
workflow. It decides what information from an interaction should be
remembered, retrieves relevant context, and stores new memories.

This initial implementation delegates directly to the existing memory
service. Future versions may add agent-specific logic such as
extracting key facts, generating summaries, or pruning old memories.
"""

from __future__ import annotations

from app.agents.models.execution import ExecutionResult
from app.core.logging import get_logger
from app.domain.message import ContentBlock, TextBlock
from app.services.memory_service import MemoryService

logger = get_logger(__name__)


class MemoryManager:
    """Manages agent-specific memory interactions.

    Usage::

        manager = MemoryManager(memory_service)
        context = await manager.get_context(
            conversation_id="...",
            goal="...",
        )
        await manager.store_result(
            conversation_id="...",
            result=execution_result,
        )
    """

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    async def get_context(
        self,
        conversation_id: str,
        goal: str,
    ) -> str | None:
        """Retrieve relevant memory context for the current goal.

        Delegates to the memory service which, in the initial
        implementation, returns recent conversation history.

        Args:
            conversation_id: The active conversation.
            goal: The current goal, used in future semantic-search
                implementations.

        Returns:
            A formatted context string, or ``None``.
        """
        user_content: list[ContentBlock] = [TextBlock(text=goal)]
        return await self._memory_service.get_relevant_context(
            conversation_id=conversation_id,
            user_content=user_content,
        )

    async def store_result(
        self,
        conversation_id: str,
        result: ExecutionResult,
    ) -> None:
        """Store an execution result as a memory entry.

        Args:
            conversation_id: The active conversation.
            result: The execution result to remember.
        """
        if not result.output and not result.error:
            return

        content_parts: list[str] = []
        if result.output:
            content_parts.append(f"Output: {result.output}")
        if result.error:
            content_parts.append(f"Error: {result.error}")

        content = f"Task {result.task_id}: {'; '.join(content_parts)}"

        await self._memory_service.store_memory(
            conversation_id=conversation_id,
            content=content,
            metadata={
                "task_id": result.task_id,
                "tool_name": result.tool_name,
                "status": result.status.value,
                "source": "agent_executor",
            },
        )
        logger.debug(
            "memory_manager.stored",
            conversation_id=conversation_id,
            task_id=result.task_id,
        )
