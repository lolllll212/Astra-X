"""Memory manager for the agent framework.

The :class:`MemoryManager` bridges execution results into the memory
subsystem through a feedback loop:

    Execution
        |
    Reflection
        |
    Importance Scoring
        |
    Store (episodic + semantic + procedural)
        |
    Knowledge Graph

This transforms raw execution traces into structured, queryable knowledge
that the agent can use in future interactions.
"""

from __future__ import annotations

from app.agents.feedback import FeedbackLoop
from app.agents.models.execution import ExecutionResult, ReflectionResult
from app.core.logging import get_logger
from app.domain.message import ContentBlock, TextBlock
from app.services.memory_service import MemoryService

logger = get_logger(__name__)


class MemoryManager:
    """Manages agent-specific memory interactions with feedback loop.

    Usage::

        manager = MemoryManager(memory_service)
        context = await manager.get_context(conversation_id="...", goal="...")
        await manager.store_result(conversation_id="...", result=..., assessment=...)
    """

    def __init__(self, memory_service: MemoryService) -> None:
        self._memory_service = memory_service

    # -- context retrieval ---------------------------------------------------

    async def get_context(
        self,
        conversation_id: str,
        goal: str,
    ) -> str | None:
        """Retrieve relevant memory context for the current goal.

        Delegates to the memory service which first tries semantic
        retrieval and falls back to conversation history.

        Args:
            conversation_id: The active conversation.
            goal: The current goal, used for semantic search.

        Returns:
            A formatted context string, or ``None``.
        """
        user_content: list[ContentBlock] = [TextBlock(text=goal)]
        return await self._memory_service.get_relevant_context(
            conversation_id=conversation_id,
            user_content=user_content,
        )

    # -- feedback loop -------------------------------------------------------

    async def store_result(
        self,
        conversation_id: str,
        result: ExecutionResult,
        assessment: ReflectionResult | None = None,
    ) -> None:
        """Store an execution result through the full memory feedback loop.

        The feedback loop:
        1. Computes importance based on outcome and reflection.
        2. Builds episodic, semantic, and procedural memories.
        3. Extracts knowledge-graph triples.
        4. Persists everything through the core memory manager.

        Args:
            conversation_id: The active conversation.
            result: The execution result to process.
            assessment: Optional reflection assessment of the result.
        """
        if not result.output and not result.error and not result.tool_name:
            logger.debug("memory_manager.skipped_empty", task_id=result.task_id)
            return

        # Step 1: Run the feedback loop.
        feedback = FeedbackLoop.process(result, assessment, conversation_id=conversation_id)
        logger.debug(
            "memory_manager.feedback",
            task_id=result.task_id,
            importance=round(feedback.importance, 3),
            episodic=len(feedback.episodic_memories),
            semantic=len(feedback.semantic_memories),
            procedural=len(feedback.procedural_memories),
            triples=len(feedback.knowledge_triples),
        )

        # Step 2: Persist via the core memory manager.
        core = self._memory_service.core_memory_manager
        if core is not None:
            await FeedbackLoop.persist(core, conversation_id, feedback)
        else:
            # Fallback: flat store via MemoryService (no semantic pipeline).
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
            feedback_summary=feedback.summary,
        )
