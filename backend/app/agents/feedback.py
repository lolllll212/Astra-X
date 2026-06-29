"""Memory feedback loop — captures reasoning, decisions, and outcomes.

Every completed agent task flows through:

    Execution
        |
    Reflection
        |
    Importance Scoring
        |
    Episodic + Semantic + Procedural memories
        |
    Knowledge Graph triples

This transforms raw execution traces into structured, queryable knowledge
that the agent can use in future interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from app.agents.models.execution import ExecutionResult, ReflectionResult
from app.agents.models.task import TaskStatus
from app.core.logging import get_logger
from app.domain.enums import MemoryScope
from app.memory.manager import MemoryManager as CoreMemoryManager
from app.memory.models.knowledge import KnowledgeTriple
from app.memory.models.memory import MemoryType

logger = get_logger(__name__)


@dataclass
class FeedbackMemory:
    """A single memory produced by the feedback loop."""

    content: str
    memory_type: MemoryType
    importance: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class FeedbackResult:
    """Structured output of processing an execution through the feedback loop."""

    importance: float
    episodic_memories: list[FeedbackMemory] = field(default_factory=list)
    semantic_memories: list[FeedbackMemory] = field(default_factory=list)
    procedural_memories: list[FeedbackMemory] = field(default_factory=list)
    knowledge_triples: list[KnowledgeTriple] = field(default_factory=list)
    summary: str = ""


class FeedbackLoop:
    """Processes execution results into structured, importance-weighted memories.

    Usage::

        feedback = FeedbackLoop()
        result = feedback.process(execution_result, reflection_result)
        await feedback.persist(core_memory_manager, conversation_id, result)
    """

    @staticmethod
    def compute_importance(
        execution: ExecutionResult,
        reflection: ReflectionResult | None = None,
    ) -> float:
        """Score the importance of a result from 0.1 (trivial) to 1.0 (critical).

        Factors:
        - Failed tasks are more important (they teach what not to do).
        - RETRY/ABORT reflections signal learning opportunities.
        - Tool usage indicates actionable knowledge.
        - Errors contain lessons.
        """
        score = 0.5
        if execution.status is TaskStatus.FAILED:
            score += 0.3
        if reflection is not None:
            if reflection.decision.value in ("retry", "abort"):
                score += 0.2
            if reflection.confidence < 0.5:
                score += 0.1
        if execution.tool_name:
            score += 0.1
        if execution.error:
            score += 0.1
        if execution.status is TaskStatus.COMPLETED and not execution.tool_name and not execution.error:
            score -= 0.2
        return max(0.1, min(1.0, score))

    @staticmethod
    def build_episodic_memory(
        execution: ExecutionResult,
        importance: float,
    ) -> FeedbackMemory:
        """Record what happened — the raw event."""
        parts = [f"Task {execution.task_id}"]
        if execution.tool_name:
            parts.append(f"used {execution.tool_name}")
        if execution.status is TaskStatus.FAILED:
            parts.append(f"failed: {execution.error or 'unknown error'}")
        else:
            preview = (execution.output or "")[:200]
            parts.append(f"completed: {preview}")
        return FeedbackMemory(
            content="; ".join(parts),
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            metadata={
                "task_id": execution.task_id,
                "tool_name": execution.tool_name or "",
                "status": execution.status.value,
            },
        )

    @staticmethod
    def build_semantic_memory(
        execution: ExecutionResult,
        reflection: ReflectionResult | None,
        importance: float,
    ) -> FeedbackMemory | None:
        """Extract what was learned — a generalizable insight."""
        if not execution.tool_name and not execution.error:
            return None
        if execution.status is TaskStatus.FAILED:
            content = f"Tool {execution.tool_name!r} can fail with error: {execution.error}"
        elif reflection is not None and reflection.decision.value == "retry":
            content = f"Using {execution.tool_name or 'LLM'} may require multiple attempts for reliability"
        elif execution.tool_name:
            output_preview = (execution.output or "")[:100]
            content = f"Tool {execution.tool_name!r} can produce output like: {output_preview}"
        else:
            return None
        return FeedbackMemory(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            importance=importance * 0.9,
            metadata={"source": "feedback_loop", "task_id": execution.task_id},
        )

    @staticmethod
    def build_procedural_memory(
        execution: ExecutionResult,
        reflection: ReflectionResult | None,
        importance: float,
    ) -> FeedbackMemory | None:
        """Extract a recommendation — when to use (or avoid) a certain approach."""
        if not execution.tool_name:
            return None
        if execution.status is TaskStatus.FAILED:
            content = (
                f"Avoid using {execution.tool_name!r} when {execution.error or 'conditions are uncertain'}"
            )
        elif reflection is not None and reflection.decision.value in ("retry", "abort"):
            content = (
                f"Consider retrying with a different approach when "
                f"{execution.tool_name!r} produces unsatisfactory results"
            )
        else:
            content = f"Use {execution.tool_name!r} for tasks matching the description: {execution.task_id}"
        return FeedbackMemory(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            importance=importance * 0.8,
            metadata={
                "source": "feedback_loop",
                "tool_name": execution.tool_name,
                "decision": reflection.decision.value if reflection else "accept",
            },
        )

    @staticmethod
    def build_knowledge_triples(
        execution: ExecutionResult,
        reflection: ReflectionResult | None,
        conversation_id: str,
    ) -> list[KnowledgeTriple]:
        """Extract entity-relationship triples from the execution."""
        triples: list[KnowledgeTriple] = []
        base_source = f"task:{execution.task_id}"
        if execution.tool_name:
            triples.append(KnowledgeTriple(
                id=str(uuid4()),
                subject="Agent",
                predicate="used",
                object=execution.tool_name,
                confidence=0.9,
                source=base_source,
                conversation_id=conversation_id,
            ))
            status_label = "failed" if execution.status is TaskStatus.FAILED else "completed"
            triples.append(KnowledgeTriple(
                id=str(uuid4()),
                subject=execution.tool_name.capitalize(),
                predicate="status",
                object=status_label,
                confidence=0.8,
                source=base_source,
                conversation_id=conversation_id,
            ))
        if execution.error:
            triples.append(KnowledgeTriple(
                id=str(uuid4()),
                subject="Task",
                predicate="failed_with",
                object=execution.error[:100],
                confidence=0.7,
                source=base_source,
                conversation_id=conversation_id,
            ))
        if reflection is not None:
            triples.append(KnowledgeTriple(
                id=str(uuid4()),
                subject="Reflection",
                predicate="decision",
                object=reflection.decision.value,
                confidence=reflection.confidence,
                source=base_source,
                conversation_id=conversation_id,
            ))
        return triples

    @staticmethod
    def process(
        execution: ExecutionResult,
        reflection: ReflectionResult | None = None,
        conversation_id: str = "",
    ) -> FeedbackResult:
        """Run the full feedback loop for a single execution result.

        Args:
            execution: The task execution result.
            reflection: Optional reflection assessment.
            conversation_id: Active conversation for knowledge graph triples.

        Returns:
            A structured FeedbackResult with importance, memories, and triples.
        """
        importance = FeedbackLoop.compute_importance(execution, reflection)

        episodic = FeedbackLoop.build_episodic_memory(execution, importance)
        semantic = FeedbackLoop.build_semantic_memory(execution, reflection, importance)
        procedural = FeedbackLoop.build_procedural_memory(execution, reflection, importance)
        triples = FeedbackLoop.build_knowledge_triples(execution, reflection, conversation_id)

        parts = [f"Task {execution.task_id} [{execution.status.value}]"]
        if execution.tool_name:
            parts.append(f"tool={execution.tool_name}")
        if reflection:
            parts.append(f"decision={reflection.decision.value}")
        summary = " | ".join(parts)

        return FeedbackResult(
            importance=importance,
            episodic_memories=[episodic],
            semantic_memories=[semantic] if semantic else [],
            procedural_memories=[procedural] if procedural else [],
            knowledge_triples=triples,
            summary=summary,
        )

    @staticmethod
    async def persist(
        core_memory_manager: CoreMemoryManager,
        conversation_id: str,
        feedback: FeedbackResult,
    ) -> None:
        """Persist the feedback result to the core memory subsystem.

        Args:
            core_memory_manager: The core semantic memory manager.
            conversation_id: Active conversation ID.
            feedback: The feedback result to persist.
        """
        stored_count = 0
        for mem in feedback.episodic_memories:
            await core_memory_manager.store(
                content=mem.content,
                memory_type=mem.memory_type,
                scope=MemoryScope.CONVERSATION,
                conversation_id=conversation_id,
                importance=mem.importance,
                metadata=mem.metadata,
            )
            stored_count += 1
        for mem in feedback.semantic_memories:
            await core_memory_manager.store(
                content=mem.content,
                memory_type=mem.memory_type,
                scope=MemoryScope.CONVERSATION,
                conversation_id=conversation_id,
                importance=mem.importance,
                metadata=mem.metadata,
            )
            stored_count += 1
        for mem in feedback.procedural_memories:
            await core_memory_manager.store(
                content=mem.content,
                memory_type=mem.memory_type,
                scope=MemoryScope.GLOBAL,
                conversation_id=conversation_id,
                importance=mem.importance,
                metadata=mem.metadata,
            )
            stored_count += 1
        if core_memory_manager.knowledge_graph:
            core_memory_manager.knowledge_graph.add_triples(feedback.knowledge_triples)
        logger.debug(
            "feedback.persisted",
            conversation_id=conversation_id,
            memories_stored=stored_count,
            triples=len(feedback.knowledge_triples),
            importance=round(feedback.importance, 3),
        )
