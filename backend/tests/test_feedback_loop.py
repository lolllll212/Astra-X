"""Tests for the memory feedback loop."""
from __future__ import annotations

from uuid import uuid4

from app.agents.feedback import FeedbackLoop, FeedbackResult
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.task import TaskStatus
from app.memory.embedder import Embedder
from app.memory.manager import MemoryManager
from app.memory.models.memory import MemoryType
from app.memory.vector.sqlite_vector import SQLiteVectorStore


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self._dimensions = 4
        self._model = "test"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def make_result(
    status: TaskStatus = TaskStatus.COMPLETED,
    tool: str | None = None,
    output: str | None = "done",
    error: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        task_id=str(uuid4()),
        status=status,
        output=output,
        error=error,
        tool_name=tool,
    )


def make_reflection(decision: str = "accept") -> ReflectionResult:
    return ReflectionResult(
        decision=ReflectionDecision(decision),
        reason="test",
        confidence=0.8,
    )


class TestImportance:
    def test_baseline(self) -> None:
        r = make_result()
        imp = FeedbackLoop.compute_importance(r)
        assert imp == 0.3  # 0.5 base - 0.2 for no tool, no error, completed

    def test_failed_task_higher(self) -> None:
        r = make_result(status=TaskStatus.FAILED, error="timeout")
        imp = FeedbackLoop.compute_importance(r)
        assert imp == 0.9  # 0.5 + 0.3 (failed) + 0.1 (error)

    def test_tool_usage_boost(self) -> None:
        r = make_result(tool="calculator")
        imp = FeedbackLoop.compute_importance(r)
        assert imp == 0.6  # 0.5 + 0.1 (tool) - 0.2 (no error, no reflection)

    def test_failed_with_reflection_retry(self) -> None:
        r = make_result(status=TaskStatus.FAILED, tool="web_search", error="timeout")
        ref = make_reflection("retry")
        imp = FeedbackLoop.compute_importance(r, ref)
        assert imp == 1.0  # 0.5 + 0.3 + 0.2 + 0.1 + 0.1 = 1.2 capped to 1.0


class TestMemoryBuilding:
    def test_episodic_completed(self) -> None:
        r = make_result(tool="calculator", output="42")
        imp = 0.6
        mem = FeedbackLoop.build_episodic_memory(r, imp)
        assert mem.memory_type is MemoryType.EPISODIC
        assert mem.importance == 0.6
        assert "calculator" in mem.content

    def test_episodic_failed(self) -> None:
        r = make_result(status=TaskStatus.FAILED, tool="web_search", error="connection refused")
        mem = FeedbackLoop.build_episodic_memory(r, 0.8)
        assert "failed" in mem.content
        assert "connection refused" in mem.content

    def test_semantic_for_tool_usage(self) -> None:
        r = make_result(tool="calculator", output="42")
        mem = FeedbackLoop.build_semantic_memory(r, None, 0.6)
        assert mem is not None
        assert mem.memory_type is MemoryType.SEMANTIC
        assert "calculator" in mem.content

    def test_semantic_skipped_for_chat(self) -> None:
        r = make_result(tool=None, output="Hello!")
        mem = FeedbackLoop.build_semantic_memory(r, None, 0.4)
        assert mem is None

    def test_procedural_for_successful_tool(self) -> None:
        r = make_result(tool="datetime")
        mem = FeedbackLoop.build_procedural_memory(r, None, 0.6)
        assert mem is not None
        assert mem.memory_type is MemoryType.PROCEDURAL
        assert "Use" in mem.content

    def test_procedural_skipped_for_no_tool(self) -> None:
        r = make_result()
        mem = FeedbackLoop.build_procedural_memory(r, None, 0.4)
        assert mem is None

    def test_procedural_failed_avoidance(self) -> None:
        r = make_result(status=TaskStatus.FAILED, tool="web_search", error="timeout")
        mem = FeedbackLoop.build_procedural_memory(r, None, 0.8)
        assert mem is not None
        assert "Avoid" in mem.content


class TestKnowledgeTriples:
    def test_tool_triples(self) -> None:
        r = make_result(tool="calculator", output="42")
        triples = FeedbackLoop.build_knowledge_triples(r, None, "conv-1")
        assert len(triples) == 2  # Agent→used→calculator, Calculator→status→completed
        assert any(t.subject == "Agent" and t.predicate == "used" for t in triples)

    def test_error_triple(self) -> None:
        r = make_result(status=TaskStatus.FAILED, tool="search", error="timeout")
        triples = FeedbackLoop.build_knowledge_triples(r, None, "conv-1")
        assert any(t.predicate == "failed_with" and "timeout" in t.object_ for t in triples)

    def test_reflection_triple(self) -> None:
        r = make_result(tool="calc")
        ref = make_reflection("retry")
        triples = FeedbackLoop.build_knowledge_triples(r, ref, "conv-1")
        assert any(t.subject == "Reflection" and t.predicate == "decision" for t in triples)


class TestFullFeedbackLoop:
    def test_process_completed_task(self) -> None:
        r = make_result(tool="calculator", output="42")
        ref = make_reflection("accept")
        fb = FeedbackLoop.process(r, ref, conversation_id="conv-1")
        assert fb.importance == 0.6
        assert len(fb.episodic_memories) == 1
        assert len(fb.semantic_memories) == 1
        assert len(fb.procedural_memories) == 1
        assert len(fb.knowledge_triples) >= 1

    def test_process_failed_task(self) -> None:
        r = make_result(status=TaskStatus.FAILED, tool="web_search", error="timeout")
        ref = make_reflection("retry")
        fb = FeedbackLoop.process(r, ref, conversation_id="conv-1")
        assert fb.importance > 0.8
        assert len(fb.episodic_memories) == 1
        assert len(fb.semantic_memories) >= 1
        assert len(fb.knowledge_triples) >= 3

    def test_process_chat_no_tool(self) -> None:
        r = make_result(output="Just chatting")
        fb = FeedbackLoop.process(r, None, conversation_id="conv-1")
        assert fb.importance == 0.3
        assert len(fb.semantic_memories) == 0
        assert len(fb.procedural_memories) == 0


class TestPersistence:
    async def test_persist_stores_memories(self) -> None:
        store = SQLiteVectorStore()
        embedder = FakeEmbedder()
        manager = MemoryManager(vector_store=store, embedder=embedder)

        r = make_result(tool="calculator", output="42")
        ref = make_reflection("accept")
        fb = FeedbackLoop.process(r, ref)
        await FeedbackLoop.persist(manager, "conv-1", fb)

        memories = await manager.list_memories()
        assert len(memories) >= 2  # episodic + semantic + procedural

    async def test_persist_without_knowledge_graph(self) -> None:
        store = SQLiteVectorStore()
        embedder = FakeEmbedder()
        manager = MemoryManager(vector_store=store, embedder=embedder)

        r = make_result(tool="calc")
        fb = FeedbackLoop.process(r, None)
        await FeedbackLoop.persist(manager, "conv-1", fb)
        assert await manager.list_memories()

    async def test_persist_empty_feedback_noop(self) -> None:
        store = SQLiteVectorStore()
        embedder = FakeEmbedder()
        manager = MemoryManager(vector_store=store, embedder=embedder)

        fb = FeedbackResult(importance=0.1)
        await FeedbackLoop.persist(manager, "conv-1", fb)
        assert await manager.list_memories() == []
