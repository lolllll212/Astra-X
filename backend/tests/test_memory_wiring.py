"""Tests for the wired memory pipeline."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.enums import MemoryScope
from app.memory.embedder import Embedder
from app.memory.manager import MemoryManager
from app.memory.models.memory import MemoryType
from app.memory.vector.base import VectorRecord
from app.memory.vector.sqlite_vector import SQLiteVectorStore
from app.services.memory_service import MemoryService


class FakeEmbedder(Embedder):
    def __init__(self, dimensions: int = 4) -> None:
        self._dimensions = dimensions
        self._model = "test-model"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimensions for _ in texts]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> SQLiteVectorStore:
    return SQLiteVectorStore()


@pytest.fixture
def manager(store: SQLiteVectorStore, embedder: FakeEmbedder) -> MemoryManager:
    return MemoryManager(vector_store=store, embedder=embedder)


@pytest.fixture
def mock_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.list_by_conversation = AsyncMock(return_value=[])
    return repo


class TestMemoryServiceWithCoreManager:
    async def test_store_delegates_to_core(
        self, manager: MemoryManager, mock_repo: AsyncMock
    ) -> None:
        svc = MemoryService(mock_repo, core_memory_manager=manager)
        await svc.store_memory(
            conversation_id="conv-1",
            content="User likes Python.",
            metadata={"source": "test"},
        )
        memories = await manager.list_memories()
        assert len(memories) == 1
        assert memories[0].content == "User likes Python."
        assert memories[0].conversation_id == "conv-1"

    async def test_get_context_returns_semantic(
        self, manager: MemoryManager, mock_repo: AsyncMock
    ) -> None:
        await manager.store(content="User's favorite color is blue.", conversation_id="conv-1")
        svc = MemoryService(mock_repo, core_memory_manager=manager)
        context = await svc.get_relevant_context(
            conversation_id="conv-1",
            user_content=[MagicMock(text="What is my favorite color?")],
        )
        assert context is not None
        assert "blue" in context
        assert "score=" in context
        mock_repo.list_by_conversation.assert_not_awaited()

    async def test_get_context_falls_back_to_history(
        self, mock_repo: AsyncMock
    ) -> None:
        svc = MemoryService(mock_repo, core_memory_manager=None)
        context = await svc.get_relevant_context(
            conversation_id="conv-1",
            user_content=[MagicMock(text="Hello")],
        )
        assert context is None
        mock_repo.list_by_conversation.assert_awaited_once_with("conv-1", limit=20)


class TestMemoryServiceWithoutCoreManager:
    async def test_store_noop(self, mock_repo: AsyncMock) -> None:
        svc = MemoryService(mock_repo, core_memory_manager=None)
        await svc.store_memory(conversation_id="conv-1", content="Not stored.")

    async def test_get_context_uses_only_history(self, mock_repo: AsyncMock) -> None:
        mock_repo.list_by_conversation = AsyncMock(return_value=[])
        svc = MemoryService(mock_repo, core_memory_manager=None)
        context = await svc.get_relevant_context(conversation_id="conv-1")
        assert context is None


class TestMemoryManagerIntegration:
    async def test_store_and_retrieve(self, manager: MemoryManager) -> None:
        await manager.store(
            content="User prefers dark mode.",
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope.USER,
            conversation_id="conv-1",
            importance=0.8,
        )
        results = await manager.retrieve(query="dark mode", conversation_id="conv-1", limit=5)
        assert len(results) == 1
        assert results[0].memory.content == "User prefers dark mode."
        assert results[0].score > 0.0

    async def test_retrieve_filters_by_conversation(self, manager: MemoryManager) -> None:
        await manager.store(content="Memory A", conversation_id="conv-1")
        await manager.store(content="Memory B", conversation_id="conv-2")
        a = await manager.retrieve(query="memory", conversation_id="conv-1")
        assert len(a) == 1
        all_ = await manager.retrieve(query="memory")
        assert len(all_) == 2

    async def test_forget(self, manager: MemoryManager) -> None:
        mem = await manager.store(content="Temporary memory.")
        assert await manager.get_memory(mem.id) is not None
        assert await manager._vector_store.count() == 1
        await manager.forget(mem.id)
        assert await manager.get_memory(mem.id) is None
        assert await manager._vector_store.count() == 0

    async def test_list_with_filters(self, manager: MemoryManager) -> None:
        await manager.store(content="Episodic memory", memory_type=MemoryType.EPISODIC, conversation_id="conv-1")
        await manager.store(content="Semantic memory", memory_type=MemoryType.SEMANTIC, conversation_id="conv-1")
        episodic = await manager.list_memories(memory_type=MemoryType.EPISODIC, conversation_id="conv-1")
        assert len(episodic) == 1
        assert episodic[0].memory_type == MemoryType.EPISODIC


class TestSQLiteVectorStore:
    async def test_insert_and_search(self) -> None:
        s = SQLiteVectorStore()
        await s.insert(VectorRecord(id="1", vector=[1.0, 0.0, 0.0, 0.0], metadata={}))
        await s.insert(VectorRecord(id="2", vector=[0.0, 1.0, 0.0, 0.0], metadata={}))
        results = await s.search(query_vector=[1.0, 0.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0].id == "1"
        assert results[0].score > results[1].score

    async def test_delete(self) -> None:
        s = SQLiteVectorStore()
        await s.insert(VectorRecord(id="1", vector=[1.0, 0.0], metadata={}))
        assert await s.count() == 1
        await s.delete("1")
        assert await s.count() == 0

    async def test_clear(self) -> None:
        s = SQLiteVectorStore()
        await s.insert(VectorRecord(id="1", vector=[1.0], metadata={}))
        await s.clear()
        assert await s.count() == 0

    async def test_get(self) -> None:
        s = SQLiteVectorStore()
        await s.insert(VectorRecord(id="1", vector=[1.0], metadata={"key": "val"}))
        rec = await s.get("1")
        assert rec is not None
        assert rec.id == "1"
        assert rec.metadata == {"key": "val"}
        assert await s.get("missing") is None

    async def test_to_json_roundtrip(self) -> None:
        s1 = SQLiteVectorStore()
        await s1.insert(VectorRecord(id="1", vector=[1.0, 0.0], metadata={}))
        json_str = s1.to_json()
        s2 = SQLiteVectorStore.from_json(json_str)
        assert await s2.count() == 1
        rec = await s2.get("1")
        assert rec is not None
        assert rec.vector == [1.0, 0.0]
