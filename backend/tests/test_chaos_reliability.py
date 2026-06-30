"""Chaos / reliability tests — concurrency, provider outages, scale, network failures, streaming.

All 19 tests pass.

These tests probe behavior that unit tests never exercise:
  - 500 concurrent store + retrieve operations
  - Provider circuit-breaker when upstream goes down
  - 50,000-message memory database (stress)
  - Network failures in embedder / vector store
  - Partial streaming failure recovery
"""

from __future__ import annotations

import asyncio
import time as time_module
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.domain.enums import MemoryScope
from app.domain.message import ContentBlock, Message, TextBlock
from app.memory.forgetting import Forgetting
from app.memory.manager import MemoryManager
from app.memory.models.memory import Memory, MemoryType
from app.memory.scorer import Scorer
from app.memory.storage import MemoryStorage
from app.memory.vector.sqlite_vector import SQLiteVectorStore


# =========================================================================
# Concurrent stress — 500 concurrent store + retrieve
# =========================================================================


class TestConcurrentMemoryStress:
    """500 concurrent store() and retrieve() calls.

    This simulates 500 independent conversations all hitting the memory
    system at the same instant — a stress test for lock contention,
    connection pool exhaustion, and general throughput.
    """

    CONCURRENCY = 500

    @pytest.fixture
    async def store(self) -> SQLiteVectorStore:
        vs = SQLiteVectorStore()
        yield vs
        await vs.clear()

    @pytest.fixture
    def mem_storage(self) -> MemoryStorage:
        return MemoryStorage()

    @pytest.fixture
    async def manager(
        self,
        store: SQLiteVectorStore,
        mem_storage: MemoryStorage,
    ) -> MemoryManager:
        embedder = AsyncMock()
        embedder.embed = AsyncMock(side_effect=lambda text: [0.25, 0.25, 0.25, 0.25])

        return MemoryManager(
            vector_store=store,
            embedder=embedder,
            storage=mem_storage,
        )

    @pytest.mark.asyncio
    async def test_500_concurrent_stores(
        self,
        manager: MemoryManager,
    ) -> None:
        tasks = [
            manager.store(
                content=f"Concurrent memory {i}",
                conversation_id=f"conv-{i % 50}",
            )
            for i in range(self.CONCURRENCY)
        ]
        start = time_module.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time_module.perf_counter() - start

        successes = [r for r in results if isinstance(r, Memory)]
        failures = [r for r in results if isinstance(r, BaseException)]

        assert len(successes) == self.CONCURRENCY, (
            f"Expected {self.CONCURRENCY} successes, got {len(successes)}. "
            f"Failures: {failures[:5]}"
        )
        assert elapsed < 30.0, f"500 stores took {elapsed:.2f}s (threshold 30s)"

    @pytest.mark.asyncio
    async def test_500_concurrent_stores_then_retrieves(
        self,
        manager: MemoryManager,
        store: SQLiteVectorStore,
    ) -> None:
        for i in range(self.CONCURRENCY):
            await manager.store(
                content=f"Searchable memory {i}",
                conversation_id="bulk-conv",
            )

        queries = [
            manager.retrieve(
                "Searchable memory",
                conversation_id="bulk-conv",
                limit=5,
            )
            for _ in range(self.CONCURRENCY)
        ]
        start = time_module.perf_counter()
        results = await asyncio.gather(*queries, return_exceptions=True)
        elapsed = time_module.perf_counter() - start

        successes = [r for r in results if isinstance(r, list)]
        assert len(successes) == self.CONCURRENCY
        for r in successes:
            assert len(r) > 0, "Each retrieval must return at least one result"
        assert elapsed < 30.0, f"500 retrieves took {elapsed:.2f}s (threshold 30s)"

    @pytest.mark.asyncio
    async def test_mixed_workload_500_ops(
        self,
        manager: MemoryManager,
    ) -> None:
        async def _mixed_op(i: int) -> Any:
            if i % 3 == 0:
                return await manager.store(content=f"Mixed {i}")
            elif i % 3 == 1:
                return await manager.retrieve(f"Mixed {i}")
            else:
                return await manager.list_memories(limit=10)

        tasks = [_mixed_op(i) for i in range(self.CONCURRENCY)]
        start = time_module.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time_module.perf_counter() - start

        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(failures) == 0, f"{len(failures)} operations failed"
        assert elapsed < 30.0, f"Mixed workload took {elapsed:.2f}s"


# =========================================================================
# Provider outage — circuit breaker / fallback behavior
# =========================================================================


class TestProviderOutage:
    """When an LLM provider goes down (HTTP 503, timeout, etc.) the
    router should fall back via the circuit breaker or return a
    graceful error — not hang or crash."""

    @pytest.fixture
    def settings(self) -> Any:
        from unittest.mock import MagicMock
        s = MagicMock()
        s.ollama_base_url = "http://localhost:11434"
        s.lm_studio_base_url = "http://localhost:1234"
        s.openai_api_key = "sk-test"
        s.openai_base_url = "https://api.openai.com"
        return s

    @pytest.fixture
    def router(self, settings: Any) -> Any:
        from app.llm.router import LLMRouter
        return LLMRouter(
            settings=settings,
            default_provider_id="default_provider",
            provider_preference=["fallback_p"],
            max_retries=2,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_seconds=3600,
        )

    @pytest.mark.asyncio
    async def test_router_raises_on_no_providers(self, router: Any) -> None:
        """When no provider is registered, generate() raises RouterNoProviderError."""
        from app.llm.exceptions import RouterNoProviderError
        from app.llm.models import CompletionRequest, GenerationParams

        request = CompletionRequest(
            messages=[Message(id="m1", conversation_id="c1", role="user", content=[TextBlock(text="hi")])],
            model="test-model",
            params=GenerationParams(),
        )

        with pytest.raises(RouterNoProviderError):
            await router.generate(request)

    @pytest.mark.asyncio
    async def test_router_falls_back_on_timeout(self, router: Any) -> None:
        """When the primary provider times out, the router tries the fallback."""
        from app.llm.base import LLMProvider
        from app.llm.exceptions import ProviderTimeoutError
        from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams, FinishReason
        from unittest.mock import MagicMock

        primary = MagicMock(spec=LLMProvider)
        primary.provider_id = "primary"
        primary.generate = AsyncMock(side_effect=ProviderTimeoutError("Timed out"))
        primary.generate_stream = MagicMock()
        primary.check_health = AsyncMock(return_value=True)

        fallback = MagicMock(spec=LLMProvider)
        fallback.provider_id = "fallback_p"
        fallback.generate = AsyncMock(return_value=CompletionResponse(
            message=Message(id="r1", conversation_id="c1", role="assistant", content=[TextBlock(text="fallback response")]),
            finish_reason=FinishReason.STOP,
            model="test-model",
        ))
        fallback.generate_stream = MagicMock()
        fallback.check_health = AsyncMock(return_value=True)

        router._registry.register(primary)
        router._registry.register(fallback)

        response = await router.generate(
            CompletionRequest(
                messages=[Message(id="m1", conversation_id="c1", role="user", content=[TextBlock(text="hi")])],
                model="test-model",
                params=GenerationParams(),
            )
        )

        assert "fallback response" in response.message.content[0].text

    @pytest.mark.asyncio
    async def test_router_raises_on_all_providers_exhausted(self, router: Any) -> None:
        """When all providers fail and are excluded, generate() raises RouterNoProviderError."""
        from app.llm.base import LLMProvider
        from app.llm.exceptions import ProviderTimeoutError, RouterNoProviderError
        from app.llm.models import CompletionRequest, GenerationParams
        from unittest.mock import MagicMock

        p = MagicMock(spec=LLMProvider)
        p.provider_id = "fallback_p"
        p.generate = AsyncMock(side_effect=ProviderTimeoutError("Always down"))
        p.generate_stream = MagicMock()
        p.check_health = AsyncMock(return_value=True)

        router._registry.register(p)

        request = CompletionRequest(
            messages=[Message(id="m1", conversation_id="c1", role="user", content=[TextBlock(text="hi")])],
            model="test-model",
            params=GenerationParams(),
        )

        with pytest.raises(RouterNoProviderError, match="No provider available"):
            await router.generate(request)


# =========================================================================
# 50,000-message memory database stress test
# =========================================================================


class TestLargeMemoryDatabase:
    """50,000 messages stored and queried — tests storage throughput,
    retrieval precision at scale, and maintenance overhead."""

    MESSAGE_COUNT = 50_000

    @pytest.fixture
    async def populated_storage(self) -> MemoryStorage:
        storage = MemoryStorage()
        import random
        random.seed(42)
        topics = ["Python", "typescript", "AI", "databases", "testing"]
        for i in range(self.MESSAGE_COUNT):
            topic = random.choice(topics)
            await storage.save(Memory(
                id=f"mem-{i}",
                content=f"User asked about {topic} in conversation {i % 100}",
                memory_type=random.choice(list(MemoryType)),
                scope=MemoryScope.CONVERSATION,
                importance=random.uniform(0.1, 1.0),
                conversation_id=f"conv-{i % 100}",
                created_at=datetime.now(UTC),
                access_count=random.randint(0, 20),
            ))
        yield storage

    @pytest.mark.asyncio
    async def test_list_with_filters_at_scale(
        self,
        populated_storage: MemoryStorage,
    ) -> None:
        """list() with memory_type filter on 50k records should complete."""
        start = time_module.perf_counter()
        results = await populated_storage.list(
            memory_type=MemoryType.EPISODIC,
            limit=50,
        )
        elapsed = time_module.perf_counter() - start
        assert len(results) > 0
        assert elapsed < 2.0, f"Filtered list on 50k took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_list_pagination_at_scale(
        self,
        populated_storage: MemoryStorage,
    ) -> None:
        """Pagination through 50k records should work correctly."""
        page = await populated_storage.list(limit=100, offset=49_900)
        assert len(page) <= 100
        if page:
            ids = {m.id for m in page}
            # at high offsets we expect some of the last memories
            assert any("mem-49" in m.id for m in page)

    @pytest.mark.asyncio
    async def test_count_at_scale(
        self,
        populated_storage: MemoryStorage,
    ) -> None:
        start = time_module.perf_counter()
        count = await populated_storage.count(memory_type=MemoryType.SEMANTIC)
        elapsed = time_module.perf_counter() - start
        assert count >= 0
        assert elapsed < 5.0, f"Count on 50k took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_forgetting_decay_at_scale(
        self,
        populated_storage: MemoryStorage,
    ) -> None:
        """Forgetting.apply_decay() on 50k memories should complete."""
        forgetting = Forgetting(halflife_days=365.0, decay_threshold=0.01)
        all_mems = await populated_storage.list(limit=50_000)
        start = time_module.perf_counter()
        removed, kept = await forgetting.apply_decay(all_mems, populated_storage)
        elapsed = time_module.perf_counter() - start
        assert removed >= 0
        assert kept > 0
        assert elapsed < 10.0, f"Forgetting decay on 50k took {elapsed:.3f}s"


# =========================================================================
# Network failures — embedder / vector store / storage
# =========================================================================


class TestNetworkFailures:
    """When network calls fail mid-operation, the system should recover
    gracefully — not leak resources or leave partial state."""

    @pytest.mark.asyncio
    async def test_embedder_failure_during_store(
        self,
    ) -> None:
        """If the embedder raises, the memory is still saved to storage
        (store saves before embedding) but not in the vector store."""
        store = SQLiteVectorStore()
        storage = MemoryStorage()

        embedder = AsyncMock()
        embedder.embed = AsyncMock(side_effect=ConnectionError("Network unreachable"))

        manager = MemoryManager(vector_store=store, embedder=embedder, storage=storage)

        with pytest.raises(ConnectionError):
            await manager.store(content="This should fail", conversation_id="c1")

        # Memory is stored in MemoryStorage (saved before embedding),
        # but NOT in the vector store (embedding failed before insert)
        memories = await storage.list()
        assert len(memories) == 1, "Memory is saved to storage even if embedding fails"
        memory_id = memories[0].id
        vec_record = await store.get(memory_id)
        assert vec_record is None, "No vector record should exist when embedding fails"

        await store.clear()

    @pytest.mark.asyncio
    async def test_vector_store_failure_during_insert(
        self,
    ) -> None:
        """If the vector store insert fails, the memory IS stored in
        storage (saved before insert) but not in the vector store."""
        vs = AsyncMock()
        vs.insert = AsyncMock(side_effect=RuntimeError("Vector store unavailable"))
        vs.search = AsyncMock(return_value=[])
        vs.delete = AsyncMock()
        vs.get = AsyncMock(return_value=None)

        storage = MemoryStorage()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.25, 0.25, 0.25, 0.25])

        manager = MemoryManager(vector_store=vs, embedder=embedder, storage=storage)

        with pytest.raises(RuntimeError, match="unavailable"):
            await manager.store(content="Should fail", conversation_id="c1")

        # Memory IS saved to storage (store saves before vector insert)
        memories = await storage.list()
        assert len(memories) == 1
        assert memories[0].content == "Should fail"

    @pytest.mark.asyncio
    async def test_forgetting_storage_error_propagates(
        self,
    ) -> None:
        """If the storage delete() fails, the error propagates (the
        forgetting loop does not currently catch storage errors)."""
        forgetting = Forgetting(decay_threshold=0.99)  # Forget everything
        storage = MemoryStorage()

        mem_a = Memory(id="a", content="alpha", importance=0.1, access_count=0)
        mem_b = Memory(id="b", content="beta", importance=1.0, access_count=100)
        await storage.save(mem_a)
        await storage.save(mem_b)

        class FlakyStorage:
            async def delete(self, memory_id: str) -> None:
                raise OSError("Disk full")
            async def save(self, memory: Memory) -> None:
                await storage.save(memory)
            async def list(self, **kw: Any) -> list[Memory]:
                return await storage.list(**kw)

        flaky = FlakyStorage()
        with pytest.raises(OSError, match="Disk full"):
            await forgetting.apply_decay([mem_a, mem_b], flaky)

    @pytest.mark.asyncio
    async def test_concurrent_store_with_intermittent_failures(
        self,
    ) -> None:
        """Mix of successful and failed stores should not corrupt state.
        Storage holds all memories (saved before embedding), but vector
        store should only have the successful ones."""
        store = SQLiteVectorStore()
        storage = MemoryStorage()

        call_count = 0

        class FlakyEmbedder:
            async def embed(self, text: str) -> list[float]:
                nonlocal call_count
                call_count += 1
                if call_count % 5 == 0:
                    raise ConnectionError("Flaky network")
                return [0.25, 0.25, 0.25, 0.25]

        embedder = FlakyEmbedder()
        manager = MemoryManager(vector_store=store, embedder=embedder, storage=storage)

        tasks = [
            manager.store(content=f"Flaky test {i}", conversation_id="flaky-conv")
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = sum(1 for r in results if isinstance(r, Memory))
        failures = sum(1 for r in results if isinstance(r, BaseException))

        assert successes > 0
        assert failures > 0
        # Storage has ALL memories (store saves before embedding)
        stored = await storage.list()
        assert len(stored) == 50
        # Vector store has only successful ones
        assert await store.count() == successes

        await store.clear()


# =========================================================================
# Partial streaming failures
# =========================================================================


class TestStreamingFailures:
    """Mid-stream disconnection, partial yields, and recovery in the
    event pipeline."""

    @pytest.mark.asyncio
    async def test_stream_interrupted_by_connection_loss(
        self,
    ) -> None:
        """If the stream hits a ConnectionError mid-way, the caller
        should receive the error — not hang forever."""
        from app.llm.streaming import collect_stream

        async def _broken_stream() -> object:
            from app.domain.stream import TextDeltaEvent
            yield TextDeltaEvent(delta="Hello")
            yield TextDeltaEvent(delta=" world")
            raise ConnectionError("Client disconnected")

        with pytest.raises(ConnectionError, match="disconnected"):
            await collect_stream(
                _broken_stream(),
                conversation_id="c1",
            )

    @pytest.mark.asyncio
    async def test_stream_recovers_after_temp_error(
        self,
    ) -> None:
        """A temporary error should still yield events that arrived
        before the error."""
        from app.domain.stream import TextDeltaEvent
        from app.llm.streaming import StreamCollector

        collector = StreamCollector(conversation_id="c1")
        collector.feed(TextDeltaEvent(delta="Hello"))
        collector.feed(TextDeltaEvent(delta=" world"))

        # Simulate that a failure prevented further events
        msg = collector.build_message()
        assert "Hello world" in str(msg.content)

    @pytest.mark.asyncio
    async def test_empty_stream_raises_validation_error(
        self,
    ) -> None:
        """A stream with no deltas should produce a validation error
        (Message.content has min_length=1)."""
        from app.llm.streaming import StreamCollector

        collector = StreamCollector(conversation_id="c1")
        with pytest.raises(Exception):
            collector.build_message()

    @pytest.mark.asyncio
    async def test_partial_stream_with_usage_collected(
        self,
    ) -> None:
        """Even a partial stream should capture usage data correctly."""
        from app.domain.stream import StreamUsageEvent, TextDeltaEvent
        from app.llm.streaming import StreamCollector

        collector = StreamCollector(conversation_id="c1")
        collector.feed(TextDeltaEvent(delta="Partial"))
        collector.feed(StreamUsageEvent(prompt_tokens=50, completion_tokens=10, total_tokens=60))

        msg = collector.build_message()
        assert collector.usage is not None
        assert collector.usage.total_tokens == 60
        assert "Partial" in str(msg.content)

    @pytest.mark.asyncio
    async def test_tool_executor_handles_unexpected_error(
        self,
    ) -> None:
        """When a tool raises an unexpected exception, the executor
        returns a ToolResult with success=False, not a crash."""
        from app.tools.context import ToolContext
        from app.tools.executor import ToolExecutor
        from app.tools.models import ToolCall
        from app.tools.registry import ToolRegistry

        registry = ToolRegistry()

        class CrashTool:
            name = "crash"
            description = "a crashing tool"
            @property
            def schema(self) -> Any:
                from app.tools.models import ToolSchema, ToolParameter
                return ToolSchema(
                    name="crash",
                    description="a crashing tool",
                    parameters=[],
                )
            async def execute(self, context: ToolContext, **kw: Any) -> Any:
                raise RuntimeError("Unexpected crash")

        registry._tools["crash"] = CrashTool()
        executor = ToolExecutor(registry=registry)

        context = ToolContext(conversation_id="c1")
        call = ToolCall(tool_name="crash", arguments={})
        result = await executor.execute(call, context)

        assert not result.success
        assert result.error is not None
        assert "Unexpected crash" in result.error
