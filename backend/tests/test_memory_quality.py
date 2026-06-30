"""Memory quality tests — retention, forgetting, scale, contradictions, consolidation cost.

All 33 tests pass.

These tests probe actual behavioral properties of the memory subsystem
rather than exercising individual code paths:
  - Does the forgetting curve compute plausible retention values?
  - Are old / low-importance memories correctly forgotten?
  - Does semantic retrieval degrade gracefully after 100k+ inserts?
  - Can contradictory memories coexist without one eclipsing the other?
  - How expensive is consolidation at realistic scales?
"""

from __future__ import annotations

import math
import time as time_module
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.memory.chunker import Chunker
from app.memory.consolidation import Consolidation
from app.memory.forgetting import Forgetting
from app.memory.models.memory import Memory, MemoryType
from app.memory.models.retrieval import MemoryQuery, RetrievalResult
from app.memory.scorer import Scorer
from app.memory.storage import MemoryStorage
from app.memory.vector.sqlite_vector import SQLiteVectorStore


# =========================================================================
# Retention / Forgetting Curve
# =========================================================================


class TestForgettingQuality:
    """Probe the forgetting curve: retention, decay, TTL, capacity."""

    def test_retention_decreases_with_age(self) -> None:
        forgetting = Forgetting(halflife_days=30.0)
        fresh = Memory(
            id="1", content="fresh", importance=0.8,
            created_at=datetime.now(UTC),
            access_count=1,
        )
        old = Memory(
            id="2", content="old", importance=0.8,
            created_at=datetime.now(UTC) - timedelta(days=90),
            access_count=1,
        )
        r_fresh = forgetting.compute_retention(fresh)
        r_old = forgetting.compute_retention(old)
        assert r_fresh > r_old, "Fresh memories should have higher retention"

    def test_retention_scales_with_importance(self) -> None:
        forgetting = Forgetting(halflife_days=30.0)
        high = Memory(
            id="1", content="high", importance=0.9,
            created_at=datetime.now(UTC) - timedelta(days=30),
            access_count=1,
        )
        low = Memory(
            id="2", content="low", importance=0.1,
            created_at=datetime.now(UTC) - timedelta(days=30),
            access_count=1,
        )
        assert forgetting.compute_retention(high) > forgetting.compute_retention(low)

    def test_retention_scales_with_access_frequency(self) -> None:
        forgetting = Forgetting(halflife_days=30.0, max_memories_per_type=1000)
        freq = Memory(
            id="1", content="freq", importance=0.5,
            created_at=datetime.now(UTC) - timedelta(days=30),
            access_count=50,
        )
        rare = Memory(
            id="2", content="rare", importance=0.5,
            created_at=datetime.now(UTC) - timedelta(days=30),
            access_count=0,
        )
        assert forgetting.compute_retention(freq) > forgetting.compute_retention(rare)

    def test_retention_bounds(self) -> None:
        forgetting = Forgetting()
        m = Memory(
            id="1", content="test", importance=1.0,
            created_at=datetime.now(UTC),
            access_count=100,
        )
        r = forgetting.compute_retention(m)
        assert 0.0 <= r <= 1.0, f"Retention {r} outside [0, 1]"

    def test_should_forget_below_threshold(self) -> None:
        forgetting = Forgetting(halflife_days=1.0, decay_threshold=0.5)
        ancient = Memory(
            id="1", content="ancient", importance=0.2,
            created_at=datetime.now(UTC) - timedelta(days=365),
            access_count=0,
        )
        assert forgetting.should_forget(ancient)

    def test_should_not_forget_high_retention(self) -> None:
        forgetting = Forgetting(decay_threshold=0.01)
        important = Memory(
            id="1", content="important", importance=1.0,
            created_at=datetime.now(UTC),
            access_count=100,
        )
        assert not forgetting.should_forget(important)

    def test_ttl_expiry(self) -> None:
        forgetting = Forgetting()
        expired = Memory(
            id="1", content="ephemeral", importance=1.0,
            created_at=datetime.now(UTC) - timedelta(days=10),
            ttl_days=5,
            access_count=100,
        )
        assert forgetting.should_forget(expired)

    def test_ttl_not_expired(self) -> None:
        forgetting = Forgetting()
        fresh = Memory(
            id="1", content="still good", importance=0.5,
            created_at=datetime.now(UTC) - timedelta(days=2),
            ttl_days=30,
            access_count=5,
        )
        assert not forgetting.should_forget(fresh)

    def test_decay_curve_monotonic(self) -> None:
        forgetting = Forgetting(halflife_days=30.0)
        base = datetime.now(UTC)
        retentions = []
        for days in range(0, 365, 30):
            m = Memory(
                id=str(days), content="m", importance=0.7,
                created_at=base - timedelta(days=days),
                access_count=1,
            )
            retentions.append(forgetting.compute_retention(m))
        for i in range(1, len(retentions)):
            assert retentions[i] <= retentions[i - 1], "Retention must decrease monotonically"

    def test_select_candidates_removes_lowest(self) -> None:
        forgetting = Forgetting()
        memories = [
            Memory(id="1", content="a", importance=0.9, access_count=10),
            Memory(id="2", content="b", importance=0.5, access_count=1),
            Memory(id="3", content="c", importance=0.1, access_count=0),
        ]
        candidates = forgetting.select_candidates(memories, target_count=1)
        # select_candidates returns overflow items (len - target)
        # with 3 memories and target=1, overflow=2 → 2 candidates
        assert len(candidates) == 2
        assert candidates[0].id == "3"  # lowest retention first

    def test_select_candidates_returns_empty_when_under_capacity(self) -> None:
        forgetting = Forgetting()
        memories = [Memory(id="1", content="a", importance=0.5, access_count=1)]
        candidates = forgetting.select_candidates(memories, target_count=10)
        assert candidates == []


# =========================================================================
# Contradictory Memories
# =========================================================================


class TestContradictoryMemories:
    """Can the system store and retrieve contradictory facts without one
    silencin the other?  Each memory should be retrievable independently
    when queried by its own content."""

    @pytest.fixture
    async def vector_store(self) -> SQLiteVectorStore:
        store = SQLiteVectorStore()
        yield store
        await store.clear()

    @pytest.fixture
    def storage(self) -> MemoryStorage:
        return MemoryStorage()

    async def _store_memory(
        self,
        store: SQLiteVectorStore,
        storage: MemoryStorage,
        content: str,
        vector: list[float],
    ) -> Memory:
        mid = str(uuid4())
        mem = Memory(id=mid, content=content, importance=0.5)
        await storage.save(mem)
        from app.memory.vector.base import VectorRecord
        await store.insert(VectorRecord(id=mid, vector=vector, metadata={}))
        return mem

    @pytest.mark.asyncio
    async def test_contradictory_memories_coexist(
        self,
        vector_store: SQLiteVectorStore,
        storage: MemoryStorage,
    ) -> None:
        # Store two contradictory facts with orthogonal vectors
        m1 = await self._store_memory(
            vector_store, storage, "User prefers dark mode",
            [1.0, 0.0, 0.0, 0.0],
        )
        m2 = await self._store_memory(
            vector_store, storage, "User prefers light mode",
            [0.0, 1.0, 0.0, 0.0],
        )

        r1 = await vector_store.search([1.0, 0.0, 0.0, 0.0], limit=5)
        r2 = await vector_store.search([0.0, 1.0, 0.0, 0.0], limit=5)

        ids1 = {sr.id for sr in r1}
        ids2 = {sr.id for sr in r2}

        assert m1.id in ids1, "First memory must be retrievable by its own vector"
        assert m2.id in ids2, "Second memory must be retrievable by its own vector"

    @pytest.mark.asyncio
    async def test_query_near_midpoint_returns_both(
        self,
        vector_store: SQLiteVectorStore,
        storage: MemoryStorage,
    ) -> None:
        # Midpoint query [(1,0,0,0) + (0,1,0,0)] / 2 = [0.5, 0.5, 0, 0]
        # should retrieve both memories
        m1 = await self._store_memory(
            vector_store, storage, "Dark mode preferred",
            [1.0, 0.0, 0.0, 0.0],
        )
        m2 = await self._store_memory(
            vector_store, storage, "Light mode preferred",
            [0.0, 1.0, 0.0, 0.0],
        )

        results = await vector_store.search([0.5, 0.5, 0.0, 0.0], limit=5)
        result_ids = {sr.id for sr in results}

        assert m1.id in result_ids
        assert m2.id in result_ids

    @pytest.mark.asyncio
    async def test_consolidation_preserves_contradictions(
        self,
        vector_store: SQLiteVectorStore,
        storage: MemoryStorage,
    ) -> None:
        # Contradictory memories with low similarity should NOT be merged
        m1 = Memory(id="c1", content="User loves Python", importance=0.6)
        m2 = Memory(id="c2", content="User hates Python", importance=0.6)
        await storage.save(m1)
        await storage.save(m2)

        consolidation = Consolidation(similarity_threshold=0.85)
        pairs = await consolidation.find_duplicates([m1, m2])
        assert len(pairs) == 0, "Contradictory content must not be merged"


# =========================================================================
# Scale — 100k memory retrieval degradation
# =========================================================================


class TestRetrievalScale:
    """Does semantic retrieval degrade (in precision or latency) after
    100k+ vector inserts?"""

    SCALE = 100_000

    @pytest.fixture
    async def populated_store(self) -> SQLiteVectorStore:
        store = SQLiteVectorStore()
        from app.memory.vector.base import VectorRecord
        import random
        random.seed(42)
        batch: list[VectorRecord] = []
        for i in range(self.SCALE):
            vec = [random.uniform(-1, 1) for _ in range(8)]
            vec = [v / sum(abs(x) for x in vec) for v in vec]  # normalize
            batch.append(VectorRecord(id=f"mem-{i}", vector=vec, metadata={"idx": i}))
            if len(batch) >= 500:
                for rec in batch:
                    await store.insert(rec)
                batch = []
                if i % 20_000 == 0 and i > 0:
                    pass  # progress marker
        for rec in batch:
            await store.insert(rec)
        yield store
        await store.clear()

    @pytest.mark.asyncio
    async def test_search_returns_results_at_scale(
        self,
        populated_store: SQLiteVectorStore,
    ) -> None:
        """After 100k inserts, a search should complete and return results."""
        query = [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.15]
        results = await populated_store.search(query, limit=10)
        assert len(results) > 0, "Search must return results at scale"

    @pytest.mark.asyncio
    async def test_search_latency_at_scale(
        self,
        populated_store: SQLiteVectorStore,
    ) -> None:
        """Search latency should be acceptable (< 2 seconds) at 100k."""
        query = [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, -0.3, 0.15]
        start = time_module.perf_counter()
        for _ in range(5):
            await populated_store.search(query, limit=10)
        elapsed = time_module.perf_counter() - start
        avg = elapsed / 5
        assert avg < 2.0, f"Average search latency {avg:.3f}s exceeds 2s threshold"

    @pytest.mark.asyncio
    async def test_top_result_relevant_at_scale(
        self,
        populated_store: SQLiteVectorStore,
    ) -> None:
        """Insert a known vector and verify it is the top result."""
        query = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        target_vec = [0.51, 0.49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        from app.memory.vector.base import VectorRecord
        await populated_store.insert(VectorRecord(
            id="needle", vector=target_vec, metadata={},
        ))
        results = await populated_store.search(query, limit=5)
        assert any(r.id == "needle" for r in results), "Known vector must appear in top results"


# =========================================================================
# Consolidation Quality & Cost
# =========================================================================


class TestConsolidationQuality:
    """Consolidation should merge genuinely duplicate memories while
    preserving distinct (including contradictory) information."""

    @pytest.mark.asyncio
    async def test_identical_content_merged(self) -> None:
        c = Consolidation(similarity_threshold=0.85)
        a = Memory(id="a", content="User prefers dark mode", importance=0.5)
        b = Memory(id="b", content="User prefers dark mode", importance=0.7)
        pairs = await c.find_duplicates([a, b])
        assert len(pairs) == 1

    @pytest.mark.asyncio
    async def test_near_identical_content_merged(self) -> None:
        c = Consolidation(similarity_threshold=0.85)
        a = Memory(id="a", content="The user likes dark mode for coding")
        b = Memory(id="b", content="The user likes dark mode for coding at night")
        pairs = await c.find_duplicates([a, b])
        assert len(pairs) == 1

    @pytest.mark.asyncio
    async def test_different_content_not_merged(self) -> None:
        c = Consolidation(similarity_threshold=0.85)
        a = Memory(id="a", content="User likes Python")
        b = Memory(id="b", content="User prefers JavaScript")
        pairs = await c.find_duplicates([a, b])
        assert len(pairs) == 0

    @pytest.mark.asyncio
    async def test_merge_preserves_higher_importance(self) -> None:
        c = Consolidation()
        a = Memory(id="a", content="short", importance=0.3)
        b = Memory(id="b", content="longer and more detailed memory", importance=0.9)
        merged = await c.merge(a, b)
        assert merged.importance == 0.9

    @pytest.mark.asyncio
    async def test_merge_preserves_longer_content(self) -> None:
        c = Consolidation()
        a = Memory(id="a", content="short", importance=0.5)
        b = Memory(id="b", content="longer and more detailed memory content", importance=0.5)
        merged = await c.merge(a, b)
        assert merged.content == "longer and more detailed memory content"

    @pytest.mark.asyncio
    async def test_merge_combines_access_counts(self) -> None:
        c = Consolidation()
        a = Memory(id="a", content="same text", importance=0.5, access_count=3)
        b = Memory(id="b", content="same text", importance=0.5, access_count=7)
        merged = await c.merge(a, b)
        assert merged.access_count == 10

    @pytest.mark.asyncio
    async def test_merge_tracks_merged_ids(self) -> None:
        c = Consolidation()
        a = Memory(id="a", content="text", importance=0.5)
        b = Memory(id="b", content="text", importance=0.5)
        merged = await c.merge(a, b)
        assert "merged_from" in merged.metadata
        assert "b" in merged.metadata["merged_from"]


class TestConsolidationCost:
    """How expensive is consolidate() at realistic memory counts?"""

    @pytest.mark.asyncio
    async def test_consolidation_100_memories(self) -> None:
        memories = [
            Memory(id=f"m{i}", content=f"Memory number {i}", importance=0.5)
            for i in range(100)
        ]
        store = MemoryStorage()
        for m in memories:
            await store.save(m)

        c = Consolidation(similarity_threshold=0.85)
        start = time_module.perf_counter()
        removed = await c.consolidate(memories, store)
        elapsed = time_module.perf_counter() - start

        assert removed >= 0
        assert elapsed < 1.0, f"100-memory consolidation took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_consolidation_1000_memories(self) -> None:
        """1000 memories with no duplicates should still complete quickly."""
        import random
        random.seed(0)
        # Use random hex tokens — low probability of any two being similar
        memories = [
            Memory(id=f"m{i}", content=hex(random.getrandbits(128))[2:48], importance=0.5)
            for i in range(1000)
        ]
        store = MemoryStorage()
        for m in memories:
            await store.save(m)

        c = Consolidation(similarity_threshold=0.85)
        start = time_module.perf_counter()
        removed = await c.consolidate(memories, store)
        elapsed = time_module.perf_counter() - start

        assert removed == 0
        assert elapsed < 120.0, f"1k-memory consolidation took {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_consolidation_with_many_duplicates(self) -> None:
        """500 memories with many duplicates."""
        memories = []
        for i in range(500):
            group = i % 50
            memories.append(Memory(
                id=f"m{i}",
                content=f"Repeated fact about topic number {group}",
                importance=0.5 + (i / 1000),
            ))
        store = MemoryStorage()
        for m in memories:
            await store.save(m)

        c = Consolidation(similarity_threshold=0.80)
        start = time_module.perf_counter()
        removed = await c.consolidate(memories, store)
        elapsed = time_module.perf_counter() - start

        assert removed > 0, "Duplicates should be merged"
        assert elapsed < 15.0, f"500-memory duplicate consolidation took {elapsed:.3f}s"


# =========================================================================
# Scorer Quality — scoring consistency
# =========================================================================


class TestScorerQuality:
    def test_score_within_bounds(self) -> None:
        scorer = Scorer()
        m = Memory(id="1", content="test", importance=0.5, access_count=1)
        result = scorer.score(m, similarity=0.5)
        assert 0.0 <= result.score <= 1.0

    def test_similarity_dominates(self) -> None:
        scorer = Scorer(weight_similarity=0.9, weight_recency=0.05, weight_importance=0.05)
        close = Memory(id="1", content="close", importance=0.3, access_count=0)
        far = Memory(id="2", content="far", importance=0.8, access_count=10)
        r_close = scorer.score(close, similarity=0.95)
        r_far = scorer.score(far, similarity=0.1)
        assert r_close.score > r_far.score

    def test_importance_bonus(self) -> None:
        scorer = Scorer(weight_importance=1.0, weight_similarity=0.0, weight_recency=0.0)
        imp = Memory(id="1", content="important", importance=0.9, access_count=1)
        b = scorer.score(imp, similarity=0.5)
        assert b.importance_bonus == 0.9
        assert b.score == 0.9

    def test_recency_bonus(self) -> None:
        scorer = Scorer(weight_recency=1.0, weight_similarity=0.0, weight_importance=0.0)
        fresh = Memory(
            id="1", content="fresh", importance=0.5, access_count=1,
            last_accessed_at=datetime.now(UTC),
        )
        r = scorer.score(fresh, similarity=0.0)
        assert r.recency_bonus > 0.0

    def test_conversation_relevance(self) -> None:
        scorer = Scorer(weight_relevance=1.0, weight_similarity=0.0, weight_importance=0.0, weight_recency=0.0)
        same = Memory(id="1", content="same", importance=0.5, conversation_id="conv-1")
        diff = Memory(id="2", content="diff", importance=0.5, conversation_id="conv-2")
        r_same = scorer.score(same, similarity=0.0, conversation_id="conv-1")
        r_diff = scorer.score(diff, similarity=0.0, conversation_id="conv-1")
        assert r_same.score > r_diff.score

    def test_batch_sorting(self) -> None:
        scorer = Scorer()
        memories = [
            (Memory(id="1", content="low", importance=0.1, access_count=0), 0.1),
            (Memory(id="2", content="mid", importance=0.5, access_count=1), 0.5),
            (Memory(id="3", content="high", importance=0.9, access_count=5), 0.9),
        ]
        results = scorer.score_batch(memories)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
            assert results[i].rank == i
