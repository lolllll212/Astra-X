from __future__ import annotations

import asyncio
from collections.abc import Sequence

from app.memory.embedder import Embedder
from app.memory.fusion import fuse_reciprocal_rank, normalise_rrf
from app.memory.knowledge_graph import KnowledgeGraph
from app.memory.models.memory import Memory, MemoryType
from app.memory.models.retrieval import MemoryQuery, RetrievalResult
from app.memory.scorer import Scorer
from app.memory.storage import MemoryStorage
from app.memory.vector.base import VectorStore


class Retriever:
    """Multi-strategy memory retriever with Reciprocal Rank Fusion (RRF).

    Runs six retrieval paths in parallel and fuses their ranked lists
    before the final scoring pass:

    1. **Semantic** — vector similarity search via the embedder.
    2. **Recency** — most recently accessed memories.
    3. **Importance** — highest-importance memories.
    4. **Episodic** — recent episodic memories.
    5. **User Preferences** — PROCEDURAL-type memories with a boost.
    6. **Knowledge Graph** — entity-matching memories from the KG.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        storage: MemoryStorage,
        scorer: Scorer | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        preference_boost: float = 1.2,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._storage = storage
        self._scorer = scorer or Scorer()
        self._knowledge_graph = knowledge_graph
        self._preference_boost = preference_boost

    async def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
        # -- 1. Run all strategies in parallel ---------------------------------
        sem_task = self._semantic_retrieve(query)
        rec_task = self._recency_retrieve(query)
        imp_task = self._importance_retrieve(query)
        epi_task = (
            self._episodic_retrieve(query)
            if query.include_episodic
            else asyncio.sleep(0, [])
        )
        pref_task = (
            self._preference_retrieve(query)
            if query.include_procedural
            else asyncio.sleep(0, [])
        )
        kg_task = (
            self._kg_retrieve(query)
            if self._knowledge_graph is not None
            else asyncio.sleep(0, [])
        )

        (
            sem_results,
            rec_results,
            imp_results,
            epi_results,
            pref_results,
            kg_results,
        ) = await asyncio.gather(
            sem_task, rec_task, imp_task, epi_task, pref_task, kg_task
        )

        # -- 2. Build ranked lists for RRF -------------------------------------
        rank_lists: list[Sequence[Memory]] = []
        if sem_results:
            rank_lists.append([m for m, _ in sem_results])
        if rec_results:
            rank_lists.append(rec_results)
        if imp_results:
            rank_lists.append(imp_results)
        if epi_results:
            rank_lists.append(epi_results)
        if pref_results:
            rank_lists.append(pref_results)
        if kg_results:
            rank_lists.append(kg_results)

        raw_rrf = fuse_reciprocal_rank(rank_lists)
        rrf_scores = normalise_rrf(raw_rrf)

        # -- 3. Build Scorer input with preference boost -----------------------
        seen: set[str] = set()
        scored_input: list[tuple[Memory, float]] = []

        for mem, _sim in sem_results:
            rrf = rrf_scores.get(mem.id, 0.0)
            boost = self._preference_boost if mem.memory_type == MemoryType.PROCEDURAL else 1.0
            scored_input.append((mem, min(rrf * boost, 1.0)))
            seen.add(mem.id)

        for mem_list in (rec_results, imp_results, epi_results, pref_results, kg_results):
            for mem in mem_list:
                if mem.id not in seen:
                    rrf = rrf_scores.get(mem.id, 0.0)
                    boost = self._preference_boost if mem.memory_type == MemoryType.PROCEDURAL else 1.0
                    scored_input.append((mem, min(rrf * boost, 1.0)))
                    seen.add(mem.id)

        # -- 4. Final scoring --------------------------------------------------
        scored = self._scorer.score_batch(scored_input, query.conversation_id)

        # -- 5. Update access timestamps ---------------------------------------
        for r in scored:
            await self._storage.update_access(r.memory.id)

        return scored[: query.limit]

    # ------------------------------------------------------------------
    # Individual retrieval paths
    # ------------------------------------------------------------------

    async def _semantic_retrieve(
        self,
        query: MemoryQuery,
    ) -> list[tuple[Memory, float]]:
        """Vector similarity search with filters."""
        query_embedding = await self._embedder.embed(query.query)

        filter_: dict[str, object] = {}
        if query.conversation_id:
            filter_["conversation_id"] = query.conversation_id
        if query.user_id:
            filter_["user_id"] = query.user_id
        if query.memory_types:
            filter_["memory_type"] = [t.value for t in query.memory_types]

        search_results = await self._vector_store.search(
            query_embedding,
            limit=query.limit * 3,
            filter_=filter_,
        )

        results: list[tuple[Memory, float]] = []
        for sr in search_results:
            memory = await self._storage.get(sr.id)
            if memory is None:
                continue
            if memory.importance < query.min_importance:
                continue
            if query.memory_types and memory.memory_type not in query.memory_types:
                continue
            if query.scopes and memory.scope not in query.scopes:
                continue
            results.append((memory, sr.score))

        return results

    async def _recency_retrieve(self, query: MemoryQuery) -> list[Memory]:
        """Most recently accessed memories."""
        memories = await self._storage.list(
            conversation_id=query.conversation_id,
            user_id=query.user_id,
            limit=query.limit * 2,
        )
        memories.sort(key=lambda m: m.last_accessed_at, reverse=True)
        return self._apply_filters(memories, query)[: query.limit]

    async def _importance_retrieve(self, query: MemoryQuery) -> list[Memory]:
        """Highest-importance memories."""
        memories = await self._storage.list(
            conversation_id=query.conversation_id,
            user_id=query.user_id,
            limit=query.limit * 2,
        )
        memories.sort(key=lambda m: m.importance, reverse=True)
        return self._apply_filters(memories, query)[: query.limit]

    async def _episodic_retrieve(self, query: MemoryQuery) -> list[Memory]:
        """Recent episodic memories."""
        memories = await self._storage.list(
            memory_type=MemoryType.EPISODIC,
            conversation_id=query.conversation_id,
            user_id=query.user_id,
            limit=query.limit * 2,
        )
        memories.sort(key=lambda m: m.last_accessed_at, reverse=True)
        return self._apply_filters(memories, query)[: query.limit]

    async def _preference_retrieve(self, query: MemoryQuery) -> list[Memory]:
        """User preference / procedural memories."""
        memories = await self._storage.list(
            memory_type=MemoryType.PROCEDURAL,
            conversation_id=query.conversation_id,
            user_id=query.user_id,
            limit=query.limit * 2,
        )
        memories.sort(key=lambda m: m.importance, reverse=True)
        return self._apply_filters(memories, query)[: query.limit]

    async def _kg_retrieve(self, query: MemoryQuery) -> list[Memory]:
        """Knowledge-graph-augmented retrieval.

        Extracts candidate entity names from the query text, looks them up
        in the knowledge graph, and returns memories from the conversations
        that produced matching triples.
        """
        if self._knowledge_graph is None:
            return []

        words = query.query.lower().split()
        entity_candidates = {w for w in words if len(w) > 2}

        related_cids: set[str] = set()
        for entity in entity_candidates:
            for t in self._knowledge_graph.query(subject=entity):
                if t.conversation_id:
                    related_cids.add(t.conversation_id)
            for t in self._knowledge_graph.query(object_=entity):
                if t.conversation_id:
                    related_cids.add(t.conversation_id)
            for t in self._knowledge_graph.query(predicate=entity):
                if t.conversation_id:
                    related_cids.add(t.conversation_id)

        if not related_cids:
            return []

        all_mems: list[Memory] = []
        for cid in related_cids:
            conv_mems = await self._storage.list(
                conversation_id=cid,
                user_id=query.user_id,
                limit=query.limit,
            )
            all_mems.extend(conv_mems)

        all_mems = self._apply_filters(all_mems, query)

        seen: dict[str, Memory] = {}
        for m in all_mems:
            seen[m.id] = m

        result = sorted(seen.values(), key=lambda m: m.importance, reverse=True)
        return result[: query.limit]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_filters(
        memories: list[Memory],
        query: MemoryQuery,
    ) -> list[Memory]:
        filtered = [
            m
            for m in memories
            if m.importance >= query.min_importance
        ]
        if query.memory_types:
            filtered = [m for m in filtered if m.memory_type in query.memory_types]
        if query.scopes:
            filtered = [m for m in filtered if m.scope in query.scopes]
        if not query.include_working:
            filtered = [m for m in filtered if m.memory_type != MemoryType.WORKING]
        if not query.include_semantic:
            filtered = [m for m in filtered if m.memory_type != MemoryType.SEMANTIC]
        if not query.include_episodic:
            filtered = [m for m in filtered if m.memory_type != MemoryType.EPISODIC]
        if not query.include_procedural:
            filtered = [m for m in filtered if m.memory_type != MemoryType.PROCEDURAL]
        return filtered
