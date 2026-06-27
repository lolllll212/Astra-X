from __future__ import annotations

from app.memory.embedder import Embedder
from app.memory.models.memory import Memory
from app.memory.models.retrieval import MemoryQuery, RetrievalResult
from app.memory.scorer import Scorer
from app.memory.storage import MemoryStorage
from app.memory.vector.base import VectorStore


class Retriever:
    """Pipeline for retrieving relevant memories.

    Query
      │
      ▼
    Embed
      │
      ▼
    Vector Search
      │
      ▼
    Scorer
      │
      ▼
    Top Memories
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        storage: MemoryStorage,
        scorer: Scorer | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._storage = storage
        self._scorer = scorer or Scorer()

    async def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
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
            limit=query.limit * 2,
            filter_=filter_,
        )

        memories_with_sim: list[tuple[Memory, float]] = []
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

            memories_with_sim.append((memory, sr.score))

        scored = self._scorer.score_batch(memories_with_sim, query.conversation_id)
        for r in scored:
            if r.memory.id:
                await self._storage.update_access(r.memory.id)

        return scored[: query.limit]
