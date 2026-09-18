"""Recency-based retrieval strategy.

Ranks memories by how recently they were accessed.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.memory.models.memory import Memory
from app.memory.models.retrieval import RetrievalResult
from app.memory.strategies.base import RetrievalStrategy


class RecencyStrategy(RetrievalStrategy):
    async def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        *,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        sorted_mems = sorted(memories, key=lambda m: m.last_accessed_at, reverse=True)
        top = sorted_mems[:limit]

        if not top:
            return []

        max_time = top[0].last_accessed_at.timestamp()
        min_time = top[-1].last_accessed_at.timestamp()
        time_range = max(max_time - min_time, 1.0)

        results: list[RetrievalResult] = []
        for i, mem in enumerate(top):
            recency = (mem.last_accessed_at.timestamp() - min_time) / time_range
            results.append(
                RetrievalResult(
                    memory=mem,
                    score=recency,
                    recency_bonus=recency,
                    rank=i,
                )
            )
        return results
