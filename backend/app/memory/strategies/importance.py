"""Importance-based retrieval strategy.

Ranks memories by their importance score, optionally boosted by recency.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.memory.models.memory import Memory
from app.memory.models.retrieval import RetrievalResult
from app.memory.strategies.base import RetrievalStrategy


class ImportanceStrategy(RetrievalStrategy):
    async def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        *,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        sorted_mems = sorted(memories, key=lambda m: m.importance, reverse=True)
        top = sorted_mems[:limit]

        if not top:
            return []

        max_imp = top[0].importance if top else 1.0

        results: list[RetrievalResult] = []
        for i, mem in enumerate(top):
            norm_imp = mem.importance / max_imp if max_imp > 0 else 0.0
            results.append(
                RetrievalResult(
                    memory=mem,
                    score=norm_imp,
                    importance_bonus=norm_imp,
                    rank=i,
                )
            )
        return results
