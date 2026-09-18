"""Semantic similarity retrieval strategy.

Requires an embedder to compute query-document similarity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.memory.embedder import Embedder
from app.memory.models.memory import Memory
from app.memory.models.retrieval import RetrievalResult
from app.memory.strategies.base import RetrievalStrategy


class SemanticStrategy(RetrievalStrategy):
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        *,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        query_vec = await self._embedder.embed(query)

        scored: list[tuple[Memory, float]] = []
        for memory in memories:
            mem_vec = await self._embedder.embed(memory.content)
            sim = self._cosine(query_vec, mem_vec)
            scored.append((memory, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]

        return [
            RetrievalResult(memory=mem, score=sim, similarity=sim, rank=i)
            for i, (mem, sim) in enumerate(top)
        ]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = na = nb = 0.0
        for ai, bi in zip(a, b, strict=False):
            dot += ai * bi
            na += ai * ai
            nb += bi * bi
        denom = math.sqrt(na) * math.sqrt(nb)
        return dot / denom if denom > 0 else 0.0
