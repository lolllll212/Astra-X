"""Hybrid retrieval strategy combining multiple signals.

Combines recency, importance, and semantic similarity into a single
weighted score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

from app.memory.embedder import Embedder
from app.memory.models.memory import Memory
from app.memory.models.retrieval import RetrievalResult
from app.memory.strategies.base import RetrievalStrategy


class HybridStrategy(RetrievalStrategy):
    def __init__(
        self,
        embedder: Embedder,
        weight_semantic: float = 0.5,
        weight_recency: float = 0.25,
        weight_importance: float = 0.25,
        recency_halflife_days: float = 7.0,
    ) -> None:
        self._embedder = embedder
        self._w_sem = weight_semantic
        self._w_rec = weight_recency
        self._w_imp = weight_importance
        self._halflife = recency_halflife_days

    async def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        *,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        query_vec = await self._embedder.embed(query)

        scored: list[tuple[Memory, RetrievalResult]] = []
        for memory in memories:
            mem_vec = await self._embedder.embed(memory.content)
            similarity = self._cosine(query_vec, mem_vec)
            recency = self._recency_score(memory)
            imp = memory.importance

            combined = (
                self._w_sem * similarity
                + self._w_rec * recency
                + self._w_imp * imp
            )

            result = RetrievalResult(
                memory=memory,
                score=min(combined, 1.0),
                similarity=similarity,
                recency_bonus=recency,
                importance_bonus=imp,
            )
            scored.append((memory, result))

        scored.sort(key=lambda x: x[1].score, reverse=True)
        top = scored[:limit]

        for i, (_, r) in enumerate(top):
            r.rank = i

        return [r for _, r in top]

    def _recency_score(self, memory: Memory) -> float:
        age = (datetime.now(UTC) - memory.last_accessed_at).total_seconds() / 86400.0
        return math.exp(-math.log(2) * age / self._halflife)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = na = nb = 0.0
        for ai, bi in zip(a, b, strict=False):
            dot += ai * bi
            na += ai * ai
            nb += bi * bi
        denom = math.sqrt(na) * math.sqrt(nb)
        return dot / denom if denom > 0 else 0.0
