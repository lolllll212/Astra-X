"""Reflection-based retrieval strategy.

Prioritises memories that were extracted through reflection
(high-level insights, preferences, facts) over raw conversation
history.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.memory.models.memory import Memory, MemoryType
from app.memory.models.retrieval import RetrievalResult
from app.memory.strategies.base import RetrievalStrategy


class ReflectionStrategy(RetrievalStrategy):
    async def retrieve(
        self,
        query: str,
        memories: Sequence[Memory],
        *,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        semantic_mems = [m for m in memories if m.memory_type == MemoryType.SEMANTIC]
        episodic_mems = [m for m in memories if m.memory_type == MemoryType.EPISODIC]

        semantic_mems.sort(key=lambda m: m.importance, reverse=True)
        episodic_mems.sort(key=lambda m: m.last_accessed_at, reverse=True)

        results: list[RetrievalResult] = []
        seen: set[str] = set()

        for mem in semantic_mems:
            if len(results) >= limit:
                break
            if mem.id in seen:
                continue
            seen.add(mem.id)
            results.append(
                RetrievalResult(
                    memory=mem,
                    score=mem.importance,
                    importance_bonus=mem.importance,
                    rank=len(results),
                )
            )

        for mem in episodic_mems:
            if len(results) >= limit:
                break
            if mem.id in seen:
                continue
            seen.add(mem.id)
            results.append(
                RetrievalResult(
                    memory=mem,
                    score=0.5 + 0.5 * mem.importance,
                    importance_bonus=mem.importance * 0.5,
                    rank=len(results),
                )
            )

        return results
