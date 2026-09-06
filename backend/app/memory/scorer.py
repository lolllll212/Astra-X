from __future__ import annotations

import math
from datetime import UTC, datetime

from app.memory.models.memory import Memory
from app.memory.models.retrieval import RetrievalResult


class Scorer:
    """Ranks memories using a weighted combination of signals.

    Final score = w_sim * similarity
                 + w_rec * recency
                 + w_imp * importance
                 + w_rel * conversation_relevance
    """

    def __init__(
        self,
        weight_similarity: float = 0.5,
        weight_recency: float = 0.2,
        weight_importance: float = 0.2,
        weight_relevance: float = 0.1,
        recency_halflife_days: float = 7.0,
    ) -> None:
        self._w_sim = weight_similarity
        self._w_rec = weight_recency
        self._w_imp = weight_importance
        self._w_rel = weight_relevance
        self._halflife = recency_halflife_days

    def score(
        self,
        memory: Memory,
        similarity: float,
        conversation_id: str | None = None,
    ) -> RetrievalResult:
        recency = self._recency_score(memory)
        imp = memory.importance
        relevance = self._conversation_relevance(memory, conversation_id)

        combined = (
            self._w_sim * similarity
            + self._w_rec * recency
            + self._w_imp * imp
            + self._w_rel * relevance
        )

        return RetrievalResult(
            memory=memory,
            score=min(combined, 1.0),
            similarity=similarity,
            recency_bonus=recency,
            importance_bonus=imp,
        )

    def score_batch(
        self,
        memories: list[tuple[Memory, float]],
        conversation_id: str | None = None,
    ) -> list[RetrievalResult]:
        results = [self.score(m, sim, conversation_id) for m, sim in memories]
        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i
        return results

    def _recency_score(self, memory: Memory) -> float:
        age = (datetime.now(UTC) - memory.last_accessed_at).total_seconds()
        age_days = age / 86400.0
        decay = math.exp(-math.log(2) * age_days / self._halflife)
        return decay

    @staticmethod
    def _conversation_relevance(memory: Memory, conversation_id: str | None) -> float:
        if conversation_id is None or memory.conversation_id is None:
            return 0.0
        return 1.0 if memory.conversation_id == conversation_id else 0.0
