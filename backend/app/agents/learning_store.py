from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.agents.models.pattern import ExecutionPattern
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.database.repositories.pattern_repository import PatternRepository

logger = get_logger(__name__)

# Default ranking weights — can be overridden per-instance.
# Each weight corresponds to a scoring factor (see _rank_score docstring).
RANK_WEIGHTS: dict[str, float] = {
    "similarity": 0.30,
    "success_rate": 0.25,
    "recency": 0.20,
    "cost": 0.10,
    "confidence": 0.15,
}


class LearningStore:
    """Persistent store for execution patterns with in-memory cache.

    Patterns are loaded from the database on startup and kept in memory
    for fast retrieval. Mutations are written through to the database.

    Attributes:
        rank_weights: Factor weights for the multi-faceted ranking formula.
                      Keys: similarity, success_rate, recency, cost, confidence.
    """

    def __init__(
        self,
        repository: PatternRepository | None = None,
        rank_weights: dict[str, float] | None = None,
    ) -> None:
        self._repo = repository
        self._cache: dict[str, ExecutionPattern] = {}
        self.rank_weights = {**RANK_WEIGHTS, **(rank_weights or {})}

    async def load_all(self) -> int:
        """Load all patterns from the database into the in-memory cache."""
        if self._repo is None:
            return 0
        patterns = await self._repo.list_all()
        self._cache = {p.goal_pattern: p for p in patterns}
        logger.info("learning_store.loaded", count=len(self._cache))
        return len(self._cache)

    async def save(self, pattern: ExecutionPattern) -> ExecutionPattern:
        """Persist a pattern to the database and update the cache."""
        if self._repo is not None:
            pattern = await self._repo.upsert(pattern)
        self._cache[pattern.goal_pattern] = pattern
        return pattern

    def get_by_goal_pattern(self, goal_pattern: str) -> ExecutionPattern | None:
        return self._cache.get(goal_pattern)

    def search(self, goal: str, limit: int = 5) -> list[ExecutionPattern]:
        """Find patterns relevant to *goal* using multi-factor ranking.

        Each pattern is scored on five axes:
          1. **Similarity** — keyword match ratio against goal_pattern/tags/strategy.
          2. **Success rate** — ratio of successful uses to total uses.
          3. **Recency** — exponential decay based on days since last success.
          4. **Cost** — inverse of average execution cost (cheaper is better).
          5. **Confidence** — most recent reflection confidence.

        The final score is the weighted sum of the five factors, configurable
        via :attr:`rank_weights`. Patterns with zero similarity are excluded.
        """
        keywords = self._extract_keywords(goal)
        if not keywords:
            return []

        scored: list[tuple[ExecutionPattern, float]] = []
        for pattern in self._cache.values():
            score = self._rank_score(pattern, keywords)
            if score > 0:
                scored.append((pattern, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Multi-factor ranking
    # ------------------------------------------------------------------

    def _rank_score(self, pattern: ExecutionPattern, keywords: list[str]) -> float:
        """Compute the combined ranking score for *pattern* against *keywords*."""
        sim = self._factor_similarity(pattern, keywords)
        if sim == 0:
            return 0.0

        sr = self._factor_success_rate(pattern)
        rec = self._factor_recency(pattern)
        cost = self._factor_cost(pattern)
        conf = self._factor_confidence(pattern)

        w = self.rank_weights
        return (
            sim * w.get("similarity", 0.30)
            + sr * w.get("success_rate", 0.25)
            + rec * w.get("recency", 0.20)
            + cost * w.get("cost", 0.10)
            + conf * w.get("confidence", 0.15)
        )

    @staticmethod
    def _factor_similarity(pattern: ExecutionPattern, keywords: list[str]) -> float:
        """Keyword match ratio — 0.0 to 1.0."""
        pattern_text = (
            pattern.goal_pattern.lower()
            + " "
            + " ".join(pattern.tags).lower()
            + " "
            + pattern.strategy_summary.lower()
        )
        hits = sum(1 for kw in keywords if kw in pattern_text)
        if hits == 0:
            return 0.0
        return hits / len(keywords)

    @staticmethod
    def _factor_success_rate(pattern: ExecutionPattern) -> float:
        """Success ratio — 0.0 to 1.0."""
        return pattern.success_count / max(pattern.total_count, 1)

    @staticmethod
    def _factor_recency(pattern: ExecutionPattern) -> float:
        """Recency score using exponential decay (half-life = 7 days).

        Returns 1.0 if never updated (assumed recent), otherwise decays
        toward 0.0 as the pattern ages.
        """
        if pattern.last_success_at is None:
            return 1.0
        days_elapsed = max(
            0.0,
            (datetime.now(timezone.utc) - pattern.last_success_at).total_seconds()
            / 86400.0,
        )
        return math.exp(-days_elapsed / 7.0)

    @staticmethod
    def _factor_cost(pattern: ExecutionPattern) -> float:
        """Cost score — cheaper is better; capped at 10 s for the floor.

        Returns 1.0 for zero-cost patterns (new / unknown), otherwise
        decays as ``1 / (1 + cost_seconds)``.
        """
        cost_s = pattern.avg_execution_cost_ms / 1000.0
        if cost_s <= 0:
            return 1.0
        return 1.0 / (1.0 + cost_s)

    @staticmethod
    def _factor_confidence(pattern: ExecutionPattern) -> float:
        """Most recent reflection confidence — 0.0 to 1.0."""
        return pattern.last_reflection_confidence

    # ------------------------------------------------------------------
    # Keyword extraction (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(goal: str) -> list[str]:
        """Extract meaningful lowercase keywords from a goal string."""
        goal_lower = goal.lower()
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "because",
            "and", "but", "or", "if", "while", "about", "up", "me", "my",
            "i", "we", "you", "it", "its", "this", "that", "these", "those",
            "please", "help", "need", "want", "like", "get", "find", "make",
        }
        words = re.findall(r"[a-zA-Z]\w+", goal_lower)
        return [w for w in words if w not in stop_words and len(w) > 2]
