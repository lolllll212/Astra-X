from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.agents.models.pattern import ExecutionPattern
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.database.repositories.pattern_repository import PatternRepository

logger = get_logger(__name__)


class LearningStore:
    """Persistent store for execution patterns with in-memory cache.

    Patterns are loaded from the database on startup and kept in memory
    for fast retrieval. Mutations are written through to the database.
    """

    def __init__(self, repository: PatternRepository | None = None) -> None:
        self._repo = repository
        self._cache: dict[str, ExecutionPattern] = {}

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
        """Find patterns relevant to *goal* using keyword matching.

        Extracts meaningful keywords from the goal and matches against
        stored pattern tags and goal_pattern fields. Results are sorted
        by success ratio * frequency.
        """
        keywords = self._extract_keywords(goal)
        if not keywords:
            return []

        scored: list[tuple[ExecutionPattern, float]] = []
        for pattern in self._cache.values():
            score = self._match_score(pattern, keywords)
            if score > 0:
                scored.append((pattern, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:limit]]

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

    @staticmethod
    def _match_score(pattern: ExecutionPattern, keywords: list[str]) -> float:
        """Score how well a pattern matches a set of keywords."""
        score = 0.0
        pattern_text = (
            pattern.goal_pattern.lower()
            + " "
            + " ".join(pattern.tags).lower()
            + " "
            + pattern.strategy_summary.lower()
        )
        for kw in keywords:
            if kw in pattern_text:
                score += 1.0
        if score == 0:
            return 0.0
        match_ratio = score / len(keywords)
        success_ratio = pattern.success_count / max(pattern.total_count, 1)
        return match_ratio * 0.6 + success_ratio * 0.4
