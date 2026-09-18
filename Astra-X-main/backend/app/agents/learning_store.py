from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

from app.agents.models.pattern import ExecutionPattern
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.database.repositories.pattern_repository import PatternRepository

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Goal domain classification
# ---------------------------------------------------------------------------


class GoalDomain(StrEnum):
    CODING = "coding"
    RESEARCH = "research"
    REASONING = "reasoning"
    VISION = "vision"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Per-domain weight presets
# ---------------------------------------------------------------------------

DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    GoalDomain.CODING: {
        "similarity": 0.20,
        "success_rate": 0.30,
        "recency": 0.15,
        "cost": 0.20,
        "confidence": 0.15,
    },
    GoalDomain.RESEARCH: {
        "similarity": 0.40,
        "success_rate": 0.20,
        "recency": 0.20,
        "cost": 0.05,
        "confidence": 0.15,
    },
    GoalDomain.REASONING: {
        "similarity": 0.15,
        "success_rate": 0.30,
        "recency": 0.10,
        "cost": 0.10,
        "confidence": 0.35,
    },
    GoalDomain.VISION: {
        "similarity": 0.30,
        "success_rate": 0.25,
        "recency": 0.10,
        "cost": 0.10,
        "confidence": 0.25,
    },
    GoalDomain.GENERAL: {
        "similarity": 0.30,
        "success_rate": 0.25,
        "recency": 0.20,
        "cost": 0.10,
        "confidence": 0.15,
    },
}

# Default (backward-compatible) weights.
DEFAULT_WEIGHTS: dict[str, float] = dict(DOMAIN_WEIGHTS[GoalDomain.GENERAL])

# Domain-detection keywords.
_CODING_KEYWORDS: frozenset[str] = frozenset({
    "code", "api", "app", "function", "class", "test", "debug", "deploy",
    "database", "sql", "python", "javascript", "typescript", "react",
    "fastapi", "django", "flask", "docker", "git", "frontend", "backend",
    "fullstack", "authentication", "authorization", "endpoint", "route",
    "migration", "schema", "query", "mutation", "graphql", "rest",
    "sdk", "library", "framework", "build", "compile", "refactor",
    "implement", "integration", "continuous", "pipeline", "ci", "cd",
})
_RESEARCH_KEYWORDS: frozenset[str] = frozenset({
    "research", "search", "find", "lookup", "investigate", "explore",
    "survey", "study", "analyze", "compare", "review", "read", "learn",
    "understand", "what", "why", "how", "when", "where", "who",
    "explain", "summarize", "overview", "background", "related",
    "literature", "paper", "article", "documentation", "docs",
    "tutorial", "guide", "example", "best practice",
})
_REASONING_KEYWORDS: frozenset[str] = frozenset({
    "reason", "logic", "solve", "puzzle", "math", "equation", "proof",
    "deduce", "infer", "conclude", "think", "plan", "strategy",
    "optimize", "evaluate", "assess", "decide", "choose", "select",
    "tradeoff", "budget", "cost", "benefit", "risk", "scenario",
})
_VISION_KEYWORDS: frozenset[str] = frozenset({
    "image", "photo", "picture", "diagram", "chart", "graph", "visual",
    "vision", "detect", "recognize", "classify", "segment", "ocr",
    "object", "face", "scene", "caption", "generate image", "draw",
    "illustration", "screenshot", "ui", "ux", "design", "layout",
})


def classify_goal(goal: str) -> GoalDomain:
    """Detect the primary domain of a goal based on keyword heuristics."""
    lower = goal.lower()
    coding = sum(1 for kw in _CODING_KEYWORDS if kw in lower)
    research = sum(1 for kw in _RESEARCH_KEYWORDS if kw in lower)
    reasoning = sum(1 for kw in _REASONING_KEYWORDS if kw in lower)
    vision = sum(1 for kw in _VISION_KEYWORDS if kw in lower)

    # Weight vision higher since image tasks often also contain research/coding words.
    if vision >= 1:
        return GoalDomain.VISION
    if coding >= 2:
        return GoalDomain.CODING
    if research >= 2:
        return GoalDomain.RESEARCH
    if reasoning >= 2:
        return GoalDomain.REASONING
    # Single-match tiebreaker.
    scores = {
        GoalDomain.CODING: coding,
        GoalDomain.RESEARCH: research,
        GoalDomain.REASONING: reasoning,
    }
    best = max(scores, key=scores.get)
    if scores[best] >= 1:
        return best
    return GoalDomain.GENERAL


# ---------------------------------------------------------------------------
# LearningStore
# ---------------------------------------------------------------------------


class LearningStore:
    """Persistent store for execution patterns with in-memory cache.

    Patterns are loaded from the database on startup and kept in memory
    for fast retrieval. Mutations are written through to the database.

    Ranking is domain-adaptive: the :meth:`search` method auto-classifies
    the goal and uses per-domain weight presets defined in :data:`DOMAIN_WEIGHTS`.
    """

    def __init__(
        self,
        repository: PatternRepository | None = None,
    ) -> None:
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

    def search(
        self,
        goal: str,
        limit: int = 5,
        domain: GoalDomain | None = None,
    ) -> list[ExecutionPattern]:
        """Find patterns relevant to *goal* using domain-adaptive ranking.

        Each pattern is scored on five axes with weights selected based on
        the goal's domain (auto-detected unless *domain* is provided):

        +--------------+---------+-----------+-----------+-------+------------+
        | Domain       | Similar | Success   | Recency   | Cost  | Confidence |
        +--------------+---------+-----------+-----------+-------+------------+
        | Coding       |  0.20   |  0.30     |  0.15     |  0.20 |  0.15      |
        | Research     |  0.40   |  0.20     |  0.20     |  0.05 |  0.15      |
        | Reasoning    |  0.15   |  0.30     |  0.10     |  0.10 |  0.35      |
        | Vision       |  0.30   |  0.25     |  0.10     |  0.10 |  0.25      |
        | General      |  0.30   |  0.25     |  0.20     |  0.10 |  0.15      |
        +--------------+---------+-----------+-----------+-------+------------+
        """
        domain = domain or classify_goal(goal)
        weights = DOMAIN_WEIGHTS.get(domain, DEFAULT_WEIGHTS)

        keywords = self._extract_keywords(goal)
        if not keywords:
            return []

        scored: list[tuple[ExecutionPattern, float]] = []
        for pattern in self._cache.values():
            score = self._rank_score(pattern, keywords, weights)
            if score > 0:
                scored.append((pattern, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Multi-factor ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_score(
        pattern: ExecutionPattern,
        keywords: list[str],
        weights: dict[str, float],
    ) -> float:
        """Compute the combined ranking score for *pattern* against *keywords*."""
        sim = LearningStore._factor_similarity(pattern, keywords)
        if sim == 0:
            return 0.0

        sr = LearningStore._factor_success_rate(pattern)
        rec = LearningStore._factor_recency(pattern)
        cost = LearningStore._factor_cost(pattern)
        conf = LearningStore._factor_confidence(pattern)

        return (
            sim * weights.get("similarity", 0.30)
            + sr * weights.get("success_rate", 0.25)
            + rec * weights.get("recency", 0.20)
            + cost * weights.get("cost", 0.10)
            + conf * weights.get("confidence", 0.15)
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
    # Keyword extraction
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


    # ------------------------------------------------------------------
    # Goal generalization
    # ------------------------------------------------------------------

    @staticmethod
    def generalize_goal(goal: str) -> str:
        """Build an abstract template from a concrete goal.

        Detects technology names (frameworks, languages, tools) using a
        known-entity lexicon and replaces them with ``{technology}``.
        Also replaces numbers with ``{n}`` and quoted strings with
        ``{value}``, producing a reusable pattern like
        ``"build {technology} authentication"``.
        """
        lower = goal.lower().strip()

        # Known technology / framework / tool names (lowercase).
        technologies: frozenset[str] = frozenset({
            "fastapi", "django", "flask", "express", "spring", "rails",
            "laravel", "nextjs", "nuxt", "remix", "sveltekit", "astro",
            "react", "vue", "angular", "svelte", "solid", "jquery",
            "tailwind", "bootstrap", "materialui", "chakra", "shadcn",
            "python", "javascript", "typescript", "rust", "go", "golang",
            "java", "kotlin", "scala", "ruby", "php", "csharp", "c++",
            "swift", "kotlin", "docker", "kubernetes", "k8s", "aws",
            "gcp", "azure", "terraform", "ansible", "pulumi",
            "postgres", "postgresql", "mysql", "sqlite", "mongodb",
            "redis", "elasticsearch", "kafka", "rabbitmq",
            "pytorch", "tensorflow", "jax", "langchain", "llamaindex",
            "openai", "claude", "gemini", "mistral", "llama",
            "graphql", "rest", "grpc", "websocket",
            "linux", "ubuntu", "debian", "alpine", "nginx", "apache",
            "vscode", "neovim", "vim", "emacs", "intellij", "pycharm",
            "github", "gitlab", "bitbucket", "jira", "confluence",
        })

        # Sort by length descending so longer names match before substrings.
        sorted_techs = sorted(technologies, key=len, reverse=True)
        for tech in sorted_techs:
            # Use word-boundary replacement so "react" doesn't match "reactive".
            lower = re.sub(rf"\b{re.escape(tech)}\b", "{technology}", lower)

        # Existing normalizations.
        lower = re.sub(r'"([^"]*)"', "{value}", lower)
        lower = re.sub(r"'([^']*)'", "{value}", lower)
        lower = re.sub(r"\b\d+\b", "{n}", lower)
        lower = re.sub(r"\b(a|an|the|some|any)\s+", "", lower)
        lower = re.sub(r"\s+", " ", lower).strip()

        if len(lower) > 200:
            lower = lower[:200]
        return lower
