from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from app.agents.learning_store import LearningStore
from app.agents.models.execution import ExecutionResult, ReflectionResult
from app.agents.models.pattern import ExecutionPattern
from app.agents.models.plan import Plan
from app.agents.models.task import TaskStatus
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.database.repositories.pattern_repository import PatternRepository

logger = get_logger(__name__)


class LearningManager:
    """Manages the extraction, storage, and retrieval of execution patterns.

    After a successful agent run, :meth:`extract_pattern` creates or updates
    a pattern capturing what strategy worked. Before a new run,
    :meth:`get_lessons` retrieves relevant patterns formatted for the planner
    prompt.
    """

    def __init__(
        self,
        repository: PatternRepository | None = None,
    ) -> None:
        self._store = LearningStore(repository=repository)

    async def initialize(self) -> None:
        """Load existing patterns from the database."""
        await self._store.load_all()

    # ------------------------------------------------------------------
    # Extraction — after a successful execution
    # ------------------------------------------------------------------

    async def extract_pattern(
        self,
        *,
        goal: str,
        plan: Plan,
        results: list[ExecutionResult],
        reflections: list[ReflectionResult],
    ) -> ExecutionPattern | None:
        """Extract a learning pattern from a completed execution.

        Only extracts if the overall execution was successful (no critical
        failures and average reflection confidence >= 0.5).
        """
        if not results:
            return None

        all_ok = all(
            r.status is TaskStatus.COMPLETED for r in results
        )
        if not all_ok:
            return None

        avg_conf = (
            sum(r.confidence for r in reflections) / len(reflections)
            if reflections
            else 0.0
        )
        if avg_conf < 0.5:
            return None

        goal_pattern = self._normalize_goal(goal)
        tags = self._extract_tags(goal)
        caps_used = [
            r.tool_name for r in results if r.tool_name
        ]
        primary_cap = caps_used[0] if caps_used else None
        strategy = self._build_strategy_summary(plan, results)
        plan_tpl = self._build_plan_template(plan)
        avg_imp = sum(
            r.metadata.get("importance", 0.5) for r in results
        ) / len(results)

        existing = self._store.get_by_goal_pattern(goal_pattern)
        if existing is not None:
            merged = existing.model_copy(
                update={
                    "strategy_summary": strategy,
                    "plan_template": plan_tpl,
                    "tags": list(set(existing.tags + tags)),
                    "success_count": existing.success_count + 1,
                    "total_count": existing.total_count + 1,
                    "avg_confidence": (existing.avg_confidence * (existing.total_count - 1) + avg_conf)
                    / existing.total_count,
                    "avg_importance": (existing.avg_importance * (existing.total_count - 1) + avg_imp)
                    / existing.total_count,
                    "last_success_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
            )
        else:
            merged = ExecutionPattern(
                id=str(uuid4()),
                goal_pattern=goal_pattern,
                capability=primary_cap,
                strategy_summary=strategy,
                plan_template=plan_tpl,
                tags=tags,
                success_count=1,
                total_count=1,
                avg_confidence=avg_conf,
                avg_importance=avg_imp,
                last_success_at=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        saved = await self._store.save(merged)
        logger.info(
            "learning_manager.pattern_extracted",
            goal_pattern=goal_pattern,
            total_count=saved.total_count,
            success_count=saved.success_count,
        )
        return saved

    # ------------------------------------------------------------------
    # Retrieval — before planning
    # ------------------------------------------------------------------

    async def get_lessons(self, goal: str, max_patterns: int = 3) -> str:
        """Format relevant patterns as text for the planner prompt.

        Returns an empty string if no relevant patterns are found.
        """
        patterns = self._store.search(goal, limit=max_patterns)
        if not patterns:
            return ""

        lines: list[str] = [
            "Lessons learned from past executions:",
        ]
        for i, p in enumerate(patterns, 1):
            lines.append("")
            lines.append(f"{i}. Pattern: {p.goal_pattern}")
            lines.append(f"   Strategy: {p.strategy_summary}")
            lines.append(f"   Success: {p.success_count}/{p.total_count} times")
            if p.tags:
                lines.append(f"   Tags: {', '.join(p.tags)}")
            if p.plan_template:
                lines.append("   Plan template:")
                for line in p.plan_template.strip().split("\n"):
                    lines.append(f"     {line.strip()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_goal(goal: str) -> str:
        """Normalize a goal to a reusable pattern string.

        Replaces specific entities with placeholders:
        - Numbers → {n}
        - Quoted strings → {value}
        - Makes lowercase
        """
        import re
        g = goal.lower().strip()
        g = re.sub(r'"([^"]*)"', "{value}", g)
        g = re.sub(r"'([^']*)'", "{value}", g)
        g = re.sub(r"\b\d+\b", "{n}", g)
        g = re.sub(r"\b(a|an|the|some|any)\s+", "", g)
        g = re.sub(r"\s+", " ", g).strip()
        if len(g) > 200:
            g = g[:200]
        return g

    @staticmethod
    def _extract_tags(goal: str) -> list[str]:
        """Extract meaningful tags from a goal."""
        from app.agents.learning_store import LearningStore
        return LearningStore._extract_keywords(goal)

    @staticmethod
    def _build_strategy_summary(
        plan: Plan,
        results: list[ExecutionResult],
    ) -> str:
        """Describe what strategy worked."""
        caps = [
            r.tool_name for r in results if r.tool_name
        ]
        if caps:
            return f"Used {', '.join(sorted(set(caps)))} in sequence to accomplish: {plan.goal[:100]}"
        return f"Completed {len(plan.tasks)} task(s) via LLM reasoning: {plan.goal[:100]}"

    @staticmethod
    def _build_plan_template(plan: Plan) -> str:
        """Build a template from the task descriptions."""
        lines = []
        for t in plan.tasks:
            cap = f" [{t.capability}]" if t.capability else ""
            deps = f" (after: {', '.join(t.dependencies)})" if t.dependencies else ""
            lines.append(f"- {t.description}{cap}{deps}")
        return "\n".join(lines)
