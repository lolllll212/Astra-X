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
    from app.memory.manager import MemoryManager as CoreMemoryManager

from app.domain.enums import MemoryScope
from app.memory.models.memory import MemoryType

logger = get_logger(__name__)


class LearningManager:
    """Manages the extraction, storage, and retrieval of execution patterns.

    After a successful agent run, :meth:`extract_pattern` creates or updates
    a pattern capturing what strategy worked. Before a new run,
    :meth:`get_lessons` retrieves relevant patterns formatted for the planner
    prompt.

    When a ``memory_manager`` is provided, extracted patterns are also
    persisted as **procedural memories** in the core memory hierarchy,
    enabling cross-session retrieval alongside episodic and semantic
    memories.
    """

    def __init__(
        self,
        repository: PatternRepository | None = None,
        memory_manager: CoreMemoryManager | None = None,
    ) -> None:
        self._store = LearningStore(repository=repository)
        self._memory_manager = memory_manager

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

        total_cost_ms = sum(
            r.metadata.get("execution_cost_ms", 0) for r in results
        )

        existing = self._store.get_by_goal_pattern(goal_pattern)
        if existing is not None:
            n = existing.total_count
            merged_cost = (
                (existing.avg_execution_cost_ms * (n - 1) + total_cost_ms) / n
            )
            improved_strategy = self._improve_strategy(
                existing_strategy=existing.strategy_summary,
                new_strategy=strategy,
                existing_success_rate=existing.success_count / max(existing.total_count, 1),
                current_confidence=avg_conf,
                reflections=reflections,
            )
            improved_plan = self._improve_plan_template(
                existing_plan=existing.plan_template,
                new_plan=plan_tpl,
                reflections=reflections,
            )
            merged = existing.model_copy(
                update={
                    "strategy_summary": improved_strategy,
                    "plan_template": improved_plan,
                    "tags": list(set(existing.tags + tags)),
                    "success_count": existing.success_count + 1,
                    "total_count": n + 1,
                    "avg_confidence": (existing.avg_confidence * (n - 1) + avg_conf)
                    / n,
                    "avg_importance": (existing.avg_importance * (n - 1) + avg_imp)
                    / n,
                    "avg_execution_cost_ms": merged_cost,
                    "last_reflection_confidence": avg_conf,
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
                avg_execution_cost_ms=total_cost_ms,
                last_reflection_confidence=avg_conf,
                last_success_at=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        saved = await self._store.save(merged)

        # Bridge into core memory hierarchy as a PROCEDURAL memory.
        if self._memory_manager is not None:
            success_rate = saved.success_count / max(saved.total_count, 1)
            cost_s = saved.avg_execution_cost_ms / 1000.0
            memory_content = (
                f"Pattern: {saved.goal_pattern}\n"
                f"Strategy: {saved.strategy_summary}\n"
                f"Success: {saved.success_count}/{saved.total_count} ({success_rate:.0%})\n"
                f"Confidence: {saved.last_reflection_confidence:.2f}\n"
                f"Avg cost: {cost_s:.1f}s\n"
                f"Tags: {', '.join(saved.tags)}"
            )
            await self._memory_manager.store(
                content=memory_content,
                memory_type=MemoryType.PROCEDURAL,
                scope=MemoryScope.GLOBAL,
                importance=saved.avg_importance,
                metadata={
                    "goal_pattern": saved.goal_pattern,
                    "capability": saved.capability or "",
                    "success_count": saved.success_count,
                    "total_count": saved.total_count,
                    "avg_confidence": saved.avg_confidence,
                    "avg_execution_cost_ms": saved.avg_execution_cost_ms,
                    "tags": saved.tags,
                    "source": "learning_manager",
                },
            )

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
            success_rate = p.success_count / max(p.total_count, 1)
            cost_s = p.avg_execution_cost_ms / 1000.0
            lines.append("")
            lines.append(f"{i}. Pattern: {p.goal_pattern}")
            lines.append(f"   Strategy: {p.strategy_summary}")
            lines.append(f"   Success: {p.success_count}/{p.total_count} ({success_rate:.0%})")
            lines.append(f"   Confidence: {p.last_reflection_confidence:.2f}")
            lines.append(f"   Avg cost: {cost_s:.1f}s")
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

    # ------------------------------------------------------------------
    # Strategy improvement (closed learning loop)
    # ------------------------------------------------------------------

    @staticmethod
    def _improve_strategy(
        existing_strategy: str,
        new_strategy: str,
        existing_success_rate: float,
        current_confidence: float,
        reflections: list[ReflectionResult],
    ) -> str:
        """Merge existing and new strategies, incorporating reflection feedback.

        Rules:
        - If the current run has high confidence (>0.8), prefer the new strategy.
        - If the existing pattern has a strong track record (>80% success),
          blend both strategies.
        - If reflections contain specific feedback, append it as a lesson.
        """
        parts: list[str] = []

        # Collect actionable feedback from reflections.
        lessons: list[str] = []
        for r in reflections:
            if r.feedback and r.confidence > 0.6:
                feedback = r.feedback.strip()
                if feedback and feedback not in lessons:
                    lessons.append(feedback)

        # Decide which strategy to keep.
        if current_confidence > 0.8 and new_strategy:
            parts.append(new_strategy)
            if existing_strategy and existing_strategy != new_strategy:
                parts.append(f"(evolved from: {existing_strategy})")
        elif existing_success_rate > 0.8 and existing_strategy:
            parts.append(existing_strategy)
            if new_strategy and new_strategy != existing_strategy:
                parts.append(
                    f"(refinement: {new_strategy})"
                )
        else:
            # Neither is clearly better — merge both.
            chosen = new_strategy or existing_strategy or ""
            parts.append(chosen)

        if lessons:
            parts.append("Lessons: " + " | ".join(lessons[:3]))

        return " | ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")

    @staticmethod
    def _improve_plan_template(
        existing_plan: str | None,
        new_plan: str | None,
        reflections: list[ReflectionResult],
    ) -> str | None:
        """Merge plan templates, preferring the more successful one.

        Keeps the plan from the version with higher confidence, but if
        reflection suggests specific improvements, appends them as notes.
        """
        if not new_plan and not existing_plan:
            return None
        if not new_plan:
            return existing_plan
        if not existing_plan:
            return new_plan

        success_reflections = [
            r for r in reflections if r.decision.value == "accept" and r.confidence > 0.7
        ]
        if success_reflections:
            return new_plan

        return existing_plan
