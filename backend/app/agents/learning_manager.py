from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from app.agents.learning_store import LearningStore, GoalDomain, classify_goal
from app.agents.models.execution import ExecutionResult, ReflectionResult
from app.agents.models.pattern import AntiPattern, ExecutionPattern
from app.agents.models.plan import Plan
from app.agents.models.task import TaskStatus
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.database.repositories.pattern_repository import PatternRepository
    from app.memory.experience_graph import ExperienceGraph
    from app.memory.manager import MemoryManager as CoreMemoryManager
    from app.memory.world_model import WorldModel

from app.domain.enums import MemoryScope
from app.memory.models.memory import MemoryType

logger = get_logger(__name__)


class LearningManager:
    """Manages the extraction, storage, and retrieval of execution patterns.

    After a successful agent run, :meth:`extract_pattern` creates or updates
    a pattern capturing what strategy worked. Before a new run,
    :meth:`get_lessons` retrieves relevant patterns formatted for the planner
    prompt along with recommended task sequences and anti-pattern warnings.

    When a ``memory_manager`` is provided, extracted patterns are also
    persisted as **procedural memories** in the core memory hierarchy.
    The optional ``world_model`` and ``experience_graph`` enable richer
    cross-session reasoning.
    """

    def __init__(
        self,
        repository: PatternRepository | None = None,
        memory_manager: CoreMemoryManager | None = None,
        world_model: WorldModel | None = None,
        experience_graph: ExperienceGraph | None = None,
    ) -> None:
        self._store = LearningStore(repository=repository)
        self._memory_manager = memory_manager
        self._world_model = world_model
        self._experience_graph = experience_graph
        # In-memory anti-pattern cache (no DB model yet).
        self._anti_patterns: dict[str, AntiPattern] = {}

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

        Uses generalised goal templates so patterns match across related
        tasks (e.g. "build FastAPI auth" and "build Django auth" both
        match the template "build {technology} auth").
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

        # Use generalized goal template for broader matching.
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

        # Derive preferred provider / model / tool sequence from results.
        preferred_provider = None
        preferred_model = None
        tool_seq = list(dict.fromkeys(
            r.tool_name for r in results if r.tool_name
        ))

        for r in results:
            pp = r.metadata.get("provider_id")
            pm = r.metadata.get("model_id")
            if pp:
                preferred_provider = pp
            if pm:
                preferred_model = pm

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
            merged_tool_seq = list(dict.fromkeys(existing.tool_sequence + tool_seq))
            merged = existing.model_copy(
                update={
                    "strategy_summary": improved_strategy,
                    "plan_template": improved_plan,
                    "preferred_provider": preferred_provider or existing.preferred_provider,
                    "preferred_model": preferred_model or existing.preferred_model,
                    "tool_sequence": merged_tool_seq,
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
                preferred_provider=preferred_provider,
                preferred_model=preferred_model,
                tool_sequence=tool_seq,
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

        # Record in the Experience Graph for trajectory-level queries.
        if self._experience_graph is not None and plan.tasks:
            self._experience_graph.record(
                goal=goal,
                goal_domain=classify_goal(goal).value,
                plan=plan,
                results=results,
                reflections=reflections,
                provider_id=preferred_provider or "",
                model_id=preferred_model or "",
                overall_decision="accept" if avg_conf >= 0.5 else "retry",
            )

        logger.info(
            "learning_manager.pattern_extracted",
            goal_pattern=goal_pattern,
            total_count=saved.total_count,
            success_count=saved.success_count,
        )
        return saved

    # ------------------------------------------------------------------
    # Negative learning — record and retrieve anti-patterns
    # ------------------------------------------------------------------

    def record_failure(
        self,
        *,
        goal: str,
        warning: str,
        failure_reason: str = "",
        suggestion: str | None = None,
    ) -> AntiPattern:
        """Record an anti-pattern — what to avoid and why.

        If the same *goal_pattern* (normalised) already has an anti-pattern,
        the occurrence count is incremented and the warning is updated.
        """
        goal_pattern = self._normalize_goal(goal)

        existing = self._anti_patterns.get(goal_pattern)
        if existing is not None:
            updated = existing.model_copy(
                update={
                    "warning": warning,
                    "occurrence_count": existing.occurrence_count + 1,
                    "suggestion": suggestion or existing.suggestion,
                    "last_seen_at": datetime.now(),
                    "updated_at": datetime.now(),
                },
            )
            self._anti_patterns[goal_pattern] = updated
            return updated

        ap = AntiPattern(
            id=str(uuid4()),
            goal_pattern=goal_pattern,
            warning=warning,
            failure_reason=failure_reason,
            suggestion=suggestion,
            tags=self._extract_tags(goal),
            last_seen_at=datetime.now(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self._anti_patterns[goal_pattern] = ap
        return ap

    # ------------------------------------------------------------------
    # Retrieval — before planning
    # ------------------------------------------------------------------

    async def get_lessons(
        self,
        goal: str,
        max_patterns: int = 3,
    ) -> str:
        """Format relevant patterns as text for the planner prompt.

        Includes:
        - Top matching patterns with success rates, confidence, cost.
        - Recommended task sequence from the best-matching pattern.
        - Anti-pattern warnings for pitfalls to avoid.

        Returns an empty string if nothing relevant is found.
        """
        patterns = self._store.search(goal, limit=max_patterns)
        anti_patterns = self._search_anti_patterns(goal)

        if not patterns and not anti_patterns:
            return ""

        lines: list[str] = []

        # --- Recommended plan (structured planner hint) ---
        if patterns:
            best = patterns[0]
            if best.plan_template:
                lines.append("Recommended task sequence (based on past success):")
                for line in best.plan_template.strip().split("\n"):
                    lines.append(f"  {line.strip()}")
                lines.append("")

        # --- Pattern details ---
        if patterns:
            lines.append("Lessons learned from past executions:")
            for i, p in enumerate(patterns, 1):
                success_rate = p.success_count / max(p.total_count, 1)
                cost_s = p.avg_execution_cost_ms / 1000.0
                lines.append("")
                lines.append(f"{i}. Pattern: {p.goal_pattern}")
                lines.append(f"   Strategy: {p.strategy_summary}")
                lines.append(f"   Success: {p.success_count}/{p.total_count} ({success_rate:.0%})")
                lines.append(f"   Confidence: {p.last_reflection_confidence:.2f}")
                lines.append(f"   Avg cost: {cost_s:.1f}s")
                if p.preferred_provider:
                    lines.append(f"   Best provider: {p.preferred_provider}")
                if p.preferred_model:
                    lines.append(f"   Best model: {p.preferred_model}")
                if p.tool_sequence:
                    lines.append(f"   Tool sequence: {' → '.join(p.tool_sequence)}")
                if p.tags:
                    lines.append(f"   Tags: {', '.join(p.tags)}")

        # --- Anti-pattern warnings ---
        if anti_patterns:
            if patterns:
                lines.append("")
            lines.append("Warnings — approaches that have failed before:")
            for i, ap in enumerate(anti_patterns, 1):
                lines.append(f"  {i}. {ap.warning}")
                if ap.suggestion:
                    lines.append(f"     Instead: {ap.suggestion}")
                lines.append(f"     Occurrences: {ap.occurrence_count}")

        return "\n".join(lines)

    def _search_anti_patterns(
        self,
        goal: str,
        limit: int = 3,
    ) -> list[AntiPattern]:
        """Find anti-patterns relevant to *goal* via keyword matching."""
        keywords = LearningStore._extract_keywords(goal)
        if not keywords:
            return []

        scored: list[tuple[AntiPattern, float]] = []
        for ap in self._anti_patterns.values():
            text = (
                ap.goal_pattern.lower()
                + " "
                + " ".join(ap.tags).lower()
                + " "
                + ap.warning.lower()
            )
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                scored.append((ap, hits / len(keywords)))

        scored.sort(key=lambda x: -x[1])
        return [ap for ap, _ in scored[:limit]]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_goal(goal: str) -> str:
        """Normalize a goal to a reusable, generalised pattern string.

        Delegates to :meth:`LearningStore.generalize_goal` which replaces
        known technology names with ``{technology}``, numbers with ``{n}``,
        and quoted strings with ``{value}``.

        Example::
            "Build FastAPI authentication for 2 users"
            → "build {technology} authentication for {n} users"
        """
        return LearningStore.generalize_goal(goal)

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

        success_reflections = [
            r for r in reflections if r.decision.value == "accept" and r.confidence > 0.7
        ]
        if success_reflections:
            return new_plan

        return existing_plan

    # ------------------------------------------------------------------
    # Experience Graph shortcuts
    # ------------------------------------------------------------------

    def best_workflow_for_domain(self, domain: str) -> str:
        """Return a human-readable best-workflow summary from the
        :class:`ExperienceGraph`, or an empty string if unavailable."""
        if self._experience_graph is None:
            return ""
        result = self._experience_graph.best_workflow_for_domain(domain)
        if not result:
            return ""
        lines = [
            f"Domain: {result['domain']}",
            f"Successful runs: {result['total_trajectories']}",
            f"Avg success rate: {result['avg_success_rate']:.0%}",
        ]
        if result.get("most_common_plan"):
            lines.append(f"Most common plan: {result['most_common_plan']}")
        if result.get("most_common_provider"):
            lines.append(f"Preferred provider: {result['most_common_provider']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Eval feedback bridge helpers
    # ------------------------------------------------------------------

    def get_patterns_for_domain(self, domain: str) -> list[ExecutionPattern]:
        """Return all cached patterns that match a domain."""
        all_patterns = self._store.search("", limit=999, domain=GoalDomain(domain))
        return all_patterns

    def get_weights_for_domain(self, domain: str) -> dict[str, float]:
        """Return the current ranking weights for a domain."""
        from app.agents.learning_store import DOMAIN_WEIGHTS

        try:
            d = GoalDomain(domain)
        except ValueError:
            d = GoalDomain.GENERAL
        return dict(DOMAIN_WEIGHTS.get(d, {}))

    def adjust_weights(self, domain: str, delta: dict[str, float]) -> None:
        """Apply a delta adjustment to per-domain ranking weights.

        Args:
            domain: The domain key (e.g. ``"coding"``).
            delta: Weight adjustments, e.g. ``{"success_rate": 0.015}``.
        """
        from app.agents.learning_store import DOMAIN_WEIGHTS

        try:
            d = GoalDomain(domain)
        except ValueError:
            d = GoalDomain.GENERAL

        weights = DOMAIN_WEIGHTS.get(d)
        if weights is None:
            return

        for key, adjustment in delta.items():
            if key in weights:
                new_val = weights[key] + adjustment
                weights[key] = max(0.0, min(1.0, new_val))

        logger.info(
            "learning_manager.weights_adjusted",
            domain=domain,
            delta=delta,
            new_weights=weights,
        )
