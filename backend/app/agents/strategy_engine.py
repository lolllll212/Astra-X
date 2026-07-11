from __future__ import annotations

from typing import TYPE_CHECKING

from app.agents.models.strategy import Strategy
from app.agents.learning_store import GoalDomain, classify_goal
from app.core.logging import get_logger
from app.memory.world_model import EntityType

if TYPE_CHECKING:
    from app.agents.learning_manager import LearningManager
    from app.memory.experience_graph import ExperienceGraph
    from app.memory.world_model import WorldModel
    from app.llm.provider_metrics import ProviderMetricsTracker

logger = get_logger(__name__)


class StrategyEngine:
    """Fuses data from Patterns, Experience Graph, World Model, and Provider
    Metrics Tracker into a single structured :class:`Strategy`.

    The Planner consumes this directly instead of ad-hoc text.
    """

    def __init__(
        self,
        learning_manager: LearningManager | None = None,
        experience_graph: ExperienceGraph | None = None,
        world_model: WorldModel | None = None,
        metrics_tracker: ProviderMetricsTracker | None = None,
    ) -> None:
        self._learning_manager = learning_manager
        self._experience_graph = experience_graph
        self._world_model = world_model
        self._metrics_tracker = metrics_tracker

    async def build_strategy(self, goal: str) -> Strategy:
        """Build a :class:`Strategy` for *goal* by querying all available sources.

        Each source is optional — if a component is unavailable the engine
        gracefully degrades to whatever data is present.
        """
        domain = classify_goal(goal).value
        patterns = self._load_patterns(goal, domain)
        patterns_context = self._load_patterns_context(goal)
        wf = self._load_workflow(domain)
        prov = self._load_provider_recommendation(goal, domain)
        world_context = self._load_world_context(goal)
        warnings = self._load_warnings(goal)

        # --- Fuse into a single strategy ---
        best_pattern = patterns[0] if patterns else None
        strategy_summary = patterns_context or ""

        # Provider: prefer provider metrics, then pattern, then workflow.
        recommended_provider = prov.get("provider_id") or (
            best_pattern.preferred_provider if best_pattern else None
        ) or wf.get("most_common_provider")
        recommended_model = prov.get("model_id") or (
            best_pattern.preferred_model if best_pattern else None
        )

        # Tool sequence from best pattern or workflow.
        tool_seq: list[str] = []
        if best_pattern and best_pattern.tool_sequence:
            tool_seq = list(best_pattern.tool_sequence)
        elif wf.get("recommended_approach"):
            tool_seq = list(wf["recommended_approach"])

        # Expected success rate: average of matched patterns.
        success_rate = 0.0
        latency_ms = 0.0
        confidence = 0.0
        n_patterns = len(patterns)
        if patterns:
            success_rate = sum(
                p.success_count / max(p.total_count, 1) for p in patterns
            ) / n_patterns
            latency_ms = sum(
                p.avg_execution_cost_ms for p in patterns
            ) / n_patterns
            confidence = sum(
                p.avg_confidence for p in patterns
            ) / n_patterns
        elif wf.get("avg_success_rate"):
            success_rate = wf["avg_success_rate"]
            n_traj = wf.get("total_trajectories", 0)
            confidence = min(success_rate, n_traj / 10.0)

        # Fallback from provider metrics runner-up.
        fallback_provider, fallback_model = self._load_fallback(goal, domain)

        return Strategy(
            goal=goal,
            goal_domain=domain,
            strategy_summary=strategy_summary,
            recommended_provider=recommended_provider,
            recommended_model=recommended_model,
            recommended_tool_sequence=tool_seq,
            expected_success_rate=success_rate,
            expected_latency_ms=latency_ms,
            confidence=confidence,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            world_context=world_context,
            warnings=warnings,
            source_pattern_count=n_patterns,
            source="strategy_engine",
        )

    # ------------------------------------------------------------------
    # Private helpers — each source returns a partial view
    # ------------------------------------------------------------------

    def _load_patterns(self, goal: str, domain: str) -> list:
        """Load top patterns from LearningManager."""
        if self._learning_manager is None:
            return []
        return self._learning_manager.get_patterns_for_domain(domain)

    def _load_patterns_context(self, goal: str) -> str:
        """Return the strategy summary from the best matching pattern."""
        if self._learning_manager is None:
            return ""
        patterns = self._learning_manager.get_patterns_for_domain(
            classify_goal(goal).value
        )
        if patterns:
            return patterns[0].strategy_summary
        return ""

    def _load_workflow(self, domain: str) -> dict:
        """Load best workflow from ExperienceGraph."""
        if self._experience_graph is None:
            return {}
        return self._experience_graph.best_workflow_for_domain(domain)

    def _load_provider_recommendation(
        self, goal: str, domain: str
    ) -> dict:
        """Load best provider+model from ProviderMetricsTracker."""
        if self._metrics_tracker is None:
            return {}
        result = self._metrics_tracker.best_for_profile()
        if result is None:
            return {}
        provider_id, model_id = result
        return {"provider_id": provider_id, "model_id": model_id}

    def _load_world_context(self, goal: str) -> str:
        """Load entity states and valid next transitions from the world model."""
        if self._world_model is None:
            return ""
        projects = self._world_model.find_entities(EntityType.PROJECT)
        lines: list[str] = []
        for proj in projects:
            state = proj.current_state
            if state:
                valid = self._world_model.get_valid_next_states(proj.id)
                line = f"  - {proj.name} (id={proj.id}): current_state={state}"
                if valid:
                    line += f", valid_next_states={sorted(valid)}"
                lines.append(line)
        if not lines:
            return ""
        return "Entity states:\n" + "\n".join(lines)

    def _load_warnings(self, goal: str) -> list[str]:
        """Load anti-pattern warnings from LearningManager."""
        if self._learning_manager is None:
            return []
        # get_lessons already includes anti-patterns as text; we need
        # the raw AntiPattern objects.
        anti = self._learning_manager._search_anti_patterns(goal)
        return [f"{ap.warning} — Instead: {ap.suggestion}" if ap.suggestion else ap.warning for ap in anti]

    def _load_fallback(
        self, goal: str, domain: str
    ) -> tuple[str | None, str | None]:
        """Return the second-best provider+model as fallback."""
        if self._metrics_tracker is None:
            return None, None
        all_stats = self._metrics_tracker.list_all_stats()
        eligible = [s for s in all_stats if s.success_rate >= 0.5]
        eligible.sort(key=lambda s: -s.success_rate)
        if len(eligible) < 2:
            return None, None
        return eligible[1].provider_id, eligible[1].model_id
