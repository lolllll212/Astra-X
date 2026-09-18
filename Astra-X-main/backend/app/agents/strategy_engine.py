from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.agents.learning_store import GoalDomain, classify_goal
from app.agents.models.policy import ExecutionMode
from app.agents.models.strategy import Strategy, StrategyAlternative
from app.core.logging import get_logger
from app.memory.world_model import EntityType

if TYPE_CHECKING:
    from app.agents.learning_manager import LearningManager
    from app.llm.provider_metrics import ProviderMetricsTracker
    from app.memory.experience_graph import ExperienceGraph
    from app.memory.world_model import WorldModel

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
        patterns_result = self._load_patterns(goal, domain)
        patterns = patterns_result if isinstance(patterns_result, list) else patterns_result[0] if patterns_result else []
        patterns_context = self._load_patterns_context(goal)
        wf = self._load_workflow(domain)
        prov = self._load_provider_recommendation(goal, domain)
        world_context = self._load_world_context(goal)
        warnings = self._load_warnings(goal)
        recommended_mode = self._load_recommended_mode(goal, domain, patterns)

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

        # Calculate confidence with decay and build alternative strategies
        confidence, alternatives = self._calculate_confidence_with_decay(patterns)

        if patterns:
            success_rate = sum(
                p.success_count / max(p.total_count, 1) for p in patterns
            ) / n_patterns
            latency_ms = sum(
                p.avg_execution_cost_ms for p in patterns
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
            recommended_mode=recommended_mode,
            warnings=warnings,
            alternatives=alternatives,
            source_pattern_count=n_patterns,
            source="strategy_engine",
        )

    # ------------------------------------------------------------------
    # Private helpers — each source returns a partial view
    # ------------------------------------------------------------------

    def _load_patterns(self, goal: str, domain: str) -> list:
        """Load top patterns from LearningManager.

        Returns: list of ExecutionPattern objects, with the first element
        being the best pattern used for confidence.
        """
        if self._learning_manager is None:
            return []
        patterns = self._learning_manager.get_patterns_for_domain(domain)
        if not patterns:
            return []
        return patterns

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
        """Load entity states, state narratives, dwell times, and
        suggested next steps from the world model."""
        if self._world_model is None:
            return ""
        projects = self._world_model.find_entities(EntityType.PROJECT)
        blocks: list[str] = []
        for proj in projects:
            state = proj.current_state
            if not state:
                continue

            # 1. Basic state info.
            valid = self._world_model.get_valid_next_states(proj.id)
            basic = f"  - {proj.name} (id={proj.id}): current_state={state}"
            if valid:
                basic += f", valid_next_states={sorted(valid)}"
            blocks.append(basic)

            # 2. State narrative — chronological summary of transitions.
            narrative = self._world_model.get_state_narrative(proj.id)
            if narrative:
                blocks.append("    State narrative:\n    " +
                              narrative.replace("\n", "\n    "))

            # 3. Dwell times — how long in each state (identify stalls).
            dwells = self._world_model.get_state_dwell_times(proj.id)
            if dwells:
                dwell_lines = [
                    f"      {d['state']}: {d['duration_human']}"
                    + (" (active)" if d.get("still_active") else "")
                    for d in dwells
                ]
                blocks.append("    Dwell times:\n" + "\n".join(dwell_lines))

            # 4. Suggested next steps (with reasoning).
            suggestions = self._world_model.suggest_state_path(proj.id)
            if suggestions:
                sug_lines = [
                    f"      → {s['to']}: {s['reason']}"
                    for s in suggestions
                ]
                blocks.append("    Suggested next steps:\n" +
                              "\n".join(sug_lines))

        if not blocks:
            return ""
        return "Entity states:\n" + "\n".join(blocks)

    def _load_warnings(self, goal: str) -> list[str]:
        """Load anti-pattern warnings from LearningManager."""
        if self._learning_manager is None:
            return []
        # get_lessons already includes anti-patterns as text; we need
        # the raw AntiPattern objects.
        anti = self._learning_manager._search_anti_patterns(goal)
        return [f"{ap.warning} — Instead: {ap.suggestion}" if ap.suggestion else ap.warning for ap in anti]

    def _calculate_confidence_with_decay(self, patterns: list) -> tuple[float, list[StrategyAlternative]]:
        """Calculate strategy confidence with time-based and failure decay.

        Returns:
            Tuple of (primary_confidence, alternative_strategies)
        """
        if not patterns:
            return 0.0, []

        alternatives = []
        now = datetime.now(UTC)

        for i, p in enumerate(patterns):
            # Base confidence from pattern's average confidence
            base_conf = getattr(p, "avg_confidence", 0.0)

            # Time decay: reduce confidence if pattern hasn't been used recently
            last_success = getattr(p, "last_success_at", None)
            time_decay = 1.0
            if last_success:
                days_since = max(0.0, (now - last_success).total_seconds() / 86400.0)
                # Exponential decay with 7-day half-life
                time_decay = max(0.1, 2.0 ** (-days_since / 7.0))

            # Failure decay: reduce confidence based on failure rate
            success_rate = getattr(p, "success_count", 1) / max(getattr(p, "total_count", 1), 1)
            failure_rate = 1.0 - success_rate
            failure_decay = 1.0 - (failure_rate * 0.5)  # Up to 50% reduction for high failure rate

            # Combined confidence with decay
            decayed_confidence = base_conf * time_decay * failure_decay

            alt = StrategyAlternative(
                name=f"Strategy {chr(65 + i)}" if i < 26 else f"Strategy {i + 1}",
                strategy_summary=getattr(p, "strategy_summary", "")[:100],
                confidence=round(decayed_confidence, 3),
                expected_success_rate=round(success_rate, 3),
                expected_latency_ms=getattr(p, "avg_execution_cost_ms", 0.0),
                recommended_provider=getattr(p, "preferred_provider", None),
                recommended_model=getattr(p, "preferred_model", None),
            )
            alternatives.append(alt)

        # Sort by confidence descending
        alternatives.sort(key=lambda x: -x.confidence)

        primary_confidence = alternatives[0].confidence if alternatives else 0.0
        return primary_confidence, alternatives

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

    def _load_recommended_mode(
        self, goal: str, domain: str, patterns: list,
    ) -> str | None:
        """Choose an execution mode based on goal domain and complexity.

        Rules:
          1. **Very short goals** (≤ 4 words) → FAST (greetings, simple Q&A).
          2. **Coding domain** → CODING.
          3. **Research domain** → RESEARCH (unless very short, caught by #1).
          4. **Vision domain** → BALANCED.
          5. **General / reasoning with patterns** → AUTONOMOUS when patterns
             indicate deep reasoning or high cost; FAST if goal ≤ 8 words.
          6. **General / reasoning without patterns** → AUTONOMOUS for long
             goals (>10 words, likely multi-step); FAST for short; else BALANCED.
        """
        word_count = len(goal.split())

        # Rule 1 — very short → FAST regardless of domain.
        if word_count <= 4:
            return ExecutionMode.FAST.value

        # Rules 2-4 — domain-specific mapping.
        domain_mode: dict[str, ExecutionMode] = {
            GoalDomain.CODING.value: ExecutionMode.CODING,
            GoalDomain.RESEARCH.value: ExecutionMode.RESEARCH,
            GoalDomain.VISION.value: ExecutionMode.BALANCED,
        }
        mode = domain_mode.get(domain)
        if mode is not None:
            return mode.value

        # Rules 5-6 — General / reasoning.
        if word_count <= 8:
            return ExecutionMode.FAST.value

        if patterns:
            avg_confidence = sum(
                getattr(p, "avg_confidence", 0) for p in patterns
            ) / max(len(patterns), 1)
            avg_cost = sum(
                getattr(p, "avg_execution_cost_ms", 0) for p in patterns
            ) / max(len(patterns), 1)
            if avg_cost > 15000 or avg_confidence > 0.7:
                return ExecutionMode.AUTONOMOUS.value

        # Long general goal → autonomous (multi-step).
        if word_count >= 10:
            return ExecutionMode.AUTONOMOUS.value

        # Medium general goal with multi-step keywords → autonomous.
        multi_step_keywords = {
            "migrate", "refactor", "implement", "build", "deploy",
            "integrate", "configure", "restructure", "convert", "redesign",
        }
        if any(word in multi_step_keywords for word in goal.lower().split()):
            return ExecutionMode.AUTONOMOUS.value

        return ExecutionMode.BALANCED.value
