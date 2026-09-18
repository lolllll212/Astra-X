from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.agents.learning_store import GoalDomain, LearningStore, classify_goal
from app.agents.models.pattern import ExecutionPattern
from app.agents.models.plan import Plan
from app.agents.models.strategy import Strategy
from app.agents.models.task import Task
from app.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass(frozen=True)
class TaskEstimate:
    """Estimated cost and success probability for a single task."""

    task_id: str
    description: str
    estimated_cost_ms: float
    success_probability: float
    confidence: float  # 0.0–1.0, how reliable we think the estimate is


@dataclass(frozen=True)
class SimulationResult:
    """Result of running a plan simulation."""

    plan_goal: str
    total_estimated_cost_ms: float
    overall_success_probability: float
    task_estimates: list[TaskEstimate] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


class PlanSimulator:
    """Heuristic-based plan simulator.

    Before executing a plan, the simulator estimates cost and success
    probability by matching each task against the pattern store and
    applying simple heuristics:

    - **Cost** is the sum of estimated per-task costs, derived from similar
      patterns' historical execution cost or a task-type baseline.
    - **Success probability** is the product of per-task success probabilities
      (tasks are sequential — the plan succeeds only if *all* tasks succeed).
    - **Confidence** reflects whether we had pattern data to base the estimate
      on (many matches → high confidence; no matches → low confidence).
    """

    def __init__(
        self,
        learning_store: LearningStore | None = None,
    ) -> None:
        self._store = learning_store

    async def simulate(
        self,
        goal: str,
        plan: Plan,
        strategy: Strategy | None = None,
    ) -> SimulationResult:
        """Run a heuristic simulation on *plan*.

        When a :class:`Strategy` is provided, its fused estimates
        (expected success rate, expected latency, recommended provider
        and tool sequence) serve as Bayesian priors that enrich the
        per-task estimates.

        Args:
            goal: The original user goal (used for pattern lookup).
            plan: The plan produced by the planner.
            strategy: Optional structured strategy guidance to inform estimates.

        Returns:
            A SimulationResult with per-task estimates and aggregates.
        """
        if not plan.tasks:
            return SimulationResult(
                plan_goal=goal,
                total_estimated_cost_ms=0.0,
                overall_success_probability=1.0,
                notes="Empty plan — trivially succeeds at zero cost.",
            )

        # Domain-level priors from the strategy engine.
        prior_success = strategy.expected_success_rate if strategy else 0.0
        prior_cost = strategy.expected_latency_ms if strategy else 0.0
        prior_confidence = strategy.confidence if strategy else 0.0
        recommended_tools = (
            set(strategy.recommended_tool_sequence) if strategy else set()
        )

        task_estimates: list[TaskEstimate] = []
        total_pattern_matches = 0

        for task in plan.tasks:
            est = await self._estimate_task(
                task, goal, prior_success, prior_cost, prior_confidence,
                recommended_tools,
            )
            task_estimates.append(est)
            if est.confidence > 0.5:
                total_pattern_matches += 1

        total_cost = sum(e.estimated_cost_ms for e in task_estimates)
        overall_success = 1.0
        for e in task_estimates:
            overall_success *= e.success_probability

        # Blend with prior if we have one and our estimate is low-confidence.
        n = len(task_estimates)
        confidence = total_pattern_matches / n if n > 0 else 0.0
        if prior_confidence > confidence and prior_success > 0:
            overall_success = (
                overall_success * (1 - prior_confidence)
                + prior_success * prior_confidence
            )
            confidence = prior_confidence

        notes = self._build_notes(task_estimates, overall_success, total_cost)
        if strategy:
            notes += f" | Strategy priors: {strategy.goal_domain} "

        return SimulationResult(
            plan_goal=goal,
            total_estimated_cost_ms=total_cost,
            overall_success_probability=overall_success,
            task_estimates=task_estimates,
            confidence=confidence,
            notes=notes,
        )

    async def _estimate_task(
        self,
        task: Task,
        goal: str,
        prior_success: float = 0.0,
        prior_cost: float = 0.0,
        prior_confidence: float = 0.0,
        recommended_tools: set[str] | None = None,
    ) -> TaskEstimate:
        """Estimate cost and success for a single task.

        Uses pattern-based estimation when available, blends with
        strategy priors, and adjusts for tool-sequence alignment.
        """
        recommended_tools = recommended_tools or set()

        # Try pattern-based estimation first.
        est: TaskEstimate | None = None
        if self._store is not None:
            domain = classify_goal(goal)
            patterns = self._store.search(
                f"{goal} {task.description} {task.capability or ''}",
                limit=3,
                domain=domain,
            )
            if patterns:
                est = self._estimate_from_patterns(task, patterns)

        if est is None:
            est = self._heuristic_estimate(task)

        # Boost confidence/success when task capability matches
        # the strategy's recommended tool sequence.
        if recommended_tools and task.capability in recommended_tools:
            boosted_success = est.success_probability * 1.05
            est = TaskEstimate(
                task_id=est.task_id,
                description=est.description,
                estimated_cost_ms=est.estimated_cost_ms,
                success_probability=min(boosted_success, 0.99),
                confidence=min(est.confidence + 0.15, 1.0),
            )

        # Blend with domain-level priors when our estimate is weak.
        if prior_confidence > 0.3 and est.confidence < prior_confidence:
            blend = prior_confidence * 0.4
            cost = est.estimated_cost_ms * (1 - blend) + prior_cost * blend
            success = (
                est.success_probability * (1 - blend)
                + prior_success * blend
            )
            est = TaskEstimate(
                task_id=est.task_id,
                description=est.description,
                estimated_cost_ms=cost,
                success_probability=success,
                confidence=(est.confidence + prior_confidence) / 2,
            )

        return est

    @staticmethod
    def _estimate_from_patterns(
        task: Task,
        patterns: list[ExecutionPattern],
    ) -> TaskEstimate:
        """Estimate based on similar historical patterns."""
        avg_cost = 0.0
        avg_success = 0.0
        avg_conf = 0.0

        for p in patterns:
            avg_cost += p.avg_execution_cost_ms
            avg_success += p.success_count / max(p.total_count, 1)
            avg_conf += p.last_reflection_confidence

        n = len(patterns)
        return TaskEstimate(
            task_id=task.id,
            description=task.description[:80],
            estimated_cost_ms=avg_cost / n,
            success_probability=avg_success / n,
            confidence=avg_conf / n,
        )

    @staticmethod
    def _heuristic_estimate(task: Task) -> TaskEstimate:
        """Fallback heuristic when no patterns match.

        Uses task properties to produce a rough estimate:
        - Tasks with capabilities (tools) are faster but may fail more.
        - Tasks with "deep" reasoning are slower but more reliable.
        - Plain LLM tasks are medium.
        """
        profile = task.profile or {}

        # Default: plain LLM call.
        cost_ms = 3000.0
        success_prob = 0.85

        if task.capability:
            # Tool execution: fast but risk of tool error.
            cost_ms = 1500.0
            success_prob = 0.80

        reasoning = profile.get("reasoning", "none")
        if reasoning in ("deep", "medium"):
            cost_ms = 8000.0
            success_prob = 0.90

        requires_coding = profile.get("requires_coding", False)
        if requires_coding:
            cost_ms = 10000.0
            success_prob = 0.75

        return TaskEstimate(
            task_id=task.id,
            description=task.description[:80],
            estimated_cost_ms=cost_ms,
            success_probability=success_prob,
            confidence=0.0,
        )

    @staticmethod
    def _build_notes(
        estimates: list[TaskEstimate],
        overall_success: float,
        total_cost: float,
    ) -> str:
        """Build a human-readable note about the simulation."""
        parts: list[str] = []
        parts.append(f"Estimated total cost: {total_cost / 1000:.1f}s")
        parts.append(f"Estimated success rate: {overall_success:.0%}")

        low_conf = [e for e in estimates if e.confidence < 0.3]
        if low_conf:
            parts.append(
                f"Low confidence for {len(low_conf)} task(s) "
                f"({', '.join(e.task_id for e in low_conf)}) — "
                "no matching historical patterns."
            )

        high_risk = [e for e in estimates if e.success_probability < 0.6]
        if high_risk:
            parts.append(
                f"High risk for {len(high_risk)} task(s) — "
                "success probability below 60%."
            )

        return " | ".join(parts)
