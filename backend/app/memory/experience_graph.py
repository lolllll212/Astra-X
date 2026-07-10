"""Experience Graph — stores full execution trajectories.

Instead of flat :class:`ExecutionPattern` objects, each execution is
recorded as a rich trajectory::

    Goal
    ├── Plan (task list with ordering)
    │   ├── Task 1 → Tool X → Reflection → Outcome
    │   ├── Task 2 → Tool Y → Reflection → Outcome
    │   └── Task 3 → (LLM only) → Reflection → Outcome
    ├── Provider: ollama / openai / …
    ├── Model: llama3.1 / gpt-4 / …
    ├── Total cost: 4.2s
    └── Final outcome: accepted / failed

This enables the system to answer questions like:
- "Which workflow historically succeeds best for coding tasks?"
- "Which provider gave the fastest results for research goals?"
- "What tool combination works best for full-stack projects?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.plan import Plan
from app.agents.models.task import TaskStatus
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TaskNode:
    """A single task within an execution trajectory."""

    task_id: str
    description: str
    capability: str | None
    tool_name: str | None
    status: str  # "completed" | "failed" | "skipped"
    latency_ms: float
    reflection_decision: str
    reflection_confidence: float
    reflection_feedback: str
    error: str | None = None


@dataclass
class ExecutionTrajectory:
    """Full record of one plan→execute→reflect cycle.

    Attributes:
        id: Unique trajectory identifier.
        goal: The original user goal.
        goal_domain: Auto-detected domain (coding, research, …).
        plan_tasks: Ordered list of tasks from the plan.
        task_nodes: Per-task execution + reflection details.
        provider_id: The provider used (if single-provider run).
        model_id: The model used.
        total_cost_ms: Sum of all task latencies.
        overall_decision: Final reflection decision (accept / retry / abort).
        overall_confidence: Average reflection confidence.
        created_at: When this trajectory was recorded.
        metadata: Arbitrary extra info.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    goal: str = ""
    goal_domain: str = "general"
    plan_tasks: list[str] = field(default_factory=list)
    task_nodes: list[TaskNode] = field(default_factory=list)
    provider_id: str = ""
    model_id: str = ""
    total_cost_ms: float = 0.0
    overall_decision: str = "accept"
    overall_confidence: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        completed = sum(1 for t in self.task_nodes if t.status == "completed")
        return completed / max(len(self.task_nodes), 1)

    @property
    def is_success(self) -> bool:
        return self.overall_decision == "accept" and self.success_rate >= 0.8


class ExperienceGraph:
    """Stores and queries execution trajectories.

    The experience graph sits alongside the pattern store and knowledge
    graph, providing a richer query surface for the planner and strategy
    engine.
    """

    def __init__(self) -> None:
        self._trajectories: dict[str, ExecutionTrajectory] = {}

    # -- Write ----------------------------------------------------------------

    def record(
        self,
        *,
        goal: str,
        goal_domain: str = "general",
        plan: Plan | None = None,
        results: list[ExecutionResult],
        reflections: list[ReflectionResult],
        provider_id: str = "",
        model_id: str = "",
        overall_decision: str = "accept",
    ) -> ExecutionTrajectory:
        """Build and store a trajectory from one execution cycle."""
        plan_tasks = [t.description for t in plan.tasks] if plan else []

        task_nodes: list[TaskNode] = []
        total_cost = 0.0
        confidences: list[float] = []

        for i, result in enumerate(results):
            lat = (
                (result.completed_at - result.started_at).total_seconds() * 1000.0
                if result.completed_at and result.started_at
                else 0.0
            )
            total_cost += lat

            ref = reflections[i] if i < len(reflections) else None
            confidences.append(ref.confidence if ref else 0.0)

            task_nodes.append(
                TaskNode(
                    task_id=result.task_id,
                    description=result.metadata.get("description", "") or "",
                    capability=result.metadata.get("capability", None),
                    tool_name=result.tool_name,
                    status=result.status.value,
                    latency_ms=lat,
                    reflection_decision=(
                        ref.decision.value if ref else "unknown"
                    ),
                    reflection_confidence=ref.confidence if ref else 0.0,
                    reflection_feedback=ref.feedback if ref else "",
                    error=result.error,
                )
            )

        trajectory = ExecutionTrajectory(
            goal=goal,
            goal_domain=goal_domain,
            plan_tasks=plan_tasks,
            task_nodes=task_nodes,
            provider_id=provider_id,
            model_id=model_id,
            total_cost_ms=total_cost,
            overall_decision=overall_decision,
            overall_confidence=(
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
        )
        self._trajectories[trajectory.id] = trajectory
        return trajectory

    # -- Read ---------------------------------------------------------------

    def get(self, trajectory_id: str) -> ExecutionTrajectory | None:
        return self._trajectories.get(trajectory_id)

    def search(
        self,
        *,
        goal_keywords: list[str] | None = None,
        domain: str | None = None,
        min_success_rate: float = 0.0,
        limit: int = 10,
    ) -> list[ExecutionTrajectory]:
        """Find trajectories matching filters, sorted by recency."""
        results = list(self._trajectories.values())

        if domain:
            results = [t for t in results if t.goal_domain == domain]
        if min_success_rate > 0:
            results = [t for t in results if t.success_rate >= min_success_rate]
        if goal_keywords:
            results = [
                t
                for t in results
                if any(kw.lower() in t.goal.lower() for kw in goal_keywords)
            ]

        results.sort(key=lambda t: t.created_at, reverse=True)
        return results[:limit]

    def best_workflow_for_domain(self, domain: str) -> dict[str, Any]:
        """Return the best-performing workflow for a domain.

        Analyses all trajectories for the domain and returns the most
        common plan structure, provider, and tool combination.
        """
        matches = self.search(domain=domain, min_success_rate=0.7)
        if not matches:
            return {}

        # Most common plan pattern (first 3 task descriptions).
        plan_patterns: dict[str, int] = {}
        provider_counts: dict[str, int] = {}
        tool_combo: dict[str, int] = {}

        for t in matches:
            plan_key = " → ".join(t.plan_tasks[:3])
            plan_patterns[plan_key] = plan_patterns.get(plan_key, 0) + 1
            if t.provider_id:
                provider_counts[t.provider_id] = (
                    provider_counts.get(t.provider_id, 0) + 1
                )
            tools = frozenset(
                n.tool_name for n in t.task_nodes if n.tool_name
            )
            if tools:
                tool_combo[str(tools)] = tool_combo.get(str(tools), 0) + 1

        best_plan = max(plan_patterns, key=plan_patterns.get) if plan_patterns else ""
        best_provider = (
            max(provider_counts, key=provider_counts.get) if provider_counts else ""
        )

        return {
            "domain": domain,
            "total_trajectories": len(matches),
            "avg_success_rate": (
                sum(t.success_rate for t in matches) / len(matches)
            ),
            "most_common_plan": best_plan,
            "most_common_provider": best_provider,
            "recommended_approach": best_plan.split(" → ") if best_plan else [],
        }

    # -- Maintenance ----------------------------------------------------------

    def clear(self) -> None:
        self._trajectories.clear()

    @property
    def count(self) -> int:
        return len(self._trajectories)
