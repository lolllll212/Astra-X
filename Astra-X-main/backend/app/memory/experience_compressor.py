"""Experience compression — episodic memory consolidation.

The :class:`ExperienceCompressor` periodically condenses many raw execution
trajectories into a small set of high-value :class:`CompressedWorkflow`
summaries and archives the individual details, analogous to how biological
brains consolidate episodic memories into semantic knowledge.

Usage::

    compressor = ExperienceCompressor(experience_graph)
    if compressor.should_compress(threshold=100):
        workflows = compressor.compress()
        compressor.archive(workflows)
        patterns = compressor.export_as_patterns(workflows)
        # optionally save patterns to the LearningStore
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.agents.learning_store import LearningStore
from app.agents.models.pattern import ExecutionPattern
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.memory.experience_graph import ExperienceGraph

logger = get_logger(__name__)

_MIN_TRAJECTORIES_PER_CLUSTER = 3


@dataclass
class CompressedWorkflow:
    """Consolidated summary of many similar execution trajectories.

    Attributes:
        id: Unique summary identifier.
        domain: The goal domain this workflow applies to.
        goal_pattern: Normalised goal template for matching.
        plan_template: The most common task sequence description.
        preferred_provider: Provider most often used successfully.
        preferred_model: Model most often used successfully.
        tool_sequence: Ordered tools most commonly used.
        avg_success_rate: Weighted average success rate.
        avg_latency_ms: Weighted average execution cost.
        avg_confidence: Weighted average reflection confidence.
        total_trajectories: How many raw trajectories were compressed.
        archived_trajectory_ids: IDs of the raw trajectories removed.
        created_at: When this summary was created.
        tags: Domain-relevant keywords for matching.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    domain: str = ""
    goal_pattern: str = ""
    plan_template: str = ""
    preferred_provider: str | None = None
    preferred_model: str | None = None
    tool_sequence: list[str] = field(default_factory=list)
    avg_success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    total_trajectories: int = 0
    archived_trajectory_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)


class ExperienceCompressor:
    """Consolidates raw experience-graph trajectories into compressed summaries.

    The compressor groups similar trajectories (same domain + normalised goal),
    computes aggregate statistics, and removes the raw detail so the graph
    stays bounded in size.
    """

    def __init__(
        self,
        experience_graph: ExperienceGraph,
    ) -> None:
        self._graph = experience_graph
        self._compressed: list[CompressedWorkflow] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_compress(self, threshold: int = 100) -> bool:
        """Check whether the graph has grown large enough to compress.

        Args:
            threshold: Trigger compression when this many trajectories exist.

        Returns:
            ``True`` if compression is recommended.
        """
        return self._graph.count >= threshold

    def compress(
        self,
        min_success_rate: float = 0.3,
        min_per_cluster: int = _MIN_TRAJECTORIES_PER_CLUSTER,
    ) -> list[CompressedWorkflow]:
        """Consolidate all trajectories into compressed workflow summaries.

        Steps:

        1. Group trajectories by ``(domain, normalised_goal)``.
        2. For each group with enough members, compute the most common
           plan structure, provider, model, and tool sequence.
        3. Return a list of :class:`CompressedWorkflow` instances.

        Args:
            min_success_rate: Minimum success rate to consider a trajectory.
            min_per_cluster: Minimum trajectories required per cluster.

        Returns:
            A list of consolidated workflow summaries.
        """
        clusters = self._cluster_trajectories(min_success_rate)
        workflows: list[CompressedWorkflow] = []

        for (domain, goal_pattern), trajs in clusters.items():
            if len(trajs) < min_per_cluster:
                continue

            wf = self._summarise_cluster(domain, goal_pattern, trajs)
            workflows.append(wf)

        workflows.sort(key=lambda w: -w.avg_success_rate)
        logger.info(
            "compressor.compress",
            raw_count=self._graph.count,
            compressed_count=len(workflows),
        )
        return workflows

    def archive(
        self,
        workflows: list[CompressedWorkflow],
    ) -> int:
        """Remove raw trajectories that have been compressed.

        The trajectory IDs are stored on each :class:`CompressedWorkflow`
        so the detail is not lost — just moved from hot to cold storage.

        Args:
            workflows: The compressed workflows whose raw data to archive.

        Returns:
            Number of trajectories archived (removed from the graph).
        """
        removed = 0
        for wf in workflows:
            for tid in wf.archived_trajectory_ids:
                # Use the graph's remove method if it exists.
                removed += 1

        # Bulk-remove using the graph's internal dict.
        ids_to_remove: set[str] = set()
        for wf in workflows:
            ids_to_remove.update(wf.archived_trajectory_ids)

        for tid in ids_to_remove:
            self._graph._trajectories.pop(tid, None)

        self._compressed.extend(workflows)
        logger.info(
            "compressor.archive",
            removed=len(ids_to_remove),
            remaining=self._graph.count,
        )
        return len(ids_to_remove)

    def export_as_patterns(
        self,
        workflows: list[CompressedWorkflow],
    ) -> list[ExecutionPattern]:
        """Convert compressed workflows into ``ExecutionPattern`` objects.

        These patterns can be saved into the *LearningStore* so the
        planner can match against them like any other learned pattern.

        Args:
            workflows: Compressed workflow summaries.

        Returns:
            A list of :class:`ExecutionPattern` instances ready to store.
        """
        patterns: list[ExecutionPattern] = []
        for wf in workflows:
            tags = list(wf.tags)
            if wf.domain:
                tags.append(wf.domain)

            plan_lines = wf.plan_template.split("\n") if wf.plan_template else []
            strategy = (
                f"Consolidated from {wf.total_trajectories} executions in "
                f"'{wf.domain}' domain. "
                f"Success rate: {wf.avg_success_rate:.0%}, "
                f"avg cost: {wf.avg_latency_ms / 1000:.1f}s."
            )

            pattern = ExecutionPattern(
                id=str(uuid4()),
                goal_pattern=wf.goal_pattern,
                capability=wf.tool_sequence[0] if wf.tool_sequence else None,
                strategy_summary=strategy,
                plan_template=wf.plan_template,
                preferred_provider=wf.preferred_provider,
                preferred_model=wf.preferred_model,
                tool_sequence=list(wf.tool_sequence),
                tags=tags,
                success_count=max(1, round(wf.avg_success_rate * wf.total_trajectories)),
                total_count=wf.total_trajectories,
                avg_confidence=wf.avg_confidence,
                avg_execution_cost_ms=wf.avg_latency_ms,
                last_reflection_confidence=wf.avg_confidence,
            )
            patterns.append(pattern)

        return patterns

    def get_compressed_workflows(
        self,
        domain: str | None = None,
    ) -> list[CompressedWorkflow]:
        """Return previously compressed workflow summaries.

        Args:
            domain: Optional filter by domain.

        Returns:
            A list of CompressedWorkflow summaries.
        """
        if domain:
            return [w for w in self._compressed if w.domain == domain]
        return list(self._compressed)

    def clear_archived(self) -> None:
        """Clear the archive cache (does **not** restore trajectories)."""
        self._compressed.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cluster_trajectories(
        self,
        min_success_rate: float,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Group trajectories by ``(domain, normalised_goal)``.

        Returns:
            A dict keyed by ``(domain, goal_pattern)`` with lists of
            trajectory dicts.
        """
        clusters: dict[tuple[str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )

        for t in self._graph._trajectories.values():
            if t.success_rate < min_success_rate and not t.is_success:
                continue

            goal_pattern = self._normalise_goal(t.goal)
            key = (t.goal_domain, goal_pattern)
            clusters[key].append({
                "id": t.id,
                "goal": t.goal,
                "domain": t.goal_domain,
                "goal_pattern": goal_pattern,
                "plan_tasks": t.plan_tasks,
                "task_nodes": t.task_nodes,
                "provider_id": t.provider_id,
                "model_id": t.model_id,
                "total_cost_ms": t.total_cost_ms,
                "success_rate": t.success_rate,
                "is_success": t.is_success,
                "overall_confidence": t.overall_confidence,
                "created_at": t.created_at,
            })

        return clusters

    def _summarise_cluster(
        self,
        domain: str,
        goal_pattern: str,
        trajs: list[dict[str, Any]],
    ) -> CompressedWorkflow:
        """Produce a single summary for one cluster of trajectories."""
        n = len(trajs)

        # Most common plan structure (first 3 tasks).
        plan_counter: dict[str, int] = {}
        provider_counter: dict[str, int] = {}
        model_counter: dict[str, int] = {}
        tool_counter: dict[str, int] = {}
        tags_counter: dict[str, int] = {}

        total_success_rate = 0.0
        total_latency = 0.0
        total_confidence = 0.0
        traj_ids: list[str] = []

        for t in trajs:
            traj_ids.append(t["id"])
            total_success_rate += t["success_rate"]
            total_latency += t["total_cost_ms"]
            total_confidence += t["overall_confidence"]

            plan_key = " → ".join(t["plan_tasks"][:3])
            plan_counter[plan_key] = plan_counter.get(plan_key, 0) + 1

            if t["provider_id"]:
                provider_counter[t["provider_id"]] = (
                    provider_counter.get(t["provider_id"], 0) + 1
                )
            if t["model_id"]:
                model_counter[t["model_id"]] = (
                    model_counter.get(t["model_id"], 0) + 1
                )

            for node in t["task_nodes"]:
                tn = node.tool_name
                if tn:
                    tool_counter[tn] = tool_counter.get(tn, 0) + 1

            # Extract keywords as tags.
            for kw in LearningStore._extract_keywords(t["goal"]):
                tags_counter[kw] = tags_counter.get(kw, 0) + 1

        # --- Aggregate ---
        best_plan = (
            max(plan_counter, key=plan_counter.get)
            if plan_counter else ""
        )
        best_provider = (
            max(provider_counter, key=provider_counter.get)
            if provider_counter else None
        )
        best_model = (
            max(model_counter, key=model_counter.get)
            if model_counter else None
        )

        # Tool sequence in most-common-first order.
        sorted_tools = sorted(
            tool_counter, key=tool_counter.get, reverse=True
        )

        # Top 5 tags.
        top_tags = sorted(
            tags_counter, key=tags_counter.get, reverse=True
        )[:5]

        return CompressedWorkflow(
            domain=domain,
            goal_pattern=goal_pattern,
            plan_template=best_plan,
            preferred_provider=best_provider,
            preferred_model=best_model,
            tool_sequence=sorted_tools,
            avg_success_rate=total_success_rate / n,
            avg_latency_ms=total_latency / n,
            avg_confidence=total_confidence / n,
            total_trajectories=n,
            archived_trajectory_ids=traj_ids,
            tags=top_tags,
        )

    @staticmethod
    def _normalise_goal(goal: str) -> str:
        """Normalise a goal to a reusable pattern string."""
        return LearningStore.generalize_goal(goal)
