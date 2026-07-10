"""Task execution graph with DAG dependencies, retries, and checkpoints.

The :class:`ExecutionGraph` manages task lifecycle through a directed
acyclic graph, supporting:

- **Dependencies**: tasks declare prerequisites that must complete first.
- **Parallel execution**: multiple ready tasks run concurrently.
- **Retries**: failed tasks can be retried up to *max_retries*.
- **Checkpoints**: the graph state can be saved and restored.
- **Pause/Resume**: execution can be paused and later resumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.agents.models.task import Task, TaskStatus


@dataclass
class TaskNode:
    """Internal node with execution metadata."""

    task: Task
    dependencies: set[str] = field(default_factory=set)
    retry_count: int = 0
    max_retries: int = 2
    last_error: str | None = None


class ExecutionGraph:
    """Tracks task dependencies and execution state.

    Usage::

        graph = ExecutionGraph()
        graph.add_task(task_a)
        graph.add_task(task_b, depends_on=[task_a.id])
        assert graph.get_ready() == [task_a]
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TaskNode] = {}
        self._paused: bool = False

    # -- Task management ------------------------------------------------------

    def add_task(
        self,
        task: Task,
        *,
        depends_on: list[str] | None = None,
        max_retries: int = 2,
    ) -> None:
        """Register a task in the graph.

        Args:
            task: The task to add.
            depends_on: Task IDs that must complete before *task* can run.
            max_retries: Maximum number of retries on failure.
        """
        self._nodes[task.id] = TaskNode(
            task=task,
            dependencies=set(depends_on or []),
            max_retries=max_retries,
        )

    def get_task(self, task_id: str) -> Task | None:
        node = self._nodes.get(task_id)
        return node.task if node else None

    # -- Queries --------------------------------------------------------------

    def get_ready(self) -> list[Task]:
        """Return all pending tasks whose dependencies are satisfied."""
        if self._paused:
            return []
        result: list[Task] = []
        for node in self._nodes.values():
            if node.task.status is not TaskStatus.PENDING:
                continue
            if self._dependencies_satisfied(node.task.id):
                result.append(node.task)
        return result

    def get_blocked(self) -> list[Task]:
        """Return all pending tasks whose dependencies are not yet met."""
        result: list[Task] = []
        for node in self._nodes.values():
            if node.task.status is not TaskStatus.PENDING:
                continue
            if not self._dependencies_satisfied(node.task.id):
                result.append(node.task)
        return result

    def is_complete(self) -> bool:
        """All tasks have reached a terminal status."""
        for node in self._nodes.values():
            if node.task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False
        return True

    @property
    def task_count(self) -> int:
        return len(self._nodes)

    @property
    def completed_count(self) -> int:
        return sum(
            1 for n in self._nodes.values() if n.task.status is TaskStatus.COMPLETED
        )

    @property
    def failed_count(self) -> int:
        return sum(
            1 for n in self._nodes.values() if n.task.status is TaskStatus.FAILED
        )

    # -- Status updates -------------------------------------------------------

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        node = self._nodes.get(task_id)
        if node is not None:
            node.task = node.task.model_copy(update={"status": status})

    def record_failure(self, task_id: str, error: str) -> TaskStatus:
        """Record a task failure and return the resulting status.

        If retries remain, the task is reset to ``PENDING`` and the
        retry counter is incremented. Otherwise the task stays ``FAILED``.

        Returns:
            The new status (``PENDING`` if retrying, ``FAILED`` otherwise).
        """
        node = self._nodes.get(task_id)
        if node is None:
            return TaskStatus.FAILED

        node.last_error = error
        node.retry_count += 1

        if node.retry_count <= node.max_retries:
            self.update_status(task_id, TaskStatus.PENDING)
            return TaskStatus.PENDING
        else:
            self.update_status(task_id, TaskStatus.FAILED)
            return TaskStatus.FAILED

    # -- Checkpoints ----------------------------------------------------------

    def save_checkpoint(self) -> dict[str, Any]:
        """Serialise the graph state for resume later."""
        return {
            "paused": self._paused,
            "nodes": {
                tid: {
                    "status": n.task.status.value,
                    "retry_count": n.retry_count,
                    "max_retries": n.max_retries,
                    "last_error": n.last_error,
                    "dependencies": list(n.dependencies),
                }
                for tid, n in self._nodes.items()
            },
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore graph state from a checkpoint."""
        self._paused = checkpoint.get("paused", False)
        for tid, data in checkpoint.get("nodes", {}).items():
            node = self._nodes.get(tid)
            if node is None:
                continue
            status = TaskStatus(data["status"])
            node.task = node.task.model_copy(update={"status": status})
            node.retry_count = data.get("retry_count", 0)
            node.max_retries = data.get("max_retries", 2)
            node.last_error = data.get("last_error")

    # -- Pause / Resume -------------------------------------------------------

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    # -- Internal -------------------------------------------------------------

    def _dependencies_satisfied(self, task_id: str) -> bool:
        node = self._nodes.get(task_id)
        if node is None:
            return False
        if not node.dependencies:
            return True
        for dep_id in node.dependencies:
            dep_node = self._nodes.get(dep_id)
            if dep_node is None or dep_node.task.status is not TaskStatus.COMPLETED:
                return False
        return True


# Backward-compatible alias for code still importing TaskGraph.
TaskGraph = ExecutionGraph  # noqa: F811
