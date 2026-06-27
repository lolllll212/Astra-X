"""Task dependency graph.

The :class:`TaskGraph` maintains a directed acyclic graph of tasks and
their dependencies. It is used by the coordinator to determine which
tasks are ready to execute and which must wait for dependencies to
complete.

This initial implementation uses a simple adjacency-list representation.
Future versions may support topological sorting, parallel execution
batches, and cycle detection.
"""

from __future__ import annotations

from app.agents.models.task import Task, TaskStatus


class TaskGraph:
    """Tracks dependencies between tasks.

    Each task may declare dependencies — task IDs that must complete
    before it can run. The graph provides queries for ready tasks,
    blocked tasks, and completion status.

    Usage::

        graph = TaskGraph()
        graph.add_task(task_a)
        graph.add_task(task_b, depends_on=[task_a.id])
        assert graph.get_ready() == [task_a]
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._dependencies: dict[str, set[str]] = {}

    def add_task(self, task: Task, *, depends_on: list[str] | None = None) -> None:
        """Register a task in the graph.

        Args:
            task: The task to add.
            depends_on: Task IDs that must complete before *task* can
                run. An empty or ``None`` value means the task has no
                dependencies and is ready immediately.
        """
        self._tasks[task.id] = task
        self._dependencies[task.id] = set(depends_on or [])

    def get_task(self, task_id: str) -> Task | None:
        """Retrieve a task by ID.

        Args:
            task_id: The task identifier.

        Returns:
            The task if found, ``None`` otherwise.
        """
        return self._tasks.get(task_id)

    def get_ready(self) -> list[Task]:
        """Return all tasks whose dependencies are satisfied.

        A task is ready when all its dependencies have a status of
        :attr:`TaskStatus.COMPLETED`.

        Returns:
            Tasks that are ready to execute, in insertion order.
        """
        result: list[Task] = []
        for task_id, task in self._tasks.items():
            if task.status is not TaskStatus.PENDING:
                continue
            if self._dependencies_satisfied(task_id):
                result.append(task)
        return result

    def get_blocked(self) -> list[Task]:
        """Return all pending tasks whose dependencies are not yet met.

        Returns:
            Tasks that are blocked, in insertion order.
        """
        result: list[Task] = []
        for task_id, task in self._tasks.items():
            if task.status is not TaskStatus.PENDING:
                continue
            if not self._dependencies_satisfied(task_id):
                result.append(task)
        return result

    def is_complete(self) -> bool:
        """Check whether every task has reached a terminal status.

        Terminal statuses are :attr:`TaskStatus.COMPLETED`,
        :attr:`TaskStatus.FAILED`, and :attr:`TaskStatus.SKIPPED`.
        """
        for task in self._tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False
        return True

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        """Update the status of a task in the graph.

        Args:
            task_id: The task to update.
            status: The new status.
        """
        task = self._tasks.get(task_id)
        if task is not None:
            self._tasks[task_id] = task.model_copy(update={"status": status})

    def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check whether all dependencies for a task are completed.

        Args:
            task_id: The task whose dependencies to check.

        Returns:
            ``True`` if all dependencies have status ``COMPLETED``.
        """
        deps = self._dependencies.get(task_id, set())
        if not deps:
            return True
        for dep_id in deps:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status is not TaskStatus.COMPLETED:
                return False
        return True

    @property
    def task_count(self) -> int:
        """Total number of tasks in the graph."""
        return len(self._tasks)

    @property
    def completed_count(self) -> int:
        """Number of tasks with status ``COMPLETED``."""
        return sum(1 for t in self._tasks.values() if t.status is TaskStatus.COMPLETED)
