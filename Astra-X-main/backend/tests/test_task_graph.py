"""Unit tests for the task graph scheduler."""

from __future__ import annotations

from app.agents.models.task import Task, TaskStatus
from app.agents.task_graph import TaskGraph


def _task(**overrides: object) -> Task:
    """Create a Task with minimal defaults for testing."""
    defaults: dict[str, object] = {
        "id": "t1",
        "description": "test task",
    }
    defaults.update(overrides)
    return Task(**defaults)  # type: ignore[arg-type]


class TestTaskGraph:
    def test_empty_graph_is_complete(self) -> None:
        graph = TaskGraph()
        assert graph.is_complete()

    def test_single_task(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        assert not graph.is_complete()
        ready = graph.get_ready()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_dependency_blocks_ready(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"), depends_on=["t1"])
        ready_ids = [t.id for t in graph.get_ready()]
        assert ready_ids == ["t1"]

    def test_dependency_satisfied(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"), depends_on=["t1"])
        graph.update_status("t1", TaskStatus.COMPLETED)
        ready_ids = [t.id for t in graph.get_ready()]
        assert "t2" in ready_ids

    def test_failed_dependency_blocks_downstream(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"), depends_on=["t1"])
        graph.update_status("t1", TaskStatus.FAILED)
        ready_ids = [t.id for t in graph.get_ready()]
        assert ready_ids == []

    def test_concurrent_tasks(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"))
        graph.add_task(_task(id="t3"))
        ready_ids = [t.id for t in graph.get_ready()]
        assert len(ready_ids) == 3

    def test_complex_dag(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"), depends_on=["t1"])
        graph.add_task(_task(id="t3"), depends_on=["t1"])
        graph.add_task(_task(id="t4"), depends_on=["t2", "t3"])

        # Initially only t1 is ready
        ready_ids = [t.id for t in graph.get_ready()]
        assert ready_ids == ["t1"]
        graph.update_status("t1", TaskStatus.COMPLETED)

        # Now t2 and t3 are ready
        ready_ids = [t.id for t in graph.get_ready()]
        assert set(ready_ids) == {"t2", "t3"}

        graph.update_status("t2", TaskStatus.COMPLETED)
        graph.update_status("t3", TaskStatus.COMPLETED)

        # Now t4 is ready
        ready_ids = [t.id for t in graph.get_ready()]
        assert ready_ids == ["t4"]

    def test_status_tracking(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        assert graph.get_task("t1") is not None
        assert graph.get_task("t1").status is TaskStatus.PENDING
        graph.update_status("t1", TaskStatus.RUNNING)
        assert graph.get_task("t1").status is TaskStatus.RUNNING
        graph.update_status("t1", TaskStatus.COMPLETED)
        assert graph.get_task("t1").status is TaskStatus.COMPLETED

    def test_get_task_unknown(self) -> None:
        graph = TaskGraph()
        assert graph.get_task("nonexistent") is None

    def test_add_nested_dependency(self) -> None:
        """Add a task that depends on an already-completed task — ready immediately."""
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.update_status("t1", TaskStatus.COMPLETED)
        graph.add_task(_task(id="t2"), depends_on=["t1"])
        ready_ids = [t.id for t in graph.get_ready()]
        assert "t2" in ready_ids

    def test_blocked_tasks(self) -> None:
        """Tasks with unsatisfied dependencies show up in get_blocked()."""
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"), depends_on=["t1"])
        blocked = graph.get_blocked()
        assert len(blocked) == 1
        assert blocked[0].id == "t2"

    def test_task_count(self) -> None:
        graph = TaskGraph()
        assert graph.task_count == 0
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"))
        assert graph.task_count == 2

    def test_completed_count(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"))
        assert graph.completed_count == 0
        graph.update_status("t1", TaskStatus.COMPLETED)
        assert graph.completed_count == 1

    def test_is_complete_with_tasks(self) -> None:
        graph = TaskGraph()
        graph.add_task(_task(id="t1"))
        graph.add_task(_task(id="t2"))
        graph.update_status("t1", TaskStatus.COMPLETED)
        assert not graph.is_complete()
        graph.update_status("t2", TaskStatus.COMPLETED)
        assert graph.is_complete()
