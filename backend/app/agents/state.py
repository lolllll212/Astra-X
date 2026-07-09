"""Agent execution state.

:class:`AgentState` holds all mutable state for a single agent
execution cycle — from receiving a user goal through to producing the
final response. It acts as the agent's working memory during one
request.

The coordinator creates a fresh ``AgentState`` for each request,
populates it, and passes it through the plan → execute → reflect loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.models.execution import ExecutionResult, ReflectionResult
from app.agents.models.plan import Plan
from app.agents.models.task import TaskStatus


@dataclass
class AgentState:
    """Mutable execution state for one agent invocation.

    Attributes:
        conversation_id: The conversation this execution belongs to.
        goal: The original user goal.
        plan: The current plan, populated by the planner.
        iteration: Number of plan→execute→reflect cycles completed.
        max_iterations: Maximum cycles before forcing a final answer.
        completed_results: Results of all completed tasks, keyed by task ID.
        memory_context: Context retrieved from the memory system.
        errors: Error messages accumulated during execution.
    """

    conversation_id: str
    goal: str

    plan: Plan | None = None
    iteration: int = 0
    max_iterations: int = 5

    completed_results: dict[str, ExecutionResult] = field(default_factory=dict)
    reflections: dict[str, ReflectionResult] = field(default_factory=dict)
    memory_context: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def is_exhausted(self) -> bool:
        """Whether the agent has reached its maximum iteration limit."""
        return self.iteration >= self.max_iterations

    @property
    def has_errors(self) -> bool:
        """Whether any errors have been recorded."""
        return len(self.errors) > 0

    @property
    def completed_task_count(self) -> int:
        """Number of tasks that completed successfully."""
        return sum(
            1 for r in self.completed_results.values()
            if r.status is TaskStatus.COMPLETED
        )

    @property
    def failed_task_count(self) -> int:
        """Number of tasks that failed."""
        return sum(
            1 for r in self.completed_results.values()
            if r.status is TaskStatus.FAILED
        )

    def record_error(self, error: str) -> None:
        """Record an error message.

        Args:
            error: The error description.
        """
        self.errors.append(error)
