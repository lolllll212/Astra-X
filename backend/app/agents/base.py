"""Abstract base class for all agents.

Every agent in the system — planner, executor, reflection, and future
specialised agents (coding, research, vision) — inherits from
:class:`Agent` and implements the three core methods:

* :meth:`plan` — decompose a goal into tasks.
* :meth:`execute` — run a single task.
* :meth:`reflect` — evaluate a result and decide next steps.

Agents are stateless by design. All execution state lives in
:class:`app.agents.state.AgentState`, which is passed into each method
as needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.agents.models.execution import ExecutionResult, ReflectionResult
from app.agents.models.plan import Plan
from app.agents.models.task import Task


@dataclass(frozen=True)
class AgentConfig:
    """Configuration shared by all agents.

    Attributes:
        model: The LLM model to use for agent reasoning.
        provider: The LLM provider to route requests through.
        max_tokens: Maximum tokens per LLM call.
        temperature: Generation temperature.
    """

    model: str = "llama3.1"
    provider: str | None = None
    max_tokens: int | None = 2048
    temperature: float = 0.7


class Agent(ABC):
    """Abstract base agent with plan / execute / reflect lifecycle.

    Usage::

        class ResearchAgent(Agent):
            async def plan(self, goal, state, context):
                ...

            async def execute(self, task, state, context):
                ...

            async def reflect(self, result, state, context):
                ...
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()

    @property
    def config(self) -> AgentConfig:
        """Return the agent's configuration."""
        return self._config

    @abstractmethod
    async def plan(
        self,
        goal: str,
        context: str | None = None,
    ) -> Plan:
        """Decompose a user goal into a structured plan of tasks.

        Args:
            goal: The user's original request.
            context: Optional additional context (memory, history).

        Returns:
            A plan containing tasks to achieve the goal.
        """
        ...

    @abstractmethod
    async def execute(
        self,
        task: Task,
    ) -> ExecutionResult:
        """Execute a single task and return the result.

        Args:
            task: The task to execute.

        Returns:
            The outcome of execution.
        """
        ...

    @abstractmethod
    async def reflect(
        self,
        result: ExecutionResult,
    ) -> ReflectionResult:
        """Evaluate an execution result and decide whether more work is needed.

        Args:
            result: The outcome of a task execution.

        Returns:
            A reflection assessment.
        """
        ...
