"""Task executor agent.

The :class:`Executor` runs individual tasks produced by the planner.
It may invoke tools (web search, code execution, etc.) or use the LLM
to generate a direct response, depending on the task's configuration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.agents.base import Agent, AgentConfig
from app.agents.models.execution import ExecutionResult
from app.agents.models.plan import Plan
from app.agents.models.task import Task, TaskStatus
from app.agents.prompts.system import AGENT_SYSTEM_PROMPT
from app.core.logging import get_logger
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter

logger = get_logger(__name__)


class Executor(Agent):
    """LLM-powered executor that runs tasks and returns results.

    Usage::

        executor = Executor(llm_router=router)
        result = await executor.execute(task)
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        config: AgentConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._llm_router = llm_router

    async def execute(self, task: Task) -> ExecutionResult:
        """Execute a single task and return the result.

        If the task specifies a ``tool_name``, the executor attempts to
        invoke that tool (placeholder — real tool execution will be
        added in Phase 8). Otherwise, the LLM generates a direct
        response.

        Args:
            task: The task to execute.

        Returns:
            An execution result with status, output, and timing.
        """
        started_at = datetime.now(UTC)
        task_id = task.id

        logger.info(
            "agent.task_executing",
            task_id=task_id,
            description=task.description,
            tool=task.tool_name,
        )

        try:
            if task.tool_name:
                output = await self._execute_tool(task)
            else:
                output = await self._execute_llm(task)

            result = ExecutionResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                tool_name=task.tool_name,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            logger.info(
                "agent.task_completed",
                task_id=task_id,
                output_length=len(output) if output else 0,
            )
            return result

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "agent.task_failed",
                task_id=task_id,
                error=error_msg,
            )
            return ExecutionResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=error_msg,
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )

    async def plan(
        self,
        goal: str,
        context: str | None = None,
    ) -> Plan:
        """Not implemented — Executor does not plan."""
        raise NotImplementedError("Executor does not plan.")

    async def reflect(self, result: ExecutionResult) -> None:  # type: ignore[override]
        """Not implemented — Executor does not reflect."""
        raise NotImplementedError("Executor does not reflect.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_llm(self, task: Task) -> str:
        """Execute a task by generating an LLM response.

        Args:
            task: The task to execute.

        Returns:
            The LLM-generated text output.
        """
        messages = [
            Message(
                id=str(uuid4()),
                conversation_id="",
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=AGENT_SYSTEM_PROMPT)],
            ),
            Message(
                id=str(uuid4()),
                conversation_id="",
                role=MessageRole.USER,
                content=[TextBlock(text=task.description)],
            ),
        ]

        request = CompletionRequest(
            messages=messages,
            model=self._config.model,
            provider=self._config.provider,
            params=GenerationParams(
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            ),
        )

        response: CompletionResponse = await self._llm_router.generate(request)

        parts: list[str] = []
        for block in response.message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
        return "".join(parts)

    async def _execute_tool(self, task: Task) -> str:
        """Execute a tool call (placeholder).

        Real tool execution will be implemented in Phase 8. For now,
        this method returns a descriptive message.

        Args:
            task: The task specifying a tool to invoke.

        Returns:
            A placeholder result string.
        """
        tool_name = task.tool_name or "unknown"
        logger.info(
            "agent.tool_not_implemented",
            tool_name=tool_name,
            task_id=task.id,
        )
        return (
            f"[Tool '{tool_name}' is not yet implemented. "
            f"Task: {task.description}]"
        )
