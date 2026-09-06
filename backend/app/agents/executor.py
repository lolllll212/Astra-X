"""Task executor agent.

The :class:`Executor` runs individual tasks produced by the planner.
It resolves capabilities to concrete tool names via the
:class:`CapabilityRegistry`, then invokes tools or uses the LLM to
generate a direct response.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from app.agents.base import Agent, AgentConfig
from app.agents.models.execution import ExecutionResult
from app.agents.models.plan import Plan
from app.agents.models.task import Task, TaskStatus
from app.agents.prompts.system import AGENT_SYSTEM_PROMPT
from app.core.logging import get_logger
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock
from app.llm.model_selector import CapabilityProfile
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter

if TYPE_CHECKING:
    from app.llm.model_selector import ModelSelector
    from app.tools.capabilities import CapabilityRegistry

logger = get_logger(__name__)


class Executor(Agent):
    """LLM-powered executor that runs tasks and returns results.

    Resolves task capabilities to concrete tool names at execution time.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        capability_registry: CapabilityRegistry | None = None,
        model_selector: ModelSelector | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._llm_router = llm_router
        self._capability_registry = capability_registry
        self._model_selector = model_selector

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

        # Resolve capability to concrete tool name.
        tool_name: str | None = task.tool_name
        if tool_name is None and task.capability is not None and self._capability_registry is not None:
            try:
                tool_name = self._capability_registry.resolve_name(task.capability)
            except Exception:
                logger.warning(
                    "agent.capability_unresolved",
                    capability=task.capability,
                    task_id=task_id,
                )

        logger.info(
            "agent.task_executing",
            task_id=task_id,
            description=task.description,
            capability=task.capability,
            tool=tool_name,
        )

        try:
            if tool_name:
                output = await self._execute_tool(task, tool_name)
            else:
                output = await self._execute_llm(task)

            result = ExecutionResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                tool_name=tool_name,
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

        Uses the model selector to choose the best model based on the
        task's capability profile. Falls back to the agent config when
        no selector is available.

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

        if self._model_selector is not None:
            profile = self._profile_from_task(task)
            model, provider = await self._model_selector.select(profile)
        else:
            model = self._config.model
            provider = self._config.provider

        request = CompletionRequest(
            messages=messages,
            model=model,
            provider=provider,
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

    @staticmethod
    def _profile_from_task(task: Task) -> CapabilityProfile:
        """Build a capability profile from a task's profile dict (if set).

        When the task has no explicit profile, derives sensible defaults
        from the task's capability (e.g. coding-related capabilities map
        to ``requires_coding=True``).

        Args:
            task: The task to derive a profile for.

        Returns:
            A capability profile for model selection.
        """
        if task.profile is not None:
            return CapabilityProfile(
                requires_coding=bool(task.profile.get("requires_coding", False)),
                reasoning=str(task.profile.get("reasoning", "none")),
                prefers_speed=bool(task.profile.get("prefers_speed", False)),
                prefers_large_context=bool(task.profile.get("prefers_large_context", False)),
                requires_vision=bool(task.profile.get("requires_vision", False)),
            )

        cap = (task.capability or "").lower()
        coding_keywords = (
            "code", "python", "javascript", "write_file", "execute_python",
            "implement", "function", "class", "algorithm",
        )
        search_keywords = (
            "search", "fetch", "scrape", "download", "read_file",
            "list_directory", "search_files",
        )

        requires_coding = any(kw in cap for kw in coding_keywords)
        prefers_speed = any(kw in cap for kw in search_keywords)
        reasoning = "medium" if requires_coding else ("low" if prefers_speed else "none")

        return CapabilityProfile(
            requires_coding=requires_coding,
            reasoning=reasoning,
            prefers_speed=prefers_speed,
        )

    async def _execute_tool(self, task: Task, tool_name: str) -> str:
        """Execute a tool call.

        In the agent pipeline the executor does not have a direct tool
        registry reference — tool resolution happens through the
        coordinator's :class:`ToolExecutor`. This method prepares the
        call for the coordinator to dispatch.

        Args:
            task: The task specifying a capability to invoke.
            tool_name: The resolved concrete tool name.

        Returns:
            A placeholder result string until the coordinator dispatches
            the actual tool call.
        """
        logger.info(
            "agent.tool_executing",
            tool_name=tool_name,
            capability=task.capability,
            task_id=task.id,
        )
        return (
            f"[Tool '{tool_name}' capability '{task.capability}' "
            f"will be executed by the coordinator. "
            f"Task: {task.description}]"
        )
