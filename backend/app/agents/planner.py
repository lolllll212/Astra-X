"""Goal planner agent.

The :class:`Planner` agent receives a user goal and decomposes it into
a structured sequence of tasks. It uses the LLM to reason about what
steps are needed, in what order, and which tools (if any) each step
requires.

The planner does NOT execute anything — it only produces a plan.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.agents.base import Agent, AgentConfig
from app.agents.models.plan import Plan
from app.agents.models.task import Task
from app.agents.prompts.planner import PLANNER_SYSTEM_PROMPT
from app.core.logging import get_logger
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock
from app.llm.exceptions import GenerationError
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter

if TYPE_CHECKING:
    from app.llm.model_selector import ModelSelector

logger = get_logger(__name__)


class Planner(Agent):
    """LLM-powered planner that decomposes goals into task sequences.

    Usage::

        planner = Planner(llm_router=router)
        plan = await planner.plan(
            goal="Research the best RTX 5070 laptops",
        )
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        model_selector: ModelSelector | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._llm_router = llm_router
        self._model_selector = model_selector

    async def plan(
        self,
        goal: str,
        memory_context: str = "",
        patterns_context: str = "",
        temperature: float | None = None,
    ) -> Plan:
        """Decompose *goal* into a plan using the LLM.

        The planner always receives relevant memories retrieved for this
        goal and is instructed to reason with them when decomposing the
        request into tasks.

        Args:
            goal: The user's request.
            memory_context: Relevant memories retrieved for this goal.
            patterns_context: Lessons learned from past executions.
            temperature: Optional override for generation temperature
                (used for generating alternative plan variants).

        Returns:
            A validated plan with tasks in execution order.

        Raises:
            GenerationError: If the LLM fails to produce a valid plan.
        """
        messages = self._build_messages(goal, memory_context, patterns_context)

        if self._model_selector is not None:
            model, provider = await self._model_selector.select_for_planning()
        else:
            model = self._config.model
            provider = self._config.provider

        request = CompletionRequest(
            messages=messages,
            model=model,
            provider=provider,
            params=GenerationParams(
                temperature=temperature if temperature is not None else self._config.temperature,
                max_tokens=self._config.max_tokens,
            ),
        )

        response: CompletionResponse = await self._llm_router.generate(request)
        raw_text = self._extract_text(response)

        tasks = self._parse_tasks(raw_text, goal)
        plan = Plan(goal=goal, tasks=tasks)

        logger.info(
            "agent.plan_created",
            goal=goal,
            task_count=len(tasks),
        )
        return plan

    def _build_messages(
        self,
        goal: str,
        memory_context: str,
        patterns_context: str = "",
    ) -> list[Message]:
        """Build the message list for the planner LLM call.

        Memory context is always prepended so the planner reasons with
        relevant prior knowledge before decomposing the goal.

        Args:
            goal: The user goal.
            memory_context: Relevant memories retrieved for this goal.
            patterns_context: Lessons learned from past executions.

        Returns:
            A list of system and user messages.
        """
        parts: list[str] = []
        if memory_context:
            parts.append(f"Relevant memories:\n{memory_context}")
        if patterns_context:
            parts.append(patterns_context)
        parts.append(f"Goal: {goal}")
        content = "\n\n".join(parts)

        return [
            Message(
                id=str(uuid4()),
                conversation_id="",
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=PLANNER_SYSTEM_PROMPT)],
            ),
            Message(
                id=str(uuid4()),
                conversation_id="",
                role=MessageRole.USER,
                content=[TextBlock(text=content)],
            ),
        ]

    def _extract_text(self, response: CompletionResponse) -> str:
        """Extract text content from a completion response.

        Args:
            response: The LLM response.

        Returns:
            Concatenated text from text blocks.
        """
        parts: list[str] = []
        for block in response.message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
        return "".join(parts)

    def _parse_tasks(self, raw: str, goal: str) -> list[Task]:
        """Parse the LLM's JSON output into a list of tasks.

        Attempts to extract and parse a JSON array from the response.
        Falls back to creating a single default task if parsing fails.

        Args:
            raw: The raw LLM output.
            goal: The original goal (used for fallback task description).

        Returns:
            A validated list of tasks.

        Raises:
            GenerationError: If the output cannot be parsed.
        """
        json_str = self._extract_json(raw)
        if json_str is None:
            logger.warning("agent.plan_parse_failed", raw=raw[:200])
            raise GenerationError("Planner output could not be parsed as JSON.")

        try:
            data: list[dict[str, Any]] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("agent.plan_json_error", error=str(exc))
            raise GenerationError(f"Planner JSON parse error: {exc}") from exc

        if not data:
            raise GenerationError("Planner returned an empty task list.")

        tasks: list[Task] = []
        for item in data:
            profile = item.get("profile")
            if profile is not None and not isinstance(profile, dict):
                profile = None
            task = Task(
                id=item.get("id", str(uuid4())),
                description=item.get("description", goal),
                capability=item.get("capability"),
                profile=profile,
                dependencies=item.get("dependencies", []),
            )
            tasks.append(task)

        return tasks

    @staticmethod
    def _extract_json(raw: str) -> str | None:
        """Extract the first JSON array or object from a string.

        Looks for the first ``[`` or ``{`` and matches it to the
        corresponding closing bracket.

        Args:
            raw: The raw text to search.

        Returns:
            The JSON substring, or ``None`` if no JSON is found.
        """
        start = raw.find("[")
        if start == -1:
            start = raw.find("{")
        if start == -1:
            return None

        brace_depth = 0
        bracket_depth = 0
        in_string = False
        escape = False

        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth -= 1

            if brace_depth == 0 and bracket_depth == 0:
                return raw[start : i + 1]

        return raw[start:] if (brace_depth == 0 and bracket_depth == 0) else None

    # ------------------------------------------------------------------
    # Unused base-class stubs (Planner only plans)
    # ------------------------------------------------------------------

    async def execute(self, task: Task) -> None:  # type: ignore[override]
        """Not implemented — Planner does not execute tasks."""
        raise NotImplementedError("Planner does not execute tasks.")

    async def reflect(self, result: object) -> None:  # type: ignore[override]
        """Not implemented — Planner does not reflect."""
        raise NotImplementedError("Planner does not reflect.")
