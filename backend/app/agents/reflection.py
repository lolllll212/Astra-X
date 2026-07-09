"""Reflection agent.

The :class:`Reflection` agent evaluates the quality of execution
results and decides whether more work is needed. This is the key
component that enables iterative improvement — instead of returning
the first answer, the system can refine its output until it meets a
quality threshold.

Reflection runs after every task execution and after the full plan
completes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.agents.base import Agent, AgentConfig
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.plan import Plan
from app.agents.models.task import Task
from app.agents.prompts.reflection import REFLECTION_SYSTEM_PROMPT
from app.core.logging import get_logger
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock
from app.llm.exceptions import GenerationError
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter

if TYPE_CHECKING:
    from app.llm.model_selector import ModelSelector

logger = get_logger(__name__)

_DEFAULT_CONFIDENCE = 0.8
"""Fallback confidence when the LLM output cannot be fully parsed."""


class Reflection(Agent):
    """LLM-powered quality evaluator for execution results.

    Usage::

        reflection = Reflection(llm_router=router)
        assessment = await reflection.reflect(execution_result)
        if assessment.needs_more_work:
            # Schedule follow-up tasks
            ...
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

    async def reflect(
        self,
        result: ExecutionResult,
    ) -> ReflectionResult:
        """Evaluate an execution result and produce an assessment.

        Args:
            result: The execution outcome to evaluate.

        Returns:
            A reflection assessment indicating whether more work is
            needed and with what confidence.
        """
        if result.status.value == "failed":
            return ReflectionResult(
                decision=ReflectionDecision.RETRY,
                feedback="The task failed during execution.",
                reason=result.error or "Unknown error.",
                confidence=0.0,
            )

        messages = self._build_messages(result)

        if self._model_selector is not None:
            model, provider = await self._model_selector.select_for_reflection()
        else:
            model = self._config.model
            provider = self._config.provider

        request = CompletionRequest(
            messages=messages,
            model=model,
            provider=provider,
            params=GenerationParams(
                temperature=0.3,
                max_tokens=self._config.max_tokens,
            ),
        )

        try:
            response: CompletionResponse = await self._llm_router.generate(request)
            assessment = self._parse_response(response)
        except (GenerationError, json.JSONDecodeError) as exc:
            logger.warning("agent.reflection_failed", error=str(exc))
            assessment = self._fallback_assessment(result)

        logger.info(
            "agent.reflection_complete",
            decision=assessment.decision.value,
            confidence=assessment.confidence,
        )
        return assessment

    async def plan(self, goal: str, context: str | None = None) -> Plan:
        """Not implemented — Reflection does not plan."""
        raise NotImplementedError("Reflection does not plan.")

    async def execute(self, task: Task) -> ExecutionResult:
        """Not implemented — Reflection does not execute."""
        raise NotImplementedError("Reflection does not execute.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, result: ExecutionResult) -> list[Message]:
        """Build the message list for the reflection LLM call.

        Args:
            result: The execution result to reflect on.

        Returns:
            A list of system and user messages.
        """
        task_context = (
            f"Task: {result.task_id}\n"
            f"Tool used: {result.tool_name or 'none'}\n"
            f"Output: {result.output or '(no output)'}\n"
            f"Error: {result.error or 'none'}"
        )

        return [
            Message(
                id=str(uuid4()),
                conversation_id="",
                role=MessageRole.SYSTEM,
                content=[TextBlock(text=REFLECTION_SYSTEM_PROMPT)],
            ),
            Message(
                id=str(uuid4()),
                conversation_id="",
                role=MessageRole.USER,
                content=[TextBlock(text=task_context)],
            ),
        ]

    def _parse_response(self, response: CompletionResponse) -> ReflectionResult:
        """Parse the LLM reflection response into a structured result.

        Args:
            response: The LLM completion response.

        Returns:
            A validated reflection result.
        """
        raw = ""
        for block in response.message.content:
            if isinstance(block, TextBlock):
                raw += block.text

        json_str = self._extract_json(raw)
        if json_str is None:
            raise GenerationError("Reflection output contains no JSON.")

        data: dict[str, Any] = json.loads(json_str)

        decision_raw = str(data.get("decision", "accept")).lower().strip()
        try:
            decision = ReflectionDecision(decision_raw)
        except ValueError:
            decision = ReflectionDecision.ACCEPT

        return ReflectionResult(
            decision=decision,
            feedback=str(data["feedback"]) if data.get("feedback") else None,
            reason=str(data.get("reason", "No reason provided.")),
            next_tasks=list(data.get("next_tasks", [])),
            confidence=float(data.get("confidence", _DEFAULT_CONFIDENCE)),
        )

    @staticmethod
    def _extract_json(raw: str) -> str | None:
        """Extract the first JSON object from a string.

        Args:
            raw: The raw text to search.

        Returns:
            The JSON substring, or ``None``.
        """
        start = raw.find("{")
        if start == -1:
            return None

        depth = 0
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
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]

        return None

    @staticmethod
    def _fallback_assessment(result: ExecutionResult) -> ReflectionResult:
        """Produce a safe fallback when parsing fails.

        Args:
            result: The execution result.

        Returns:
            A conservative reflection result.
        """
        if result.output:
            return ReflectionResult(
                decision=ReflectionDecision.ACCEPT,
                reason="Reflection parsing failed; accepting result as-is.",
                confidence=_DEFAULT_CONFIDENCE,
            )
        return ReflectionResult(
            decision=ReflectionDecision.RETRY,
            reason="Reflection parsing failed and no output was produced.",
            confidence=0.0,
        )
