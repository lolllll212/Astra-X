"""Unit tests for the reflection agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.models.execution import ExecutionResult, ReflectionDecision
from app.agents.models.task import TaskStatus
from app.agents.reflection import Reflection
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionResponse
from app.llm.router import LLMRouter


def _make_result(**overrides) -> ExecutionResult:
    defaults = {
        "task_id": "t1",
        "status": TaskStatus.COMPLETED,
        "output": "42",
    }
    defaults.update(overrides)
    return ExecutionResult(**defaults)


def _mock_router(json_response: str) -> MagicMock:
    router = MagicMock(spec=LLMRouter)
    router.generate = AsyncMock(
        return_value=CompletionResponse(
            message=Message(
                id="r1",
                conversation_id="",
                role="assistant",
                content=[TextBlock(text=json_response)],
            ),
        ),
    )
    return router


class TestReflection:
    @pytest.mark.asyncio
    async def test_accept_decision(self) -> None:
        """A good result returns decision=accept."""
        router = _mock_router(
            '{"decision": "accept", "feedback": "Good work.", "reason": "All good.", "next_tasks": [], "confidence": 0.95}',
        )
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(_make_result())
        assert result.decision is ReflectionDecision.ACCEPT
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_retry_decision(self) -> None:
        """A partial result returns decision=retry."""
        router = _mock_router(
            '{"decision": "retry", "feedback": "Incomplete.", "reason": "Only 3 of 10 items.", "next_tasks": ["Fetch remaining"], "confidence": 0.4}',
        )
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(_make_result())
        assert result.decision is ReflectionDecision.RETRY
        assert len(result.next_tasks) == 1

    @pytest.mark.asyncio
    async def test_replan_decision(self) -> None:
        """A wrong approach returns decision=replan."""
        router = _mock_router(
            '{"decision": "replan", "feedback": "Wrong source.", "reason": "Data not in README.", "next_tasks": ["Check docs/"], "confidence": 0.3}',
        )
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(_make_result())
        assert result.decision is ReflectionDecision.REPLAN

    @pytest.mark.asyncio
    async def test_abort_decision(self) -> None:
        """A non-recoverable error returns decision=abort."""
        router = _mock_router(
            '{"decision": "abort", "feedback": null, "reason": "API key invalid.", "next_tasks": [], "confidence": 0.0}',
        )
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(_make_result())
        assert result.decision is ReflectionDecision.ABORT

    @pytest.mark.asyncio
    async def test_ask_user_decision(self) -> None:
        """Ambiguous state returns decision=ask_user."""
        router = _mock_router(
            '{"decision": "ask_user", "feedback": null, "reason": "Need to know which repo.", "next_tasks": [], "confidence": 0.2}',
        )
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(_make_result())
        assert result.decision is ReflectionDecision.ASK_USER

    @pytest.mark.asyncio
    async def test_failed_task_returns_retry(self) -> None:
        """A failed execution result always returns retry without calling the LLM."""
        router = MagicMock(spec=LLMRouter)
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(
            _make_result(status=TaskStatus.FAILED, error="Connection refused"),
        )
        assert result.decision is ReflectionDecision.RETRY
        router.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_on_parse_failure(self) -> None:
        """When the LLM output is not valid JSON, a fallback is used."""
        router = _mock_router("this is not json at all")
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(_make_result(output="some output"))
        assert result.decision is ReflectionDecision.ACCEPT

    @pytest.mark.asyncio
    async def test_fallback_on_no_output(self) -> None:
        """When there is no output and parsing fails, retry is returned."""
        router = _mock_router("garbage output")
        reflection = Reflection(llm_router=router)
        result = await reflection.reflect(_make_result(output=""))
        assert result.decision is ReflectionDecision.RETRY
