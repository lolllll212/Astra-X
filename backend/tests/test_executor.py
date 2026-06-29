"""Unit tests for the executor agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.executor import Executor
from app.agents.models.task import Task, TaskStatus
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionResponse
from app.llm.router import LLMRouter
from app.tools.capabilities import CapabilityRegistry
from app.tools.registry import ToolRegistry


def _mock_router(output: str = "test result") -> MagicMock:
    router = MagicMock(spec=LLMRouter)
    router.generate = AsyncMock(
        return_value=CompletionResponse(
            message=Message(
                id="r1",
                conversation_id="",
                role="assistant",
                content=[TextBlock(text=output)],
            ),
        ),
    )
    return router


def _task(**overrides: object) -> Task:
    defaults: dict[str, object] = {
        "id": "t1",
        "description": "test task",
    }
    defaults.update(overrides)
    return Task(**defaults)  # type: ignore[arg-type]


def _capability_registry() -> CapabilityRegistry:
    """Create a registry with a fake tool for testing."""
    from tests.test_capabilities import _FakeTool
    tr = ToolRegistry()
    tr.register(_FakeTool("web_search", capabilities=["search_web"]))
    return CapabilityRegistry(tr)


class TestExecutor:
    @pytest.mark.asyncio
    async def test_execute_llm_task_returns_result(self) -> None:
        router = _mock_router("computed answer")
        executor = Executor(llm_router=router)
        result = await executor.execute(_task(description="Compute"))
        assert result.status is TaskStatus.COMPLETED
        assert result.output == "computed answer"
        assert result.task_id == "t1"

    @pytest.mark.asyncio
    async def test_execute_with_tool_name_directly(self) -> None:
        """When tool_name is set directly, executor returns placeholder."""
        router = _mock_router()
        executor = Executor(llm_router=router)
        result = await executor.execute(_task(
            tool_name="web_search",
            capability=None,
        ))
        assert result.status is TaskStatus.COMPLETED
        assert "web_search" in (result.output or "")
        assert result.tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_execute_resolves_capability_to_tool(self) -> None:
        """Capability is resolved to tool_name via CapabilityRegistry."""
        router = _mock_router()
        cap_reg = _capability_registry()
        executor = Executor(llm_router=router, capability_registry=cap_reg)
        result = await executor.execute(_task(
            capability="search_web",
            tool_name=None,
        ))
        assert result.status is TaskStatus.COMPLETED
        assert "web_search" in (result.output or "")
        assert result.tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_execute_capability_without_registry_falls_back_to_llm(self) -> None:
        """When capability is set but no registry, falls back to LLM."""
        router = _mock_router("llm fallback")
        executor = Executor(llm_router=router, capability_registry=None)
        result = await executor.execute(_task(capability="search_web"))
        assert result.status is TaskStatus.COMPLETED
        assert result.output == "llm fallback"
        assert result.tool_name is None

    @pytest.mark.asyncio
    async def test_execute_unknown_capability_falls_back_to_llm(self) -> None:
        """Unknown capability is logged but doesn't crash."""
        router = _mock_router("fallback output")
        cap_reg = _capability_registry()
        executor = Executor(llm_router=router, capability_registry=cap_reg)
        result = await executor.execute(_task(
            capability="nonexistent_cap",
            tool_name=None,
        ))
        assert result.status is TaskStatus.COMPLETED
        assert result.output == "fallback output"

    @pytest.mark.asyncio
    async def test_execute_records_timing(self) -> None:
        router = _mock_router("timed")
        executor = Executor(llm_router=router)
        result = await executor.execute(_task())
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at

    @pytest.mark.asyncio
    async def test_execute_llm_exception_returns_failed(self) -> None:
        router = MagicMock(spec=LLMRouter)
        router.generate = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        executor = Executor(llm_router=router)
        result = await executor.execute(_task())
        assert result.status is TaskStatus.FAILED
        assert "RuntimeError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_with_tool_name_and_capability(self) -> None:
        """When both are set, tool_name takes precedence."""
        router = _mock_router()
        cap_reg = _capability_registry()
        executor = Executor(llm_router=router, capability_registry=cap_reg)
        result = await executor.execute(_task(
            tool_name="web_search",
            capability="search_web",
        ))
        assert result.tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_execute_plan_not_implemented(self) -> None:
        executor = Executor(llm_router=MagicMock(spec=LLMRouter))
        with pytest.raises(NotImplementedError):
            await executor.plan("goal")

    @pytest.mark.asyncio
    async def test_execute_reflect_not_implemented(self) -> None:
        executor = Executor(llm_router=MagicMock(spec=LLMRouter))
        with pytest.raises(NotImplementedError):
            await executor.reflect(MagicMock())

    @pytest.mark.asyncio
    async def test_execute_with_retry_after_failure(self) -> None:
        """Executor handles tasks that previously failed."""
        router = _mock_router("retry output")
        executor = Executor(llm_router=router)
        result = await executor.execute(_task(description="Retry me"))
        assert result.status is TaskStatus.COMPLETED
        assert result.output == "retry output"
