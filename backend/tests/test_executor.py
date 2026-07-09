"""Unit tests for the executor agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.executor import Executor
from app.agents.models.task import Task, TaskStatus
from app.domain.message import Message, TextBlock
from app.llm.model_selector import CapabilityProfile, ModelSelector
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

    # ------------------------------------------------------------------
    # Profile-based model selection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_uses_model_selector_when_available(self) -> None:
        """Executor uses ModelSelector when provided to pick model for LLM tasks."""
        router = _mock_router("selected result")
        model_selector = MagicMock(spec=ModelSelector)
        model_selector.select = AsyncMock(return_value=("deepseek-coder", "ollama"))
        executor = Executor(llm_router=router, model_selector=model_selector)

        result = await executor.execute(_task(description="Code task"))

        assert result.status is TaskStatus.COMPLETED
        assert result.output == "selected result"
        model_selector.select.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_passes_correct_profile_to_selector(self) -> None:
        """The derived profile is passed to ModelSelector.select()."""
        router = _mock_router("ok")
        model_selector = MagicMock(spec=ModelSelector)
        model_selector.select = AsyncMock(return_value=("llama3.1", "ollama"))
        executor = Executor(llm_router=router, model_selector=model_selector)

        await executor.execute(_task(
            description="Implement sorting",
            capability="execute_python",
        ))

        call_args = model_selector.select.await_args
        assert call_args is not None
        profile: CapabilityProfile = call_args[0][0]
        assert profile.requires_coding is True
        assert profile.reasoning == "medium"

    @pytest.mark.asyncio
    async def test_execute_profile_from_task_profile_dict(self) -> None:
        """Task profile dict is used directly for model selection."""
        router = _mock_router("ok")
        model_selector = MagicMock(spec=ModelSelector)
        model_selector.select = AsyncMock(return_value=("mistral-nemo", "lm_studio"))
        executor = Executor(llm_router=router, model_selector=model_selector)

        await executor.execute(_task(
            description="Analyze large document",
            profile={"prefers_large_context": True, "reasoning": "deep"},
        ))

        call_args = model_selector.select.await_args
        assert call_args is not None
        profile: CapabilityProfile = call_args[0][0]
        assert profile.prefers_large_context is True
        assert profile.reasoning == "deep"
        assert profile.requires_coding is False

    @pytest.mark.asyncio
    async def test_execute_without_model_selector_falls_back_to_config(self) -> None:
        """Executor falls back to AgentConfig when no model_selector is provided."""
        router = _mock_router("fallback result")
        executor = Executor(llm_router=router)

        result = await executor.execute(_task(description="Simple task"))

        assert result.status is TaskStatus.COMPLETED
        assert result.output == "fallback result"

    # ------------------------------------------------------------------
    # Profile-from-task derivation
    # ------------------------------------------------------------------

    def test_profile_from_coding_capability(self) -> None:
        task = _task(capability="write_python_code")
        profile = Executor._profile_from_task(task)
        assert profile.requires_coding is True
        assert profile.reasoning == "medium"

    def test_profile_from_search_capability(self) -> None:
        task = _task(capability="search_web")
        profile = Executor._profile_from_task(task)
        assert profile.requires_coding is False
        assert profile.prefers_speed is True
        assert profile.reasoning == "low"

    def test_profile_from_task_profile_dict_overrides(self) -> None:
        task = _task(
            description="Generic task",
            capability=None,
            profile={"requires_coding": True, "reasoning": "deep"},
        )
        profile = Executor._profile_from_task(task)
        assert profile.requires_coding is True
        assert profile.reasoning == "deep"

    def test_profile_defaults_when_no_profile_and_no_capability(self) -> None:
        task = _task(description="Simple")
        profile = Executor._profile_from_task(task)
        assert profile.requires_coding is False
        assert profile.reasoning == "none"
        assert profile.prefers_speed is False
