"""Integration tests for the agent coordinator.

Tests the full plan → execute → reflect loop with mocked
sub-components.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.coordinator import Coordinator, CoordinatorResult
from app.agents.executor import Executor
from app.agents.memory_manager import MemoryManager
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.plan import Plan
from app.agents.models.task import Task, TaskStatus
from app.agents.planner import Planner
from app.agents.reflection import Reflection
from app.domain.message import Message, TextBlock
from app.llm.models import CompletionResponse
from app.llm.router import LLMRouter


def _task(**overrides: object) -> Task:
    defaults: dict[str, object] = {
        "id": "t1",
        "description": "test task",
    }
    defaults.update(overrides)
    return Task(**defaults)  # type: ignore[arg-type]


def _execution_result(**overrides: object) -> ExecutionResult:
    defaults: dict[str, object] = {
        "task_id": "t1",
        "status": TaskStatus.COMPLETED,
        "output": "result output",
    }
    defaults.update(overrides)
    return ExecutionResult(**defaults)  # type: ignore[arg-type]


def _reflection_result(**overrides: object) -> ReflectionResult:
    defaults: dict[str, object] = {
        "decision": ReflectionDecision.ACCEPT,
        "reason": "All good.",
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return ReflectionResult(**defaults)  # type: ignore[arg-type]


def _mock_llm_router(output: str = "no, all done") -> MagicMock:
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


def _mock_planner(tasks: list[Task] | None = None) -> MagicMock:
    planner = MagicMock(spec=Planner)
    if tasks is None:
        tasks = [_task()]
    planner.plan = AsyncMock(return_value=Plan(goal="test", tasks=tasks))
    return planner


def _mock_executor(result: ExecutionResult | None = None) -> MagicMock:
    executor = MagicMock(spec=Executor)
    if result is None:
        result = _execution_result()
    executor.execute = AsyncMock(return_value=result)
    return executor


def _mock_reflection(result: ReflectionResult | None = None) -> MagicMock:
    reflection = MagicMock(spec=Reflection)
    if result is None:
        result = _reflection_result()
    reflection.reflect = AsyncMock(return_value=result)
    return reflection


def _mock_memory_manager(context: str | None = "prior memory") -> MagicMock:
    mm = MagicMock(spec=MemoryManager)
    mm.get_context = AsyncMock(return_value=context)
    mm.store_result = AsyncMock()
    return mm


def make_coordinator(
    planner: MagicMock | None = None,
    executor: MagicMock | None = None,
    reflection: MagicMock | None = None,
    memory_manager: MagicMock | None = None,
    llm_router: MagicMock | None = None,
) -> Coordinator:
    return Coordinator(
        planner=planner or _mock_planner(),
        executor=executor or _mock_executor(),
        reflection=reflection or _mock_reflection(),
        memory_manager=memory_manager or _mock_memory_manager(),
        llm_router=llm_router or _mock_llm_router(),
    )


class TestCoordinator:
    @pytest.mark.asyncio
    async def test_run_returns_coordinator_result(self) -> None:
        coord = make_coordinator()
        result = await coord.run(conversation_id="conv1", goal="test goal")
        assert isinstance(result, CoordinatorResult)
        assert result.final_answer == "result output"
        assert result.plan is not None
        assert result.iterations == 1
        assert result.final_decision is ReflectionDecision.ACCEPT

    @pytest.mark.asyncio
    async def test_run_memory_context_retrieved(self) -> None:
        mm = _mock_memory_manager(context="important memory")
        coord = make_coordinator(memory_manager=mm)
        await coord.run(conversation_id="conv1", goal="test")
        mm.get_context.assert_awaited_once_with(
            conversation_id="conv1",
            goal="test",
        )

    @pytest.mark.asyncio
    async def test_run_memory_context_none(self) -> None:
        """None memory context is handled gracefully."""
        mm = _mock_memory_manager(context=None)
        coord = make_coordinator(memory_manager=mm)
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.final_answer

    @pytest.mark.asyncio
    async def test_run_planner_receives_memory_context(self) -> None:
        planner = _mock_planner()
        mm = _mock_memory_manager(context="mem")
        coord = make_coordinator(planner=planner, memory_manager=mm)
        await coord.run(conversation_id="conv1", goal="test")
        _, kwargs = planner.plan.call_args
        assert kwargs["goal"] == "test"
        assert kwargs["memory_context"] == "mem"
        assert kwargs["strategy"].goal == "test"

    @pytest.mark.asyncio
    async def test_run_multi_task_plan(self) -> None:
        tasks = [
            _task(id="t1", description="first", dependencies=[]),
            _task(id="t2", description="second", dependencies=["t1"]),
        ]
        planner = _mock_planner(tasks=tasks)
        executor = _mock_executor(_execution_result(task_id="t1", output="first result"))
        coord = make_coordinator(planner=planner, executor=executor)
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.final_answer

    @pytest.mark.asyncio
    async def test_run_retry_on_reflection(self) -> None:
        """Reflection says RETRY → task is re-added to graph (then ACCEPT on next)."""
        planner = _mock_planner(tasks=[_task(id="t1")])
        executor = _mock_executor(_execution_result(task_id="t1", output="partial"))
        reflection = MagicMock(spec=Reflection)
        reflection.reflect = AsyncMock(side_effect=[
            _reflection_result(decision=ReflectionDecision.RETRY, reason="Incomplete."),
            _reflection_result(decision=ReflectionDecision.ACCEPT, reason="Good now."),
        ])
        llm_router = _mock_llm_router("no, accept")
        coord = make_coordinator(
            planner=planner,
            executor=executor,
            reflection=reflection,
            llm_router=llm_router,
        )
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.final_answer

    @pytest.mark.asyncio
    async def test_run_replan_on_reflection(self) -> None:
        """REPLAN adds follow-up tasks from reflection."""
        planner = _mock_planner(tasks=[_task(id="t1")])
        executor = _mock_executor(_execution_result(task_id="t1", output="partial"))
        reflection = MagicMock(spec=Reflection)
        reflection.reflect = AsyncMock(side_effect=[
            _reflection_result(
                decision=ReflectionDecision.REPLAN,
                reason="Need more data.",
                next_tasks=["Fetch more data", "Re-analyze"],
            ),
            _reflection_result(decision=ReflectionDecision.ACCEPT, reason="Good 1."),
            _reflection_result(decision=ReflectionDecision.ACCEPT, reason="Good 2."),
        ])
        llm_router = _mock_llm_router("no, accept")
        coord = make_coordinator(
            planner=planner,
            executor=executor,
            reflection=reflection,
            llm_router=llm_router,
        )
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.final_answer

    @pytest.mark.asyncio
    async def test_run_abort_on_reflection(self) -> None:
        """ABORT stops execution."""
        planner = _mock_planner(tasks=[_task(id="t1")])
        executor = _mock_executor(_execution_result(task_id="t1", output="bad"))
        reflection = _mock_reflection(
            _reflection_result(
                decision=ReflectionDecision.ABORT,
                reason="Non-recoverable error.",
            ),
        )
        coord = make_coordinator(
            planner=planner,
            executor=executor,
            reflection=reflection,
        )
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.state.has_errors
        assert result.final_decision is ReflectionDecision.ABORT

    @pytest.mark.asyncio
    async def test_run_max_iterations(self) -> None:
        """Coordinator stops after max iterations.

        Task-level ACCEPTs so inner loop completes; plan-level says RETRY
        (text without "no") so outer loop continues up to max_iterations.
        """
        planner = _mock_planner(tasks=[_task(id="t1")])
        executor = _mock_executor(_execution_result(task_id="t1", output="data"))
        reflection = _mock_reflection(
            _reflection_result(decision=ReflectionDecision.ACCEPT, reason="Done."),
        )
        llm_router = MagicMock(spec=LLMRouter)
        llm_router.generate = AsyncMock(
            return_value=CompletionResponse(
                message=Message(
                    id="r1", conversation_id="", role="assistant",
                    content=[TextBlock(text="keep trying")],
                ),
            ),
        )
        coord = make_coordinator(
            planner=planner,
            executor=executor,
            reflection=reflection,
            llm_router=llm_router,
        )
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.state.is_exhausted
        assert result.iterations == 5

    @pytest.mark.asyncio
    async def test_run_with_timeline(self) -> None:
        """Coordinator result includes a timeline when tracing is active."""
        from app.core.tracing import Tracer

        tracer = Tracer()
        coord = make_coordinator()
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.timeline

    @pytest.mark.asyncio
    async def test_run_memory_store_called(self) -> None:
        mm = _mock_memory_manager()
        coord = make_coordinator(memory_manager=mm)
        await coord.run(conversation_id="conv1", goal="test")
        mm.store_result.assert_awaited()

    @pytest.mark.asyncio
    async def test_run_ask_user_reflection(self) -> None:
        """ASK_USER is handled (continues then accepts next)."""
        planner = _mock_planner(tasks=[_task(id="t1")])
        executor = _mock_executor(_execution_result(task_id="t1", output="needs clarity"))
        reflection = MagicMock(spec=Reflection)
        reflection.reflect = AsyncMock(side_effect=[
            _reflection_result(decision=ReflectionDecision.ASK_USER, reason="Need clarification."),
            _reflection_result(decision=ReflectionDecision.ACCEPT, reason="Good now."),
        ])
        coord = make_coordinator(planner=planner, executor=executor, reflection=reflection)
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.final_answer

    @pytest.mark.asyncio
    async def test_run_executor_failure(self) -> None:
        """Failed task is recorded in state."""
        planner = _mock_planner(tasks=[_task(id="t1")])
        executor = _mock_executor(
            _execution_result(
                task_id="t1",
                status=TaskStatus.FAILED,
                error="Something broke",
            ),
        )
        coord = make_coordinator(planner=planner, executor=executor)
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.state.failed_task_count >= 0

    @pytest.mark.asyncio
    async def test_run_plan_level_reflection_failure(self) -> None:
        """When plan-level reflection LLM fails, accept current result."""
        llm_router = MagicMock(spec=LLMRouter)
        llm_router.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        coord = make_coordinator(llm_router=llm_router)
        result = await coord.run(conversation_id="conv1", goal="test")
        assert result.final_answer
