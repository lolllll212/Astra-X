"""Unit tests for ChatCoordinator — the streaming event pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.agents.base import AgentConfig
from app.agents.executor import Executor as AgentExecutor
from app.agents.memory_manager import MemoryManager as AgentMemoryManager
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.plan import Plan
from app.agents.models.task import Task, TaskStatus
from app.agents.planner import Planner
from app.agents.reflection import Reflection
from app.domain.conversation import Conversation
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock, ToolCallBlock
from app.domain.stream import (
    ArtifactEvent,
    PlanEvent,
    ReflectionEvent,
    StreamDoneEvent,
    StreamEvent,
    StreamMetadataEvent,
    StreamStartEvent,
    StreamUsageEvent,
    TaskProgressEvent,
    TextDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolProgressEvent,
    ToolResultStreamEvent,
)
from app.llm.router import LLMRouter
from app.services.coordinator import ChatCoordinator, _extract_user_goal, _inject_tool_schemas
from app.tools.executor import ToolExecutor
from app.tools.models import ToolParameter, ToolSchema
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conversation(**kw: Any) -> Conversation:
    return Conversation(id="conv-1", title="test", **kw)


def _user_message(text: str = "hello") -> Message:
    return Message(
        id=str(uuid4()),
        conversation_id="conv-1",
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
    )


def _task(**overrides: Any) -> Task:
    kwargs: dict[str, Any] = dict(id="t1", description="test task")
    kwargs.update(overrides)
    return Task(**kwargs)


def _execution_result(**overrides: Any) -> ExecutionResult:
    kwargs: dict[str, Any] = dict(task_id="t1", status=TaskStatus.COMPLETED, output="42")
    kwargs.update(overrides)
    return ExecutionResult(**kwargs)


def _reflection_result(**overrides: Any) -> ReflectionResult:
    kwargs: dict[str, Any] = dict(decision=ReflectionDecision.ACCEPT, reason="ok", confidence=0.9)
    kwargs.update(overrides)
    return ReflectionResult(**kwargs)


def _mock_stream(events: list[StreamEvent]) -> Any:
    """Return an async generator function that yields *events*."""

    async def _gen(*args: object, **kw: object) -> AsyncIterator[StreamEvent]:
        for e in events:
            yield e

    return _gen


def _mock_router(*, stream_events: list[StreamEvent] | None = None) -> MagicMock:
    router = MagicMock(spec=LLMRouter)
    router.generate = AsyncMock(
        return_value=MagicMock(
            message=Message(
                id="r1", conversation_id="", role="assistant",
                content=[TextBlock(text="no, all done")],
            ),
        ),
    )
    router.generate_stream = _mock_stream(stream_events or [])
    return router


def _mock_planner(tasks: list[Task] | None = None) -> MagicMock:
    planner = MagicMock(spec=Planner)
    planner.plan = AsyncMock(return_value=Plan(goal="test", tasks=tasks or [_task()]))
    return planner


def _mock_agent_executor(result: ExecutionResult | None = None) -> MagicMock:
    executor = MagicMock(spec=AgentExecutor)
    executor.execute = AsyncMock(return_value=result or _execution_result())
    return executor


def _mock_reflection(result: ReflectionResult | None = None) -> MagicMock:
    reflection = MagicMock(spec=Reflection)
    reflection.reflect = AsyncMock(return_value=result or _reflection_result())
    return reflection


def _mock_agent_memory_manager(*, context: str | None = "prior memory") -> MagicMock:
    mm = MagicMock(spec=AgentMemoryManager)
    mm.get_context = AsyncMock(return_value=context)
    mm.store_result = AsyncMock()
    return mm


def _mock_tool_registry(*, schemas: list[ToolSchema] | None = None) -> MagicMock:
    registry = MagicMock(spec=ToolRegistry)
    if schemas is not None:
        registry.count = len(schemas)
        registry.schemas = MagicMock(return_value=schemas)
    else:
        registry.count = 0
        registry.schemas = MagicMock(return_value=[])
    return registry


def _mock_tool_executor(result: ToolResult | None = None) -> MagicMock:
    exe = MagicMock(spec=ToolExecutor)
    exe.execute = AsyncMock(
        return_value=result or ToolResult(success=True, output="tool result"),
    )
    return exe


def _make_coordinator(
    *,
    router: MagicMock | None = None,
    registry: MagicMock | None = None,
    tool_executor: MagicMock | None = None,
    planner: MagicMock | None = None,
    agent_executor: MagicMock | None = None,
    reflection: MagicMock | None = None,
    memory_manager: MagicMock | None = None,
    capability_registry: MagicMock | None = None,
    agent_config: AgentConfig | None = None,
) -> ChatCoordinator:
    return ChatCoordinator(
        llm_router=router or _mock_router(),
        tool_registry=registry,
        tool_executor=tool_executor,
        planner=planner,
        agent_executor=agent_executor,
        reflection=reflection,
        memory_manager=memory_manager,
        capability_registry=capability_registry,
        agent_config=agent_config,
    )


# ---------------------------------------------------------------------------
# Tests : _extract_user_goal
# ---------------------------------------------------------------------------


class TestExtractUserGoal:
    def test_from_text_block(self) -> None:
        msg = _user_message("what is the meaning of life")
        assert _extract_user_goal(msg) == "what is the meaning of life"

    def test_empty_when_no_text_blocks(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role=MessageRole.USER,
            content=[ToolCallBlock(tool_call_id="tc1", tool_name="calc", arguments={})],
        )
        assert _extract_user_goal(msg) == ""

    def test_picks_first_text_block(self) -> None:
        msg = Message(
            id="m1",
            conversation_id="c1",
            role=MessageRole.USER,
            content=[TextBlock(text="first"), TextBlock(text="second")],
        )
        assert _extract_user_goal(msg) == "first"


# ---------------------------------------------------------------------------
# Tests : _inject_tool_schemas
# ---------------------------------------------------------------------------


class TestInjectToolSchemas:
    def test_appends_schema_message(self) -> None:
        messages = [
            Message(id="m1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="hi")]),
        ]
        schemas = [
            ToolSchema(
                name="calc",
                description="a calculator",
                parameters=[ToolParameter(name="expr", type_="string", required=True)],
            ),
        ]
        registry = _mock_tool_registry(schemas=schemas)
        result = _inject_tool_schemas(messages, registry)
        assert len(result) == 2
        assert result[1].role is MessageRole.SYSTEM
        assert "calc" in str(result[1].content)

    def test_returns_original_when_registry_empty(self) -> None:
        messages = [
            Message(id="m1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="hi")]),
        ]
        registry = _mock_tool_registry(schemas=[])
        result = _inject_tool_schemas(messages, registry)
        assert len(result) == 1

    def test_returns_original_when_registry_none(self) -> None:
        messages = [
            Message(id="m1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="hi")]),
        ]
        result = _inject_tool_schemas(messages, None)
        assert len(result) == 1

    def test_renders_parameter_properties(self) -> None:
        messages = [
            Message(id="m1", conversation_id="c1", role=MessageRole.USER, content=[TextBlock(text="go")]),
        ]
        schemas = [
            ToolSchema(
                name="search",
                description="web search",
                parameters=[
                    ToolParameter(name="q", description="query", type_="string", required=True),
                    ToolParameter(name="limit", description="max results", type_="integer", required=False),
                ],
            ),
        ]
        registry = _mock_tool_registry(schemas=schemas)
        result = _inject_tool_schemas(messages, registry)
        text = str(result[1].content)
        assert "search" in text
        assert "web search" in text
        assert "query" in text
        assert "string" in text


# ---------------------------------------------------------------------------
# Tests : ChatCoordinator — agent pipeline (full flow)
# ---------------------------------------------------------------------------


class TestChatCoordinatorAgentPipeline:
    @pytest.mark.asyncio
    async def test_run_agent_pipeline_yields_full_event_sequence(self) -> None:
        """Agent mode yields start → metadata → plan → task progress → reflection → text deltas → usage → done."""
        router = _mock_router(
            stream_events=[
                TextDeltaEvent(delta="Hello"),
                StreamUsageEvent(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ],
        )
        planner = _mock_planner(tasks=[_task(id="t1", description="count to 42")])
        exec_result = _execution_result(task_id="t1", output="1 2 3")
        agent_executor = _mock_agent_executor(exec_result)
        reflection = _mock_reflection(_reflection_result(decision=ReflectionDecision.ACCEPT))
        mm = _mock_agent_memory_manager(context="mem")
        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
            memory_manager=mm,
        )

        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do the thing"),
        ):
            events.append(event)

        types = [e.type for e in events]
        assert StreamStartEvent().type in types
        assert StreamMetadataEvent(type="metadata", conversation_id="", message_id="", model="", provider="").type in types
        assert PlanEvent(goal="x", tasks=[], iteration=0).type in types
        assert any(isinstance(e, TaskProgressEvent) for e in events)
        assert ReflectionEvent(decision="", reason="", confidence=0.0, iteration=0).type in types
        assert TextDeltaEvent(delta="Hello").type in types
        assert StreamUsageEvent(prompt_tokens=0, completion_tokens=0, total_tokens=0).type in types
        assert StreamDoneEvent(finish_reason="stop").type in types

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_memory_context_passed_to_planner(self) -> None:
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="done")],
        )
        planner = _mock_planner()
        agent_executor = _mock_agent_executor()
        reflection = _mock_reflection()
        mm = _mock_agent_memory_manager(context="important memory")

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
            memory_manager=mm,
        )
        async for _ in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do it"),
        ):
            pass

        mm.get_context.assert_awaited_once_with(conversation_id="conv-1", goal="do it")
        planner.plan.assert_awaited_once_with(goal="do it", memory_context="important memory", patterns_context="")

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_no_memory_manager(self) -> None:
        """When memory_manager is None, planner gets empty string."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="ok")],
        )
        planner = _mock_planner()
        agent_executor = _mock_agent_executor()
        reflection = _mock_reflection()

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
            memory_manager=None,
        )
        async for _ in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("go"),
        ):
            pass

        planner.plan.assert_awaited_once_with(goal="go", memory_context="", patterns_context="")

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_memory_context_none(self) -> None:
        """When get_context returns None, planner gets empty string."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="ok")],
        )
        planner = _mock_planner()
        agent_executor = _mock_agent_executor()
        reflection = _mock_reflection()
        mm = _mock_agent_memory_manager(context=None)

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
            memory_manager=mm,
        )
        async for _ in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("go"),
        ):
            pass

        planner.plan.assert_awaited_once_with(goal="go", memory_context="", patterns_context="")

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_stores_memory_after_execution(self) -> None:
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="ok")],
        )
        planner = _mock_planner(tasks=[_task(id="t1")])
        exec_result = _execution_result(task_id="t1", output="data")
        agent_executor = _mock_agent_executor(exec_result)
        reflection = _mock_reflection(_reflection_result(decision=ReflectionDecision.ACCEPT))
        mm = _mock_agent_memory_manager(context="mem")

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
            memory_manager=mm,
        )
        async for _ in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("go"),
        ):
            pass

        mm.store_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_task_failure(self) -> None:
        """Failed task yields failed TaskProgressEvent."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="failed")],
        )
        planner = _mock_planner(tasks=[_task(id="t1")])
        agent_executor = _mock_agent_executor(
            _execution_result(task_id="t1", status=TaskStatus.FAILED, error="boom"),
        )
        reflection = _mock_reflection(_reflection_result(decision=ReflectionDecision.ACCEPT))

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do"),
        ):
            events.append(event)

        progress = [e for e in events if isinstance(e, TaskProgressEvent)]
        assert any(p.status == "failed" for p in progress)

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_task_exception(self) -> None:
        """Exception in executor yields failed TaskProgressEvent."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="oops")],
        )
        planner = _mock_planner(tasks=[_task(id="t1")])
        agent_executor = MagicMock(spec=AgentExecutor)
        agent_executor.execute = AsyncMock(side_effect=ValueError("bad"))
        reflection = _mock_reflection(_reflection_result(decision=ReflectionDecision.ACCEPT))

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do"),
        ):
            events.append(event)

        progress = [e for e in events if isinstance(e, TaskProgressEvent)]
        assert any(p.status == "failed" for p in progress)

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_retry_decision(self) -> None:
        """RETRY reflection adds a follow-up task."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="retried")],
        )
        planner = _mock_planner(tasks=[_task(id="t1")])
        exec_result = _execution_result(task_id="t1", output="partial")
        agent_executor = _mock_agent_executor(exec_result)
        reflection = MagicMock(spec=Reflection)
        reflection.reflect = AsyncMock(side_effect=[
            _reflection_result(decision=ReflectionDecision.RETRY, reason="incomplete"),
            _reflection_result(decision=ReflectionDecision.ACCEPT),
        ])

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
        )
        async for _ in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do"),
        ):
            pass

        assert reflection.reflect.await_count >= 2

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_replan_decision(self) -> None:
        """REPLAN adds follow-up tasks from reflection."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="replanned")],
        )
        planner = _mock_planner(tasks=[_task(id="t1")])
        agent_executor = _mock_agent_executor()
        reflection = MagicMock(spec=Reflection)
        reflection.reflect = AsyncMock(side_effect=[
            _reflection_result(
                decision=ReflectionDecision.REPLAN,
                reason="need more",
                next_tasks=["fetch more", "re-analyze"],
            ),
            _reflection_result(decision=ReflectionDecision.ACCEPT),
            _reflection_result(decision=ReflectionDecision.ACCEPT),
        ])

        # Wrap executor to return correct task_id per task
        real_execute = agent_executor.execute

        async def _execute_with_task_id(task: Task) -> ExecutionResult:
            result: ExecutionResult = await real_execute(task)
            return result.model_copy(update={"task_id": task.id})

        agent_executor.execute = _execute_with_task_id

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
        )
        async for _ in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do"),
        ):
            pass

        assert reflection.reflect.await_count >= 3  # REPLAN + 2 follow-up ACCEPTs

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_artifacts_yielded(self) -> None:
        """Artifacts from execution metadata are yielded as ArtifactEvent."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="done")],
        )
        planner = _mock_planner(tasks=[_task(id="t1")])
        exec_result = _execution_result(
            task_id="t1",
            output="data",
            metadata={
                "artifacts": [
                    {"label": "output.json", "type": "json", "data": {"key": "val"}},
                ],
            },
        )
        agent_executor = _mock_agent_executor(exec_result)
        reflection = _mock_reflection()

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do"),
        ):
            events.append(event)

        artifacts = [e for e in events if isinstance(e, ArtifactEvent)]
        assert len(artifacts) == 1
        assert artifacts[0].label == "output.json"

    @pytest.mark.asyncio
    async def test_run_agent_pipeline_no_ready_tasks(self) -> None:
        """When no tasks are ready, the inner loop breaks and final answer is streamed."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="final")],
        )
        planner = _mock_planner(tasks=[])
        agent_executor = _mock_agent_executor()
        reflection = _mock_reflection()

        coord = _make_coordinator(
            router=router,
            planner=planner,
            agent_executor=agent_executor,
            reflection=reflection,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("do"),
        ):
            events.append(event)

        assert any(isinstance(e, StreamDoneEvent) for e in events)


# ---------------------------------------------------------------------------
# Tests : ChatCoordinator — legacy pipeline (no agent components)
# ---------------------------------------------------------------------------


class TestChatCoordinatorLegacyPipeline:
    @pytest.mark.asyncio
    async def test_legacy_pipeline_no_tool_calls(self) -> None:
        """When LLM returns no tool calls, pipeline terminates."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="hello world")],
        )
        coord = _make_coordinator(
            router=router,
            registry=_mock_tool_registry(schemas=[ToolSchema(name="calc", description="calc")]),
            tool_executor=_mock_tool_executor(),
            planner=None,
            agent_executor=None,
            reflection=None,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("hi"),
        ):
            events.append(event)

        types = [e.type for e in events]
        assert "start" in types
        assert "text_delta" in types
        assert "done" in types

    @pytest.mark.asyncio
    async def test_legacy_pipeline_with_tool_calls(self) -> None:
        """Tool call blocks are executed and results fed back."""
        router = _mock_router(
            stream_events=[],
        )
        call_count = 0

        async def _gen_stream(*args: object, **kw: object) -> AsyncIterator[StreamEvent]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield TextDeltaEvent(delta="let me check")
                yield ToolCallStartEvent(tool_call_id="tc1", tool_name="calc")
                yield ToolCallEndEvent(tool_call_id="tc1", tool_name="calc", arguments={"expr": "2+2"})
            else:
                yield TextDeltaEvent(delta="the answer is 4")

        router.generate_stream = _gen_stream
        tool_exec = _mock_tool_executor(ToolResult(success=True, output="4"))

        coord = _make_coordinator(
            router=router,
            registry=_mock_tool_registry(schemas=[ToolSchema(name="calc", description="calc")]),
            tool_executor=tool_exec,
            planner=None,
            agent_executor=None,
            reflection=None,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("calculate 2+2"),
        ):
            events.append(event)

        assert any(isinstance(e, ToolProgressEvent) for e in events)
        assert any(isinstance(e, ToolResultStreamEvent) for e in events)
        tool_exec.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_legacy_pipeline_no_tool_registry(self) -> None:
        """Without a registry, tool schema injection is skipped but pipeline works."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="direct answer")],
        )
        coord = _make_coordinator(
            router=router,
            registry=None,
            tool_executor=None,
            planner=None,
            agent_executor=None,
            reflection=None,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("hello"),
        ):
            events.append(event)

        assert any(isinstance(e, TextDeltaEvent) for e in events)
        assert any(isinstance(e, StreamDoneEvent) for e in events)


# ---------------------------------------------------------------------------
# Tests : ChatCoordinator — run_nonstream
# ---------------------------------------------------------------------------


class TestChatCoordinatorNonStream:
    @pytest.mark.asyncio
    async def test_run_nonstream_returns_message(self) -> None:
        router = _mock_router(
            stream_events=[
                TextDeltaEvent(delta="Hello "),
                TextDeltaEvent(delta="world"),
            ],
        )
        coord = _make_coordinator(
            router=router,
            planner=None,
            agent_executor=None,
            reflection=None,
        )
        msg = await coord.run_nonstream(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("say hi"),
        )
        assert isinstance(msg, Message)
        assert msg.role is MessageRole.ASSISTANT


# ---------------------------------------------------------------------------
# Tests : ChatCoordinator — integration
# ---------------------------------------------------------------------------


class TestChatCoordinatorIntegration:
    @pytest.mark.asyncio
    async def test_cumulative_usage_emitted_at_end(self) -> None:
        """StreamUsageEvent from within the pipeline is re-yielded after pipeline events."""
        router = _mock_router(
            stream_events=[
                TextDeltaEvent(delta="token"),
                StreamUsageEvent(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            ],
        )
        coord = _make_coordinator(
            router=router,
            planner=None,
            agent_executor=None,
            reflection=None,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("hi"),
        ):
            events.append(event)

        usage_events = [e for e in events if isinstance(e, StreamUsageEvent)]
        assert len(usage_events) == 1
        assert usage_events[0].total_tokens == 8

    @pytest.mark.asyncio
    async def test_done_event_always_last(self) -> None:
        """StreamDoneEvent is always the final event."""
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="hello")],
        )
        coord = _make_coordinator(
            router=router,
            planner=None,
            agent_executor=None,
            reflection=None,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("hi"),
        ):
            events.append(event)

        assert isinstance(events[-1], StreamDoneEvent)

    @pytest.mark.asyncio
    async def test_start_event_always_first(self) -> None:
        router = _mock_router(
            stream_events=[TextDeltaEvent(delta="hello")],
        )
        coord = _make_coordinator(
            router=router,
            planner=None,
            agent_executor=None,
            reflection=None,
        )
        events: list[StreamEvent] = []
        async for event in coord.run(
            conversation=_conversation(),
            messages=[],
            user_message=_user_message("hi"),
        ):
            events.append(event)

        assert isinstance(events[0], StreamStartEvent)
