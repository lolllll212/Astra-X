"""Chat coordinator — the orchestration pipeline for every chat interaction.

The :class:`ChatCoordinator` sits between the service layer and the LLM
router. It is responsible for:

* **Stream orchestration** — yielding the correct sequence of events for
  every chat interaction.
* **Agent pipeline** — planning, parallel task execution, and reflection
  when agent components are provided.
* **Tool execution** — intercepting tool calls from the LLM, invoking
  the tool via the :class:`ToolExecutor`, and feeding the result back.
* **Multi-step reasoning** — driving the plan -> execute -> reflect loop
  until the goal is achieved or max iterations is reached.
* **Pipeline enrichment** — inserting ``StreamStartEvent`` before the
  first event and ``StreamUsageEvent`` / ``StreamDoneEvent`` after.

The coordinator is itself a pipeline: it receives a prepared context
and yields typed :class:`StreamEvent` instances. No layer above the
coordinator buffers the full response — everything flows through async
generators end-to-end.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from app.agents.base import AgentConfig
from app.agents.executor import Executor as AgentExecutor
from app.agents.learning_manager import LearningManager
from app.agents.memory_manager import MemoryManager
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.task import Task, TaskStatus
from app.agents.planner import Planner
from app.agents.reflection import Reflection
from app.agents.task_graph import TaskGraph
from app.core.logging import get_logger
from app.core.tracing import get_tracer
from app.domain.conversation import Conversation
from app.domain.enums import MessageRole
from app.domain.message import (
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from app.domain.stream import (
    ArtifactEvent,
    PlanEvent,
    PlannedTaskSchema,
    ReflectionEvent,
    StreamDoneEvent,
    StreamEvent,
    StreamMetadataEvent,
    StreamStartEvent,
    StreamUsageEvent,
    TaskProgressEvent,
    TextDeltaEvent,
    ToolProgressEvent,
    ToolResultStreamEvent,
)
from app.llm.models import CompletionRequest, GenerationParams
from app.llm.router import LLMRouter
from app.tools.context import ToolContext as ToolExecContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from app.tools.capabilities import CapabilityRegistry

logger = get_logger(__name__)

_MAX_TOOL_ITERATIONS = 10
_MAX_AGENT_ITERATIONS = 5


class ChatCoordinator:
    """Orchestrates a single chat interaction as an event pipeline.

    Supports two modes:

    * **Agent mode** — when ``planner``, ``executor``, and ``reflection``
      are provided, the coordinator runs the full plan -> execute -> reflect
      loop, yielding ``PlanEvent``, ``TaskProgressEvent``, ``ReflectionEvent``,
      and ``ArtifactEvent``.
    * **Legacy mode** — falls back to the simple tool-calling loop when
      no agent components are given.

    Usage::

        coordinator = ChatCoordinator(
            llm_router=router,
            tool_registry=registry,
            tool_executor=executor,
            planner=planner,
            executor=agent_executor,
            reflection=reflection,
            memory_manager=memory_manager,
        )
        async for event in coordinator.run(
            conversation=conversation,
            messages=messages,
            user_message=user_message,
        ):
            ...
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        planner: Planner | None = None,
        agent_executor: AgentExecutor | None = None,
        reflection: Reflection | None = None,
        memory_manager: MemoryManager | None = None,
        learning_manager: LearningManager | None = None,
        capability_registry: CapabilityRegistry | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._planner = planner
        self._agent_executor = agent_executor
        self._reflection = reflection
        self._memory_manager = memory_manager
        self._learning_manager = learning_manager
        self._capability_registry = capability_registry
        self._agent_config = agent_config or AgentConfig()

    @property
    def _has_agent_pipeline(self) -> bool:
        """Whether agent components have been wired in."""
        return all((self._planner, self._agent_executor, self._reflection))

    async def run(
        self,
        *,
        conversation: Conversation,
        messages: list[Message],
        user_message: Message,
        model: str | None = None,
        provider: str | None = None,
        params: GenerationParams | None = None,
        assistant_message_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run the full chat pipeline, yielding stream events.

        The pipeline:

        1. Yields ``StreamStartEvent``.
        2. Yields ``StreamMetadataEvent`` with conversation / model info.
        3. If agent components are wired: runs the plan -> execute -> reflect
           loop, yielding plan, task progress, reflection, and artifact events,
           then streams the final answer as text deltas.
        4. Otherwise: runs the legacy tool-calling loop.
        5. Yields ``StreamUsageEvent`` (cumulative).
        6. Yields ``StreamDoneEvent``.

        Args:
            conversation: The conversation being continued.
            messages: Full message history for context building.
            user_message: The user's latest message.
            model: Model override.
            provider: Provider override.
            params: Generation parameter overrides.
            assistant_message_id: Pre-generated ID for the assistant message.

        Yields:
            Stream events for the frontend or for collection.
        """

        msg_id = assistant_message_id or str(uuid4())

        yield StreamStartEvent()

        yield StreamMetadataEvent(
            conversation_id=conversation.id,
            message_id=msg_id,
            model=model or conversation.metadata.model or "",
            provider=provider or conversation.metadata.provider or "",
        )

        cumulative_usage: StreamUsageEvent | None = None

        if self._has_agent_pipeline:
            async for event in self._run_agent_pipeline(
                conversation=conversation,
                messages=messages,
                user_message=user_message,
                model=model,
                provider=provider,
                params=params,
            ):
                match event:
                    case StreamUsageEvent():
                        cumulative_usage = event
                        continue
                    case _:
                        yield event
        else:
            async for event in self._run_legacy_pipeline(
                conversation=conversation,
                messages=messages,
                model=model,
                provider=provider,
                params=params,
            ):
                match event:
                    case StreamUsageEvent():
                        cumulative_usage = event
                        continue
                    case _:
                        yield event

        if cumulative_usage is not None:
            yield cumulative_usage
        yield StreamDoneEvent(finish_reason="stop")

    async def run_nonstream(
        self,
        *,
        conversation: Conversation,
        messages: list[Message],
        user_message: Message,
        model: str | None = None,
        provider: str | None = None,
        params: GenerationParams | None = None,
        assistant_message_id: str | None = None,
    ) -> Message:
        """Run the pipeline in non-streaming mode.

        Args:
            Same as :meth:`run`.

        Returns:
            The assistant message built from the collected stream.
        """
        from app.llm.streaming import collect_stream

        collector = await collect_stream(
            self.run(
                conversation=conversation,
                messages=messages,
                user_message=user_message,
                model=model,
                provider=provider,
                params=params,
                assistant_message_id=assistant_message_id,
            ),
            conversation_id=conversation.id,
        )
        return collector.build_message()

    # ------------------------------------------------------------------
    # Agent pipeline — plan, execute, reflect, stream answer
    # ------------------------------------------------------------------

    async def _run_agent_pipeline(
        self,
        *,
        conversation: Conversation,
        messages: list[Message],
        user_message: Message,
        model: str | None,
        provider: str | None,
        params: GenerationParams | None,
    ) -> AsyncIterator[StreamEvent]:
        """Full agent pipeline: plan -> execute (parallel) -> reflect -> answer."""
        goal = _extract_user_goal(user_message)
        tracer = get_tracer()
        memory_context: str = ""

        patterns_context = ""
        if self._learning_manager is not None:
            with tracer.span("Pattern Retrieval", category="learning"):
                patterns_context = await self._learning_manager.get_lessons(goal)

        with tracer.span("Request", category="chat", conversation_id=conversation.id):
            if self._memory_manager is not None:
                with tracer.span("Memory Retrieval", category="memory"):
                    ctx = await self._memory_manager.get_context(
                        conversation_id=conversation.id,
                        goal=goal,
                    )
                    if ctx is not None:
                        memory_context = ctx

            iteration = 0
            all_task_outputs: dict[str, list[str]] = {}

            while iteration < _MAX_AGENT_ITERATIONS:
                iteration += 1

                # Step 1 — Plan (always uses memory + patterns context).
                assert self._planner is not None  # guarded by _has_agent_pipeline
                with tracer.span("Planner", category="agent"):
                    plan = await self._planner.plan(
                        goal=goal,
                        memory_context=memory_context,
                        patterns_context=patterns_context,
                    )

                yield PlanEvent(
                    goal=plan.goal,
                    tasks=[
                        PlannedTaskSchema(
                            id=t.id,
                            description=t.description,
                            status=t.status.value,
                            dependencies=list(t.dependencies),
                            tool_name=t.tool_name,
                            capability=t.capability,
                        )
                        for t in plan.tasks
                    ],
                    iteration=iteration - 1,
                )

                # Step 2 — Execute tasks (parallel via TaskGraph)
                graph = TaskGraph()
                for task in plan.tasks:
                    graph.add_task(task, depends_on=task.dependencies)

                completed_results: dict[str, ExecutionResult] = {}
                reflection = ReflectionResult(
                    decision=ReflectionDecision.ACCEPT,
                    reason="Initial execution.",
                    confidence=0.0,
                )

                while not graph.is_complete():
                    ready_tasks = graph.get_ready()
                    if not ready_tasks:
                        logger.warning("coordinator.no_ready_tasks")
                        break

                    for task in ready_tasks:
                        graph.update_status(task.id, TaskStatus.RUNNING)
                        yield TaskProgressEvent(
                            task_id=task.id,
                            description=task.description,
                            status="running",
                        )

                    # Execute ready tasks concurrently.
                    async def run_task(task: Task) -> ExecutionResult:
                        with tracer.span(f"Execute {task.id}", category="execution", task_id=task.id):
                            result = await self._agent_executor.execute(task)  # type: ignore[union-attr]
                            return result

                    tasks_with_ids = [(t, run_task(t)) for t in ready_tasks]
                    results = await asyncio.gather(
                        *(coro for _, coro in tasks_with_ids),
                        return_exceptions=True,
                    )

                    for task, result_or_exc in zip(ready_tasks, results, strict=False):
                        if isinstance(result_or_exc, BaseException):
                            graph.update_status(task.id, TaskStatus.FAILED)
                            yield TaskProgressEvent(
                                task_id=task.id,
                                description=task.description,
                                status="failed",
                                error=str(result_or_exc),
                            )
                            continue

                        completed_results[result_or_exc.task_id] = result_or_exc
                        task_outputs = all_task_outputs.setdefault(task.id, [])
                        if result_or_exc.output:
                            task_outputs.append(result_or_exc.output)

                        new_status = (
                            TaskStatus.COMPLETED
                            if result_or_exc.status is TaskStatus.COMPLETED
                            else TaskStatus.FAILED
                        )
                        graph.update_status(task.id, new_status)

                        yield TaskProgressEvent(
                            task_id=task.id,
                            description=task.description,
                            status=new_status.value,
                            result=result_or_exc.output,
                            error=result_or_exc.error,
                        )

                        # Yield any artifacts from the execution metadata.
                        task_artifacts = result_or_exc.metadata.get("artifacts", [])
                        if isinstance(task_artifacts, list):
                            for artifact in task_artifacts:
                                if isinstance(artifact, dict):
                                    yield ArtifactEvent(
                                        task_id=task.id,
                                        label=artifact.get("label", ""),
                                        artifact_type=artifact.get("type", "json"),
                                        data=artifact.get("data", artifact),
                                        metadata=artifact.get("metadata", {}),
                                    )

                        # Store result in memory if manager is available.
                        if self._memory_manager is not None:
                            with tracer.span("Memory Store", category="memory"):
                                await self._memory_manager.store_result(
                                    conversation_id=conversation.id,
                                    result=result_or_exc,
                                )

                    # Step 3 — Reflect on this batch.
                    for task in ready_tasks:
                        if task.id in completed_results:
                            with tracer.span("Reflection", category="agent"):
                                assessment = await self._reflection.reflect(  # type: ignore[union-attr]
                                    completed_results[task.id],
                                )
                            reflection = assessment
                            yield ReflectionEvent(
                                decision=assessment.decision.value,
                                feedback=assessment.feedback,
                                reason=assessment.reason,
                                confidence=assessment.confidence,
                                iteration=iteration - 1,
                            )

                            self._handle_reflection_decision(
                                decision=assessment.decision,
                                assessment=assessment,
                                graph=graph,
                                task=task,
                            )

                # Check overall plan completeness.
                with tracer.span("Plan-level Reflection", category="agent"):
                    if reflection.decision is ReflectionDecision.ACCEPT:
                        break

        # Step 4 — Extract learning pattern if execution was successful.
        if self._learning_manager is not None and completed_results:
            with tracer.span("Pattern Extraction", category="learning"):
                await self._learning_manager.extract_pattern(
                    goal=goal,
                    plan=plan,
                    results=list(completed_results.values()),
                    reflections=[],
                )

        # Step 5 — Stream the final answer.
        async for event in self._stream_final_answer(
            goal=goal,
            completed_results=completed_results,
            conversation=conversation,
            model=model,
            provider=provider,
            params=params,
        ):
            yield event

    @staticmethod
    def _handle_reflection_decision(
        decision: ReflectionDecision,
        assessment: ReflectionResult,
        graph: TaskGraph,
        task: Task,
    ) -> None:
        """Mutate the task graph based on a reflection decision."""
        match decision:
            case ReflectionDecision.RETRY:
                follow_up = Task(
                    id=str(uuid4()),
                    description=task.description,
                    dependencies=[task.id],
                    capability=task.capability,
                    tool_name=task.tool_name,
                )
                graph.add_task(follow_up, depends_on=[task.id])

            case ReflectionDecision.REPLAN:
                for desc in assessment.next_tasks:
                    follow_up = Task(
                        id=str(uuid4()),
                        description=desc,
                        dependencies=[task.id],
                    )
                    graph.add_task(follow_up, depends_on=[task.id])

            case _:
                pass

    # ------------------------------------------------------------------
    # Legacy pipeline — simple tool-calling loop
    # ------------------------------------------------------------------

    async def _run_legacy_pipeline(
        self,
        *,
        conversation: Conversation,
        messages: list[Message],
        model: str | None,
        provider: str | None,
        params: GenerationParams | None,
    ) -> AsyncIterator[StreamEvent]:
        """Simple tool-calling loop for backwards compatibility."""
        prompt_messages = _inject_tool_schemas(list(messages), self._tool_registry)

        for _ in range(_MAX_TOOL_ITERATIONS):
            llm_request = CompletionRequest(
                messages=prompt_messages,
                model=model or conversation.metadata.model or "",
                provider=provider or conversation.metadata.provider,
                params=params or GenerationParams(),
                stream=True,
            )

            from app.llm.streaming import StreamCollector

            collector = StreamCollector(conversation_id=conversation.id)

            async for event in self._llm_router.generate_stream(llm_request):
                collector.feed(event)
                yield event

            assistant_message = collector.build_message()
            tool_call_blocks = [
                b for b in assistant_message.content
                if isinstance(b, ToolCallBlock)
            ]

            if not tool_call_blocks:
                return

            prompt_messages.append(assistant_message)

            for tc_block in tool_call_blocks:
                yield ToolProgressEvent(
                    tool_name=tc_block.tool_name,
                    status="running",
                    message=f"Executing {tc_block.tool_name}...",
                )

                tool_context = ToolExecContext(
                    conversation_id=conversation.id,
                    logger=logger,  # type: ignore[arg-type]
                )

                from app.tools.models import ToolCall as ToolCallModel

                tool_call_model = ToolCallModel(
                    tool_name=tc_block.tool_name,
                    arguments=tc_block.arguments,
                )

                result = await self._tool_executor.execute(tool_call_model, tool_context)  # type: ignore[union-attr]

                yield ToolResultStreamEvent(
                    tool_name=tc_block.tool_name,
                    tool_call_id=tc_block.tool_call_id,
                    output=result.output,
                    is_error=not result.success,
                    duration_ms=result.execution_time_ms,
                )

                tool_result_msg = Message(
                    id=str(uuid4()),
                    conversation_id=conversation.id,
                    role=MessageRole.TOOL,
                    content=[
                        ToolResultBlock(
                            tool_call_id=tc_block.tool_call_id,
                            tool_name=tc_block.tool_name,
                            output=result.output,
                            is_error=not result.success,
                        ),
                    ],
                    created_at=datetime.now(UTC),
                )
                prompt_messages.append(tool_result_msg)

        else:
            logger.warning(
                "coordinator.max_iterations_reached",
                iterations=_MAX_TOOL_ITERATIONS,
            )

    # ------------------------------------------------------------------
    # Final answer synthesis
    # ------------------------------------------------------------------

    async def _stream_final_answer(
        self,
        *,
        goal: str,
        completed_results: dict[str, ExecutionResult],
        conversation: Conversation,
        model: str | None,
        provider: str | None,
        params: GenerationParams | None,
    ) -> AsyncIterator[StreamEvent]:
        """Synthesize collected task outputs into a streaming final answer."""
        parts: list[str] = []
        for result in completed_results.values():
            if result.output:
                parts.append(result.output)

        if not parts:
            yield TextDeltaEvent(
                delta="I was unable to produce a result. Please try rephrasing your request.",
            )
            return

        # Use the LLM to generate a polished final answer from collected results.
        synthesis_prompt = (
            "You have completed research or analysis on the following goal:\n\n"
            f"GOAL: {goal}\n\n"
            "Here are the collected findings:\n\n"
            + "\n\n".join(parts)
            + "\n\n"
            "Synthesize these findings into a clear, well-structured final answer "
            "for the user. Do not mention that you are synthesizing — just provide "
            "the answer directly."
        )

        synthesis_message = Message(
            id=str(uuid4()),
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=[TextBlock(text=synthesis_prompt)],
            created_at=datetime.now(UTC),
        )

        llm_request = CompletionRequest(
            messages=[synthesis_message],
            model=model or conversation.metadata.model or "",
            provider=provider or conversation.metadata.provider,
            params=params or GenerationParams(),
            stream=True,
        )

        from app.llm.streaming import StreamCollector

        collector = StreamCollector(conversation_id=conversation.id)

        async for event in self._llm_router.generate_stream(llm_request):
            collector.feed(event)
            yield event


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_user_goal(user_message: Message) -> str:
    """Extract the user's goal text from their message."""
    for block in user_message.content:
        if isinstance(block, TextBlock):
            return block.text
    return ""


def _inject_tool_schemas(
    messages: list[Message],
    registry: ToolRegistry | None,
) -> list[Message]:
    """Append a system message describing available tool schemas.

    This is a provider-independent way to tell the LLM what tools are
    available.  Every registered tool's schema is rendered as a JSON
    array and attached to the message list.

    Args:
        messages: The current message list to extend.
        registry: The tool registry, or ``None``.

    Returns:
        A new list with the tool schemas message appended (or the
        original list unchanged if *registry* is ``None`` or empty).
    """
    if registry is None or registry.count == 0:
        return messages

    schemas = registry.schemas()
    schema_lines: list[str] = [
        "You have access to the following tools. "
        "When you need to use a tool, respond with the appropriate tool call.",
        "",
        "```json",
    ]
    for s in schemas:
        params_list = [
            {
                "name": p.name,
                "type": p.type_,
                "description": p.description,
                "required": p.required,
            }
            for p in s.parameters
        ]
        schema_lines.append(
            f'  {{"name": "{s.name}", '
            f'"description": "{s.description}", '
            f'"parameters": {params_list}}}'
        )
    schema_lines.append("```")

    tool_message = Message(
        id=str(uuid4()),
        conversation_id=messages[0].conversation_id if messages else "",
        role=MessageRole.SYSTEM,
        content=[TextBlock(text="\n".join(schema_lines))],
        created_at=datetime.now(UTC),
    )

    result = list(messages)
    result.append(tool_message)
    return result
