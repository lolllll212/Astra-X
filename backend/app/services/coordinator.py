"""Chat coordinator — the orchestration pipeline for every chat interaction.

The :class:`ChatCoordinator` sits between the service layer and the LLM
router. It is responsible for:

* **Stream orchestration** — yielding the correct sequence of events for
  every chat interaction.
* **Tool execution** — intercepting ``ToolCallBlock`` from the LLM,
  invoking the tool via the :class:`ToolExecutor`, and feeding the result
  back to the LLM.
* **Multi-step reasoning** — driving the think → act → observe loop until
  the LLM produces a final answer, max iterations is reached, or no tools
  are called.
* **Pipeline enrichment** — inserting ``StreamStartEvent`` before the
  first event and ``StreamUsageEvent`` / ``StreamDoneEvent`` after.

The coordinator is itself a pipeline: it receives a prepared context
and yields typed :class:`StreamEvent` instances. No layer above the
coordinator buffers the full response — everything flows through async
generators end-to-end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

from app.core.logging import get_logger
from app.domain.conversation import Conversation
from app.domain.enums import MessageRole
from app.domain.message import (
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from app.domain.stream import (
    StreamDoneEvent,
    StreamEvent,
    StreamMetadataEvent,
    StreamStartEvent,
    StreamUsageEvent,
    ToolProgressEvent,
    ToolResultStreamEvent,
)
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter
from app.tools.context import ToolContext as ToolExecContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)

_MAX_TOOL_ITERATIONS = 10


class ChatCoordinator:
    """Orchestrates a single chat interaction as an event pipeline.

    Usage::

        coordinator = ChatCoordinator(
            llm_router=router,
            tool_registry=registry,
            tool_executor=executor,
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
    ) -> None:
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor

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
        3. Enters the tool-calling loop:
           a. Sends messages (with tool schemas) to the LLM via streaming.
           b. Collects events — text deltas, tool calls, usage.
           c. If the LLM emitted tool calls: yields ``ToolProgressEvent``
              and ``ToolResultStreamEvent``, feeds results back, repeats.
           d. If no tool calls: the message is the final answer.
        4. Yields ``StreamUsageEvent`` (cumulative).
        5. Yields ``StreamDoneEvent``.

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

        prompt_messages = list(messages)

        # Inject tool schemas so the LLM knows what it can call.
        prompt_messages = _inject_tool_schemas(prompt_messages, self._tool_registry)

        cumulative_usage: StreamUsageEvent | None = None

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

                match event:
                    case StreamUsageEvent():
                        cumulative_usage = event
                        continue
                    case StreamDoneEvent():
                        continue
                    case _:
                        yield event

            assistant_message = collector.build_message()
            tool_call_blocks = [
                b for b in assistant_message.content
                if isinstance(b, ToolCallBlock)
            ]

            if not tool_call_blocks:
                break

            # Add the assistant message (with tool calls) to the history.
            prompt_messages.append(assistant_message)

            # Execute each tool and feed results back.
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
    ) -> CompletionResponse:
        """Run the pipeline in non-streaming mode.

        Args:
            Same as :meth:`run`.

        Returns:
            A complete :class:`CompletionResponse`.
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

        return CompletionResponse(
            message=collector.build_message(),
            usage=collector.usage,
            finish_reason=collector.finish_reason,
            model=model or conversation.metadata.model or "",
        )


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
