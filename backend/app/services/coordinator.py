"""Chat coordinator — the orchestration pipeline for every chat interaction.

The :class:`ChatCoordinator` sits between the service layer and the LLM
router. It is responsible for:

* **Stream orchestration** — yielding the correct sequence of events for
  every chat interaction.
* **Tool execution** — intercepting ``ToolCallEndEvent`` from the router,
  invoking the tool, and injecting ``ToolResultBlock`` back into the
  conversation (future).
* **Multi-step reasoning** — driving the think → act → observe loop if the
  provider supports it (future).
* **Pipeline enrichment** — inserting ``StreamStartEvent`` before the
  first event and ``StreamUsageEvent`` / ``StreamDoneEvent`` after.

The coordinator is itself a pipeline: it receives a prepared context
and yields typed :class:`StreamEvent` instances. No layer above the
coordinator buffers the full response — everything flows through async
generators end-to-end.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from app.core.logging import get_logger
from app.domain.conversation import Conversation
from app.domain.message import Message
from app.domain.stream import (
    StreamEvent,
    StreamMetadataEvent,
    StreamStartEvent,
)
from app.llm.models import CompletionRequest, CompletionResponse, GenerationParams
from app.llm.router import LLMRouter

logger = get_logger(__name__)


class ChatCoordinator:
    """Orchestrates a single chat interaction as an event pipeline.

    Usage::

        coordinator = ChatCoordinator(llm_router=router)
        async for event in coordinator.run(
            conversation=conversation,
            messages=messages,
            user_message=user_message,
            model=model,
            provider=provider,
            params=params,
        ):
            ...
    """

    def __init__(
        self,
        llm_router: LLMRouter,
    ) -> None:
        self._llm_router = llm_router

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
        3. Yields raw events from the LLM router's ``generate_stream``.
        4. Yields ``StreamDoneEvent`` once generation is complete.

        In the future steps 2-4 will be interleaved with tool execution
        and multi-step reasoning.

        Args:
            conversation: The conversation being continued.
            messages: Full message history for context building.
            user_message: The user's latest message.
            model: Model override.
            provider: Provider override.
            params: Generation parameter overrides.
            assistant_message_id: Pre-generated ID for the assistant message.
                Generated automatically if omitted.

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

        llm_request = CompletionRequest(
            messages=messages,
            model=model or conversation.metadata.model or "",
            provider=provider or conversation.metadata.provider,
            params=params or GenerationParams(),
            stream=True,
        )

        async for event in self._llm_router.generate_stream(llm_request):
            yield event

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
            Same as :meth:`run`, plus *assistant_message_id*.

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
