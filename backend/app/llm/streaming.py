"""Streaming helpers for LLM responses.

Provides utilities to consume, accumulate, and transform streaming
events from provider adapters.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

from app.domain.enums import MessageRole
from app.domain.message import ContentBlock, Message, TextBlock, ToolCallBlock
from app.domain.stream import (
    StreamDoneEvent,
    StreamEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from app.domain.usage import Usage
from app.llm.models import FinishReason


class StreamCollector:
    """Accumulates streaming events and builds a final :class:`CompletionResponse`.

    Usage::

        collector = StreamCollector(conversation_id="...", message_id="...")
        async for event in provider.generate_stream(request):
            collector.feed(event)
        response = collector.build_response(model="llama3.1")
    """

    def __init__(
        self,
        conversation_id: str,
        message_id: str | None = None,
    ) -> None:
        self._conversation_id = conversation_id
        self._message_id = message_id or str(uuid4())
        self._text_parts: list[str] = []
        self._tool_calls: dict[str, dict[str, object]] = {}
        self._tool_call_names: dict[str, str] = {}
        self._tool_call_raw_args: dict[str, str] = {}
        self._finish_reason: str = FinishReason.NULL
        self._usage: Usage | None = None

    def feed(self, event: StreamEvent) -> None:
        """Process a single stream event.

        Args:
            event: An event yielded by a provider adapter.
        """
        match event:
            case TextDeltaEvent(delta=delta):
                self._text_parts.append(delta)

            case ToolCallStartEvent(tool_call_id=tcid, tool_name=name):
                self._tool_calls[tcid] = {}
                self._tool_call_names[tcid] = name

            case ToolCallDeltaEvent(tool_call_id=tcid, arguments_delta=delta):
                if tcid in self._tool_calls:
                    existing = self._tool_call_raw_args.get(tcid, "")
                    self._tool_call_raw_args[tcid] = existing + delta

            case ToolCallEndEvent(tool_call_id=tcid, tool_name=name, arguments=args):
                self._tool_calls[tcid] = dict(args)
                self._tool_call_names[tcid] = name

            case StreamDoneEvent(finish_reason=reason, usage=usage):
                if reason is not None:
                    self._finish_reason = reason
                if usage is not None:
                    self._usage = Usage(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    )

    def build_message(self) -> Message:
        """Build the assistant message from all accumulated events.

        Returns:
            A complete :class:`Message` with text and tool-call content
            blocks.
        """
        content: list[ContentBlock] = []

        full_text = "".join(self._text_parts)
        if full_text:
            content.append(TextBlock(text=full_text))

        for tcid, args in self._tool_calls.items():
            name = self._tool_call_names.get(tcid, "unknown")
            content.append(
                ToolCallBlock(
                    tool_call_id=tcid,
                    tool_name=name,
                    arguments=dict(args),
                ),
            )

        return Message(
            id=self._message_id,
            conversation_id=self._conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            created_at=datetime.now(UTC),
        )

    @property
    def usage(self) -> Usage | None:
        """Token usage accumulated from the stream, if provided."""
        return self._usage

    @property
    def finish_reason(self) -> str:
        """Finish reason from the stream, or ``"null"`` if not yet known."""
        return self._finish_reason


async def collect_stream(
    stream: AsyncIterator[StreamEvent],
    conversation_id: str,
    message_id: str | None = None,
) -> StreamCollector:
    """Consume an entire stream iterator and return the collector.

    Args:
        stream: An async iterator of stream events.
        conversation_id: The conversation these events belong to.
        message_id: Optional message ID; auto-generated if omitted.

    Returns:
        A :class:`StreamCollector` with all events accumulated.
    """
    collector = StreamCollector(
        conversation_id=conversation_id,
        message_id=message_id,
    )
    async for event in stream:
        collector.feed(event)
    return collector
