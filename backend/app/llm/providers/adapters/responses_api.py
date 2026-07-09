"""Responses API protocol adapter (``/responses``).

This adapter implements the :class:`ProtocolAdapter` interface for
OpenAI's newer Responses API endpoint.  The Responses API uses a
different request/response shape than Chat Completions — ``input``
array instead of ``messages``, and an ``output`` array with typed items
in the response — but produces the same domain :class:`StreamEvent`
types so that the planner and agent layers are unaffected.
"""

from __future__ import annotations

import json
from typing import Any

from app.domain.enums import OpenAIProtocol
from app.domain.message import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from app.domain.usage import Usage
from app.domain.stream import (
    StreamDoneEvent,
    StreamEvent,
    StreamUsageEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from app.llm.models import (
    CompletionRequest,
    CompletionResponse,
    FinishReason,
)
from app.llm.providers.adapters.base import ProtocolAdapter

__all__ = [
    "ResponsesApiAdapter",
    "domain_to_responses_input",
]


def domain_to_responses_input(
    messages: list[Message],
) -> list[dict[str, Any]]:
    """Convert domain messages to Responses API ``input`` array format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role.value}

        texts: list[str] = []
        has_multimodal = any(isinstance(b, ImageBlock) for b in msg.content)

        if has_multimodal:
            content_list: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "input_text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append({
                        "type": "input_image",
                        "image_url": block.data_uri,
                    })
                elif isinstance(block, ToolCallBlock):
                    entry.setdefault("tool_calls", []).append({
                        "id": block.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": json.dumps(block.arguments),
                        },
                    })
                elif isinstance(block, ToolResultBlock):
                    result.append({
                        "role": "tool",
                        "tool_call_id": block.tool_call_id,
                        "content": block.output,
                    })
                    continue
            if content_list:
                entry["content"] = content_list
        else:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    texts.append(block.text)
                elif isinstance(block, ToolCallBlock):
                    entry.setdefault("tool_calls", []).append({
                        "id": block.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": json.dumps(block.arguments),
                        },
                    })
                elif isinstance(block, ToolResultBlock):
                    result.append({
                        "role": "tool",
                        "tool_call_id": block.tool_call_id,
                        "content": block.output,
                    })
                    continue
            if texts:
                entry["content"] = "\n".join(texts)

        result.append(entry)
    return result


class ResponsesApiAdapter(ProtocolAdapter):
    """Adapter for the ``/responses`` API protocol."""

    @property
    def protocol(self) -> OpenAIProtocol:
        return OpenAIProtocol.RESPONSES

    @property
    def endpoint(self) -> str:
        return "/responses"

    def build_request(
        self,
        request: CompletionRequest,
        model: str,
    ) -> dict[str, Any]:
        input_data = domain_to_responses_input(request.messages)
        payload: dict[str, Any] = {
            "model": model,
            "input": input_data,
            "stream": True,
            "temperature": request.params.temperature,
            "max_output_tokens": request.params.max_tokens,
        }
        return payload

    def parse_response(
        self,
        raw: dict[str, Any],
        model: str,
    ) -> CompletionResponse:
        content_blocks: list[ContentBlock] = []

        for item in raw.get("output", []):
            item_type = item.get("type", "")
            if item_type == "message":
                content = item.get("content", "") or ""
                if content:
                    content_blocks.append(TextBlock(text=content))
            elif item_type == "function_call":
                func = item.get("function", {})
                try:
                    arguments = json.loads(func.get("arguments", "{}")) if func.get("arguments") else {}
                except json.JSONDecodeError:
                    arguments = {}
                content_blocks.append(
                    ToolCallBlock(
                        tool_call_id=item.get("id", ""),
                        tool_name=func.get("name", "unknown"),
                        arguments=arguments,
                    ),
                )

        usage_data = raw.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=(
                usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0)
            ),
        ) if usage_data.get("input_tokens") else None

        from datetime import UTC, datetime
        if not content_blocks:
            content_blocks.append(TextBlock(text=" "))

        assistant_msg = Message(
            id="",
            conversation_id="",
            role="assistant",
            content=content_blocks,
            created_at=datetime.now(UTC),
        )

        finish_reason = FinishReason.STOP
        status = raw.get("status", "")
        if status == "incomplete":
            finish_reason = FinishReason.LENGTH
        elif status == "failed":
            finish_reason = FinishReason.ERROR

        return CompletionResponse(
            message=assistant_msg,
            usage=usage,
            finish_reason=finish_reason,
            model=model,
        )

    def parse_stream_chunk(
        self,
        chunk: dict[str, Any],
        tool_call_state: dict[int, dict[str, str]] | None = None,
    ) -> tuple[list[StreamEvent], dict[int, dict[str, str]]]:
        if tool_call_state is None:
            tool_call_state = {}

        events: list[StreamEvent] = []
        event_type = chunk.get("type", "")

        # Text delta events
        if event_type == "response.output_text.delta":
            delta = chunk.get("delta", "")
            if delta:
                events.append(TextDeltaEvent(delta=delta))

        # Function call argument deltas
        elif event_type == "response.function_call_arguments.delta":
            delta = chunk.get("delta", "")
            call_id = chunk.get("item_id", "")
            if delta and call_id:
                events.append(ToolCallDeltaEvent(
                    tool_call_id=call_id,
                    arguments_delta=delta,
                ))

        # Function call start
        elif event_type == "response.function_call_arguments.done":
            call_id = chunk.get("item_id", "")
            name = chunk.get("name", "unknown")
            arguments_str = chunk.get("arguments", "{}")
            try:
                parsed_args = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError:
                parsed_args = {}
            if call_id:
                events.append(ToolCallEndEvent(
                    tool_call_id=call_id,
                    tool_name=name,
                    arguments=parsed_args,
                ))

        # Stream completed
        elif event_type == "response.completed":
            response = chunk.get("response", {})
            usage_data = response.get("usage") or {}
            pt = usage_data.get("input_tokens", 0)
            ct = usage_data.get("output_tokens", 0)
            tt = pt + ct
            if pt or ct:
                events.append(StreamUsageEvent(
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                ))
            status = response.get("status", "")
            finish_reason = FinishReason.STOP
            if status == "incomplete":
                finish_reason = FinishReason.LENGTH
            elif status == "failed":
                finish_reason = FinishReason.ERROR
            events.append(StreamDoneEvent(finish_reason=finish_reason))

        # In-progress responses (used for tool call tracking)
        elif event_type == "response.in_progress":
            pass  # No action needed — used for metadata

        return events, tool_call_state

    def parse_stream_chunk_done(
        self,
        chunk: dict[str, Any],
    ) -> bool:
        return chunk.get("type") == "response.completed"
