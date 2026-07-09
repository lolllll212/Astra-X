"""Chat Completions protocol adapter (``/chat/completions``).

This adapter implements the :class:`ProtocolAdapter` interface for the
OpenAI Chat Completions API (SSE-based streaming).  The same wire format
is used by OpenAI, Azure OpenAI, Together AI, Groq, LM Studio, and most
other OpenAI-compatible backends.
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
from app.domain.usage import Usage
from app.llm.providers.adapters.base import ProtocolAdapter

__all__ = [
    "ChatCompletionsAdapter",
    "domain_to_openai_messages",
    "parse_finish_reason",
]


def domain_to_openai_messages(
    messages: list[Message],
) -> list[dict[str, Any]]:
    """Convert domain messages to OpenAI-compatible message format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role.value}

        texts: list[str] = []
        has_multimodal = any(isinstance(b, ImageBlock) for b in msg.content)

        if has_multimodal:
            content_list: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": block.data_uri},
                    })
                elif isinstance(block, ToolCallBlock):
                    entry.setdefault("tool_calls", []).append({
                        "id": block.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": block.tool_name,
                            "arguments": block.arguments,
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
                            "arguments": block.arguments,
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


def parse_finish_reason(reason: str | None) -> str:
    """Map an OpenAI finish reason to a domain :class:`FinishReason`."""
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(reason or "", FinishReason.STOP)


class ChatCompletionsAdapter(ProtocolAdapter):
    """Adapter for the ``/chat/completions`` SSE streaming protocol."""

    @property
    def protocol(self) -> OpenAIProtocol:
        return OpenAIProtocol.CHAT_COMPLETIONS

    @property
    def endpoint(self) -> str:
        return "/chat/completions"

    def build_request(
        self,
        request: CompletionRequest,
        model: str,
    ) -> dict[str, Any]:
        messages = domain_to_openai_messages(request.messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
            "max_tokens": request.params.max_tokens,
            "stop": request.params.stop or None,
        }
        return payload

    def parse_response(
        self,
        raw: dict[str, Any],
        model: str,
    ) -> CompletionResponse:
        choice = raw.get("choices", [{}])[0]
        message_data = choice.get("message", {})

        content_blocks: list[ContentBlock] = []
        content = message_data.get("content", "") or ""
        if content:
            content_blocks.append(TextBlock(text=content))

        for tc in message_data.get("tool_calls") or []:
            func = tc.get("function", {})
            try:
                arguments = json.loads(func.get("arguments", "{}")) if func.get("arguments") else {}
            except json.JSONDecodeError:
                arguments = {}
            content_blocks.append(
                ToolCallBlock(
                    tool_call_id=tc.get("id", ""),
                    tool_name=func.get("name", "unknown"),
                    arguments=arguments,
                ),
            )

        usage_data = raw.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ) if usage_data.get("prompt_tokens") else None

        if not content_blocks:
            content_blocks.append(TextBlock(text=" "))

        from datetime import UTC, datetime
        assistant_msg = Message(
            id="",
            conversation_id="",
            role="assistant",
            content=content_blocks,
            created_at=datetime.now(UTC),
        )

        return CompletionResponse(
            message=assistant_msg,
            usage=usage,
            finish_reason=parse_finish_reason(choice.get("finish_reason")),
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
        choices = chunk.get("choices", [])
        if not choices:
            return events, tool_call_state

        choice = choices[0]
        delta = choice.get("delta", {})

        content_delta = delta.get("content", "")
        if content_delta:
            events.append(TextDeltaEvent(delta=content_delta))

        for tc_delta in delta.get("tool_calls") or []:
            idx = tc_delta.get("index", 0)
            func = tc_delta.get("function", {})
            tcid = tc_delta.get("id", "")

            if idx not in tool_call_state:
                tool_call_state = dict(tool_call_state)
                tool_call_state[idx] = {
                    "id": tcid,
                    "name": func.get("name", ""),
                    "arguments": "",
                }
                events.append(ToolCallStartEvent(
                    tool_call_id=tcid,
                    tool_name=func.get("name", "unknown"),
                ))
            else:
                args_delta = func.get("arguments", "")
                if args_delta:
                    tool_call_state = dict(tool_call_state)
                    tool_call_state[idx] = dict(tool_call_state[idx])
                    tool_call_state[idx]["arguments"] += args_delta
                    events.append(ToolCallDeltaEvent(
                        tool_call_id=tool_call_state[idx]["id"],
                        arguments_delta=args_delta,
                    ))

        finish = choice.get("finish_reason")
        if finish:
            for tc in tool_call_state.values():
                try:
                    parsed = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    parsed = {}
                events.append(ToolCallEndEvent(
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    arguments=parsed,
                ))
            usage_data = chunk.get("usage") or {}
            pt = usage_data.get("prompt_tokens", 0)
            ct = usage_data.get("completion_tokens", 0)
            tt = usage_data.get("total_tokens", 0)
            if pt or ct or tt:
                events.append(StreamUsageEvent(
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=tt,
                ))
            events.append(StreamDoneEvent(
                finish_reason=parse_finish_reason(finish),
            ))

        return events, tool_call_state

    def parse_stream_chunk_done(
        self,
        chunk: dict[str, Any],
    ) -> bool:
        return False  # Chat Completions uses "[DONE]" sentinel, not a JSON field
