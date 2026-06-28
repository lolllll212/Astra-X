"""Generic OpenAI-compatible provider adapter.

Connects to any OpenAI-compatible API endpoint (e.g. OpenAI itself,
Azure OpenAI, Together AI, Groq, etc.) given a base URL and API key.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx

from app.domain.stream import (
    StreamDoneEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamMetadataEvent,
    StreamUsageEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from app.llm.base import LLMProvider
from app.llm.exceptions import (
    ProviderConnectionError,
    ProviderTimeoutError,
)
from app.llm.models import CompletionRequest, CompletionResponse, FinishReason


def _domain_to_openai_messages(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Convert domain messages to OpenAI-compatible message format."""
    from app.domain.message import (
        ImageBlock,
        TextBlock,
        ToolCallBlock,
        ToolResultBlock,
    )

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


def _parse_finish_reason(reason: str | None) -> str:
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(reason or "", FinishReason.STOP)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider adapter for any OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-4o",
        timeout_seconds: float = 60.0,
        provider_id: str = "openai_compatible",
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
        )

    async def generate(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        from app.llm.streaming import StreamCollector

        conversation_id = request.messages[0].conversation_id if request.messages else ""
        collector = StreamCollector(conversation_id=conversation_id)

        async for event in self.generate_stream(request):
            collector.feed(event)

        return CompletionResponse(
            message=collector.build_message(),
            usage=collector.usage,
            finish_reason=collector.finish_reason,
            model=request.model or self.model_id,
        )

    async def generate_stream(
        self,
        request: CompletionRequest,
    ) -> AsyncIterator[StreamEvent]:
        messages = _domain_to_openai_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model or self.model_id,
            "messages": messages,
            "stream": True,
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
            "max_tokens": request.params.max_tokens,
            "stop": request.params.stop or None,
        }

        conversation_id = request.messages[0].conversation_id if request.messages else ""
        message_id = str(uuid4())

        yield StreamMetadataEvent(
            conversation_id=conversation_id,
            message_id=message_id,
            model=request.model or self.model_id,
            provider=self.provider_id,
        )

        tool_calls_in_progress: dict[int, dict[str, str]] = {}

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_data = line.removeprefix("data: ").strip()
                    if chunk_data == "[DONE]":
                        break
                    if not chunk_data:
                        continue

                    chunk = json.loads(chunk_data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # Tool call deltas
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        func = tc_delta.get("function", {})
                        tcid = tc_delta.get("id", "")

                        if idx not in tool_calls_in_progress:
                            tool_calls_in_progress[idx] = {
                                "id": tcid,
                                "name": func.get("name", ""),
                                "arguments": "",
                            }
                            yield ToolCallStartEvent(
                                tool_call_id=tcid,
                                tool_name=func.get("name", "unknown"),
                            )
                        else:
                            args_delta = func.get("arguments", "")
                            if args_delta:
                                tool_calls_in_progress[idx]["arguments"] += args_delta
                                yield ToolCallDeltaEvent(
                                    tool_call_id=tool_calls_in_progress[idx]["id"],
                                    arguments_delta=args_delta,
                                )

                    # Text deltas
                    content_delta = delta.get("content", "")
                    if content_delta:
                        yield TextDeltaEvent(delta=content_delta)

                    finish = choice.get("finish_reason")
                    if finish:
                        for tc in tool_calls_in_progress.values():
                            try:
                                parsed = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except json.JSONDecodeError:
                                parsed = {}
                            yield ToolCallEndEvent(
                                tool_call_id=tc["id"],
                                tool_name=tc["name"],
                                arguments=parsed,
                            )
                        usage_data = chunk.get("usage") or {}
                        pt = usage_data.get("prompt_tokens", 0)
                        ct = usage_data.get("completion_tokens", 0)
                        tt = usage_data.get("total_tokens", 0)
                        if pt or ct or tt:
                            yield StreamUsageEvent(
                                prompt_tokens=pt,
                                completion_tokens=ct,
                                total_tokens=tt,
                            )
                        yield StreamDoneEvent(
                            finish_reason=_parse_finish_reason(finish),
                        )
                        return

        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Provider stream timed out after {self._timeout}s",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Could not connect to {self._base_url}",
            ) from exc
        except httpx.HTTPStatusError as exc:
            yield StreamErrorEvent(
                error_code="provider_error",
                message=f"Provider returned {exc.response.status_code}",
            )

    async def check_health(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return [self.model_id]

    async def close(self) -> None:
        await self._client.aclose()
