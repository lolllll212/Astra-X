"""Ollama provider adapter.

Communicates with a local Ollama server via its REST API.
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
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from app.llm.base import LLMProvider
from app.llm.exceptions import (
    ProviderConnectionError,
    ProviderTimeoutError,
)
from app.llm.models import (
    CompletionRequest,
    CompletionResponse,
    FinishReason,
)


def _domain_to_ollama_messages(
    messages: list[Any],
) -> list[dict[str, Any]]:
    """Convert domain messages to Ollama API message format."""
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
        for block in msg.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, ImageBlock):
                texts.append(block.data_uri)
            elif isinstance(block, ToolCallBlock):
                entry.setdefault("tool_calls", []).append({
                    "function": {
                        "name": block.tool_name,
                        "arguments": block.arguments,
                    },
                })
            elif isinstance(block, ToolResultBlock):
                entry = {
                    "role": "tool",
                    "content": block.output,
                }
                if block.tool_call_id:
                    result.append(entry)
                    continue
        if texts:
            entry["content"] = "\n".join(texts)
        result.append(entry)
    return result


class OllamaProvider(LLMProvider):
    """Provider adapter for Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout_seconds: float = 60.0,
        provider_id: str = "ollama",
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
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
        messages = _domain_to_ollama_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model or self.model_id,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.params.temperature,
                "top_p": request.params.top_p,
                "num_predict": request.params.max_tokens,
                "stop": request.params.stop or None,
            },
        }

        conversation_id = request.messages[0].conversation_id if request.messages else ""
        message_id = str(uuid4())

        yield StreamMetadataEvent(
            conversation_id=conversation_id,
            message_id=message_id,
            model=request.model or self.model_id,
            provider=self.provider_id,
        )

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue

                    chunk = json.loads(line)
                    msg_data = chunk.get("message", {})
                    delta = msg_data.get("content", "")

                    if delta:
                        yield TextDeltaEvent(delta=delta)

                    if chunk.get("done"):
                        for tc in msg_data.get("tool_calls") or []:
                            func = tc.get("function", {})
                            tcid = str(uuid4())
                            yield ToolCallStartEvent(
                                tool_call_id=tcid,
                                tool_name=func.get("name", "unknown"),
                            )
                            yield ToolCallEndEvent(
                                tool_call_id=tcid,
                                tool_name=func.get("name", "unknown"),
                                arguments=func.get("arguments", {}),
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
                        yield StreamDoneEvent(finish_reason=FinishReason.STOP)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama stream timed out after {self._timeout}s",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Could not connect to Ollama at {self._base_url}",
            ) from exc
        except httpx.HTTPStatusError as exc:
            yield StreamErrorEvent(
                error_code="provider_error",
                message=f"Ollama returned {exc.response.status_code}",
            )

    async def check_health(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return [self.model_id]

    async def close(self) -> None:
        await self._client.aclose()
