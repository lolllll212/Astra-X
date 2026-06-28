"""Ollama provider adapter.

Communicates with a local Ollama server via its REST API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from app.domain.enums import MessageRole
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
    StreamErrorEvent,
    StreamEvent,
    StreamMetadataEvent,
    StreamUsageEvent,
    TextDeltaEvent,
)
from app.domain.usage import Usage
from app.llm.base import LLMProvider
from app.llm.exceptions import (
    GenerationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
)
from app.llm.models import (
    CompletionRequest,
    CompletionResponse,
    FinishReason,
)


def _domain_to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert domain messages to Ollama API message format."""
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


def _ollama_to_domain_message(
    data: dict[str, Any],
    conversation_id: str,
    message_id: str | None = None,
) -> Message:
    """Convert an Ollama response message to a domain Message."""
    content: list[ContentBlock] = []
    role_str = data.get("role", "assistant")
    content_raw = data.get("content", "")

    if content_raw:
        content.append(TextBlock(text=content_raw))

    for tc in data.get("tool_calls") or []:
        func = tc.get("function", {})
        content.append(
            ToolCallBlock(
                tool_call_id=str(uuid4()),
                tool_name=func.get("name", "unknown"),
                arguments=func.get("arguments", {}),
            ),
        )

    return Message(
        id=message_id or str(uuid4()),
        conversation_id=conversation_id,
        role=MessageRole(role_str),
        content=content or [TextBlock(text="")],
        created_at=datetime.now(UTC),
    )


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
        messages = _domain_to_ollama_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model or self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.params.temperature,
                "top_p": request.params.top_p,
                "top_k": request.params.top_k,
                "num_predict": request.params.max_tokens,
                "stop": request.params.stop or None,
            },
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self._timeout}s",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Could not connect to Ollama at {self._base_url}",
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise ProviderAuthenticationError("Ollama authentication failed") from exc
            raise GenerationError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text}",
            ) from exc

        domain_message = _ollama_to_domain_message(
            data.get("message", {}),
            conversation_id=request.messages[0].conversation_id if request.messages else "",
        )
        usage_data = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return CompletionResponse(
            message=domain_message,
            usage=usage,
            finish_reason=FinishReason.STOP if data.get("done") else FinishReason.LENGTH,
            model=data.get("model", self.model_id),
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
                    import json as _json

                    chunk = _json.loads(line)
                    msg_data = chunk.get("message", {})
                    delta = msg_data.get("content", "")

                    if delta:
                        yield TextDeltaEvent(delta=delta)

                    if chunk.get("done"):
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
