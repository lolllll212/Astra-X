"""Generic OpenAI-compatible provider adapter.

Connects to any OpenAI-compatible API endpoint (e.g. OpenAI itself,
Azure OpenAI, Together AI, Groq, etc.) given a base URL and API key.
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
from app.llm.models import CompletionRequest, CompletionResponse, FinishReason


def _domain_to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
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


def _openai_to_domain_message(
    choice: dict[str, Any],
    conversation_id: str,
    message_id: str | None = None,
) -> Message:
    """Convert an OpenAI-compatible response choice to a domain Message."""
    content: list[ContentBlock] = []
    content_raw = choice.get("message", {}).get("content") or ""

    if content_raw:
        content.append(TextBlock(text=content_raw))

    for tc in choice.get("message", {}).get("tool_calls") or []:
        func = tc.get("function", {})
        content.append(
            ToolCallBlock(
                tool_call_id=tc.get("id", str(uuid4())),
                tool_name=func.get("name", "unknown"),
                arguments=func.get("arguments", {}),
            ),
        )

    role = choice.get("message", {}).get("role", "assistant")
    return Message(
        id=message_id or str(uuid4()),
        conversation_id=conversation_id,
        role=MessageRole(role),
        content=content or [TextBlock(text="")],
        created_at=datetime.now(UTC),
    )


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
        messages = _domain_to_openai_messages(request.messages)
        payload: dict[str, Any] = {
            "model": request.model or self.model_id,
            "messages": messages,
            "stream": False,
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
            "max_tokens": request.params.max_tokens,
            "stop": request.params.stop or None,
            "presence_penalty": request.params.presence_penalty,
            "frequency_penalty": request.params.frequency_penalty,
        }

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Provider request timed out after {self._timeout}s",
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderConnectionError(
                f"Could not connect to {self._base_url}",
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise ProviderAuthenticationError("Provider authentication failed") from exc
            raise GenerationError(
                f"Provider returned {exc.response.status_code}: {exc.response.text}",
            ) from exc

        choice = data["choices"][0]
        domain_message = _openai_to_domain_message(
            choice,
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
            finish_reason=_parse_finish_reason(choice.get("finish_reason")),
            model=data.get("model", self.model_id),
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
            model_id=request.model or self.model_id,
            provider_id=self.provider_id,
        )

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
                        yield StreamDoneEvent(finish_reason=FinishReason.STOP)
                        return
                    if not chunk_data:
                        continue

                    import json as _json

                    chunk = _json.loads(chunk_data)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content_delta = delta.get("content", "")
                    if content_delta:
                        yield TextDeltaEvent(delta=content_delta)

                    finish = choices[0].get("finish_reason")
                    if finish:
                        yield StreamDoneEvent(
                            finish_reason=_parse_finish_reason(finish),
                        )

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
