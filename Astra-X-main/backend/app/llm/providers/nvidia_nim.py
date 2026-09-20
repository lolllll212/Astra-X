"""NVIDIA NIM API provider adapter.

Communicates with NVIDIA NIM API for high-performance model inference.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from app.domain.enums import OpenAIProtocol
from app.domain.stream import (
    StreamErrorEvent,
    StreamEvent,
    StreamMetadataEvent,
)
from app.llm.base import LLMProvider
from app.llm.exceptions import (
    ProviderConnectionError,
    ProviderTimeoutError,
)
from app.llm.models import CompletionRequest, CompletionResponse
from app.llm.providers.adapters.chat_completions import (
    ChatCompletionsAdapter,
    domain_to_openai_messages,
    parse_finish_reason,
)
from app.llm.providers.transport import Transport

__all__ = [
    "NvidiaNimProvider",
]


class NvidiaNimProvider(LLMProvider):
    """Provider adapter for NVIDIA NIM API (OpenAI-compatible)."""

    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-3-ultra",
        timeout_seconds: float = 120.0,
        provider_id: str = "nvidia_nim",
        api_key: str | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model

        import os
        api_key = api_key or os.getenv("NVIDIA_API_KEY")

        self._transport = Transport(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            extra_headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

        self._adapter = ChatCompletionsAdapter()

    @property
    def protocol(self) -> OpenAIProtocol:
        return OpenAIProtocol.CHAT_COMPLETIONS

    async def generate(
        self,
        request,
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
        request,
    ) -> AsyncIterator:
        model = request.model or self.model_id
        payload = self._adapter.build_request(request, model)

        conversation_id = request.messages[0].conversation_id if request.messages else ""
        message_id = str(uuid4())

        yield StreamMetadataEvent(
            conversation_id=conversation_id,
            message_id=str(uuid4()),
            model=model,
            provider=self.provider_id,
        )

        tool_call_state: dict[int, dict[str, str]] = {}

        try:
            async for chunk in self._transport.stream(
                "POST", "/chat/completions", payload,
            ):
                events, tool_call_state = self._adapter.parse_stream_chunk(
                    chunk, tool_call_state,
                )
                for event in events:
                    yield event

                if self._adapter.parse_stream_chunk_done(chunk):
                    return

        except Exception as exc:
            yield StreamErrorEvent(
                error_code="provider_error",
                message=f"NVIDIA NIM returned error: {exc}",
            )

    async def check_health(self) -> bool:
        try:
            await self._transport.request("GET", "/models")
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            data = await self._transport.request("GET", "/models")
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return ["nvidia/nemotron-3-ultra", "meta/llama-3.1-405b-instruct", "meta/llama-3.1-70b-instruct"]

    async def close(self) -> None:
        await self._transport.close()