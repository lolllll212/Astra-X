"""Generic OpenAI-compatible provider adapter.

Connects to any OpenAI-compatible API endpoint (e.g. OpenAI itself,
Azure OpenAI, Together AI, Groq, etc.) given a base URL and API key.

Now protocol-aware: delegates request building and response parsing to
a :class:`ProtocolAdapter` (see ``adapters/``) so that the same provider
code works with both ``/chat/completions`` and ``/responses``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from app.domain.enums import OpenAIProtocol
from app.domain.stream import (
    StreamDoneEvent,
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
from app.llm.providers.adapters.base import ProtocolAdapter
from app.llm.providers.adapters.chat_completions import (
    ChatCompletionsAdapter,
    domain_to_openai_messages,
    parse_finish_reason,
)
from app.llm.providers.transport import Transport

__all__ = [
    "OpenAICompatibleProvider",
    "_domain_to_openai_messages",
    "_parse_finish_reason",
]

# Re-exports for backwards compatibility with existing tests
_domain_to_openai_messages = domain_to_openai_messages
_parse_finish_reason = parse_finish_reason


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider adapter for any OpenAI-compatible API.

    Uses a :class:`Transport` for HTTP communication and a
    :class:`ProtocolAdapter` for request/response formatting.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-4o",
        timeout_seconds: float = 60.0,
        provider_id: str = "openai_compatible",
        protocol: OpenAIProtocol = OpenAIProtocol.CHAT_COMPLETIONS,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model
        self._protocol = protocol

        self._transport = Transport(
            base_url=base_url,
            api_key=api_key,
            extra_headers=extra_headers,
            timeout=timeout_seconds,
        )

        self._adapter = self._select_adapter(protocol)

    @property
    def protocol(self) -> OpenAIProtocol:
        return self._protocol

    @staticmethod
    def _select_adapter(protocol: OpenAIProtocol) -> ProtocolAdapter:
        from app.llm.providers.adapters.responses_api import ResponsesApiAdapter

        mapping: dict[OpenAIProtocol, type[ProtocolAdapter]] = {
            OpenAIProtocol.CHAT_COMPLETIONS: ChatCompletionsAdapter,
            OpenAIProtocol.RESPONSES: ResponsesApiAdapter,
            OpenAIProtocol.COMPATIBLE: ChatCompletionsAdapter,
        }
        adapter_cls = mapping.get(protocol, ChatCompletionsAdapter)
        return adapter_cls()

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
        model = request.model or self.model_id
        payload = self._adapter.build_request(request, model)

        conversation_id = request.messages[0].conversation_id if request.messages else ""
        message_id = str(uuid4())

        yield StreamMetadataEvent(
            conversation_id=conversation_id,
            message_id=message_id,
            model=model,
            provider=self.provider_id,
        )

        tool_call_state: dict[int, dict[str, str]] = {}

        try:
            async for chunk in self._transport.stream(
                "POST", self._adapter.endpoint, payload,
            ):
                events, tool_call_state = self._adapter.parse_stream_chunk(
                    chunk, tool_call_state,
                )
                for event in events:
                    yield event

                if self._adapter.parse_stream_chunk_done(chunk):
                    return

        except ProviderTimeoutError as exc:
            raise ProviderTimeoutError(
                f"Provider stream timed out after {self._transport._client.timeout.read}s",
            ) from exc
        except ProviderConnectionError as exc:
            raise ProviderConnectionError(
                f"Could not connect to {self._transport._base_url}",
            ) from exc
        except Exception as exc:
            yield StreamErrorEvent(
                error_code="provider_error",
                message=str(exc),
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
            return [self.model_id]

    async def close(self) -> None:
        await self._transport.close()
