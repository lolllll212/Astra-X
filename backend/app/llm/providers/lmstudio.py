"""LM Studio provider adapter.

Communicates with a local LM Studio server via its OpenAI-compatible API.
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
    "LMStudioProvider",
    "_domain_to_openai_messages",
    "_parse_finish_reason",
]

_domain_to_openai_messages = domain_to_openai_messages
_parse_finish_reason = parse_finish_reason


class LMStudioProvider(LLMProvider):
    """Provider adapter for LM Studio (OpenAI-compatible local server).

    Uses the Chat Completions protocol adapter under the hood.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        timeout_seconds: float = 60.0,
        provider_id: str = "lm_studio",
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model

        self._transport = Transport(
            base_url=base_url,
            timeout=timeout_seconds,
        )

        self._adapter = ChatCompletionsAdapter()

    @property
    def protocol(self) -> OpenAIProtocol:
        return OpenAIProtocol.CHAT_COMPLETIONS

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
                f"LM Studio stream timed out after {self._transport._client.timeout.read}s",
            ) from exc
        except ProviderConnectionError as exc:
            raise ProviderConnectionError(
                f"Could not connect to LM Studio at {self._transport._base_url}",
            ) from exc
        except Exception as exc:
            yield StreamErrorEvent(
                error_code="provider_error",
                message=f"LM Studio returned error: {exc}",
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
