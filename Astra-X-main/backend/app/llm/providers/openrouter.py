"""OpenRouter API provider adapter (free models only).

Communicates with OpenRouter API for accessing free models.
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
    "OpenRouterProvider",
]

# Curated list of free models on OpenRouter (as of 2024)
FREE_MODELS = [
    "meta-llama/llama-3.1-405b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "mistralai/mistral-nemo:free",
    "google/gemma-2-9b-it:free",
    "google/gemma-2-27b-it:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "microsoft/phi-3.5-mini-instruct:free",
    "microsoft/phi-3.5-vision-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "cognitivecomputations/dolphin-3.0-r1-mistral-24b:free",
    "cognitivecomputations/dolphin-3.0-mistral-24b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nousresearch/hermes-3-llama-3.1-70b:free",
    "openchat/openchat-7b:free",
    "undi95/toppy-m-7b:free",
    "gryphe/mythomax-l2-13b:free",
]


class OpenRouterProvider(LLMProvider):
    """Provider adapter for OpenRouter API (free models only)."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        timeout_seconds: float = 120.0,
        provider_id: str = "openrouter",
        api_key: str | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model
        self._free_models = FREE_MODELS.copy()

        import os
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        headers["HTTP-Referer"] = "https://astra-x.ai"
        headers["X-Title"] = "Astra X"

        self._transport = Transport(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            extra_headers=headers,
        )

        from app.llm.providers.adapters.chat_completions import ChatCompletionsAdapter
        self._adapter = ChatCompletionsAdapter()

    @property
    def protocol(self) -> OpenAIProtocol:
        return OpenAIProtocol.CHAT_COMPLETIONS

    async def generate(
        self,
        request,
    ):
        from app.llm.streaming import StreamCollector
        from app.llm.models import CompletionResponse

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
    ):
        model = request.model or self.model_id

        # Validate model is free
        if model not in self._free_models:
            # Fall back to default free model
            model = self._free_models[0]

        from app.llm.providers.adapters.chat_completions import ChatCompletionsAdapter
        adapter = ChatCompletionsAdapter()
        payload = adapter.build_request(request, model)

        conversation_id = request.messages[0].conversation_id if request.messages else ""
        message_id = str(uuid4())

        from app.domain.stream import StreamMetadataEvent
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
                from app.llm.providers.adapters.chat_completions import ChatCompletionsAdapter
                adapter = ChatCompletionsAdapter()
                events, tool_call_state = adapter.parse_stream_chunk(
                    chunk, tool_call_state,
                )
                for event in events:
                    yield event

                if adapter.parse_stream_chunk_done(chunk):
                    return

        except Exception as exc:
            from app.domain.stream import StreamErrorEvent
            yield StreamErrorEvent(
                error_code="provider_error",
                message=f"OpenRouter returned error: {exc}",
            )

    def get_free_models(self) -> list[str]:
        """Return list of available free models."""
        return self._free_models.copy()

    async def check_health(self) -> bool:
        try:
            await self._transport.request("GET", "/models")
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            data = await self._transport.request("GET", "/models")
            # Filter to only free models
            all_models = [m["id"] for m in data.get("data", [])]
            free_models = [m for m in all_models if m in self._free_models or ":free" in m]
            return free_models if free_models else self._free_models
        except Exception:
            return self._free_models

    async def close(self) -> None:
        await self._transport.close()