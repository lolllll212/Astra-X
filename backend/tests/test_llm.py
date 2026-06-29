"""Comprehensive unit tests for the LLM layer.

Covers models, exceptions, base, registry, factory, streaming, all three
provider adapters, and the router.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.settings import Settings
from app.domain.enums import MessageRole, ModelCapability, ProviderType
from app.domain.message import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from app.domain.provider import ProviderSpec
from app.domain.stream import (
    StreamDoneEvent,
    StreamErrorEvent,
    StreamMetadataEvent,
    StreamStartEvent,
    StreamUsageEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from app.domain.usage import Usage
from app.llm.base import LLMProvider
from app.llm.exceptions import (
    GenerationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
    RouterNoProviderError,
)
from app.llm.factory import create_provider_adapter, provider_type_to_default_base_url
from app.llm.models import (
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    GenerationParams,
)
from app.llm.providers.lmstudio import LMStudioProvider, _domain_to_openai_messages, _parse_finish_reason
from app.llm.providers.ollama import OllamaProvider, _domain_to_ollama_messages
from app.llm.providers.openai_compatible import OpenAICompatibleProvider
from app.llm.registry import ProviderRegistry
from app.llm.router import LLMRouter
from app.llm.streaming import StreamCollector, collect_stream


# =========================================================================
# Helpers
# =========================================================================


def _make_message(text: str = "hello", role: MessageRole = MessageRole.USER,
                  conversation_id: str = "conv1") -> Message:
    return Message(
        id="msg1",
        conversation_id=conversation_id,
        role=role,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def _make_request(text: str = "hello", model: str = "test-model",
                  provider: str | None = None,
                  conversation_id: str = "conv1") -> CompletionRequest:
    return CompletionRequest(
        messages=[_make_message(text=text, conversation_id=conversation_id)],
        model=model,
        provider=provider,
    )


def _settings() -> MagicMock:
    s = MagicMock(spec=Settings)
    s.ollama_base_url = "http://localhost:11434"
    s.lm_studio_base_url = "http://localhost:1234/v1"
    s.openai_compatible_base_url = None
    s.llm_request_timeout_seconds = 60.0
    s.default_llm_model = "llama3.1"
    return s


def _make_spec(provider_type: ProviderType = ProviderType.OLLAMA,
               provider_id: str = "ollama",
               models: list[str] | None = None,
               base_url: str | None = None) -> ProviderSpec:
    return ProviderSpec(
        id=provider_id,
        provider_type=provider_type,
        display_name=provider_id,
        models=models or ["llama3.1"],
        base_url=base_url,
        is_enabled=True,
    )


def _mock_stream_response(
    lines: list[str],
) -> MagicMock:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = MagicMock(return_value=_async_iter(lines))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock()
    mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)

    return mock_client


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


# =========================================================================
# FinishReason
# =========================================================================


class TestFinishReason:
    def test_constants(self) -> None:
        assert FinishReason.STOP == "stop"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.TOOL_CALLS == "tool_calls"
        assert FinishReason.CONTENT_FILTER == "content_filter"
        assert FinishReason.ERROR == "error"
        assert FinishReason.NULL == "null"


# =========================================================================
# GenerationParams
# =========================================================================


class TestGenerationParams:
    def test_defaults(self) -> None:
        p = GenerationParams()
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.max_tokens is None
        assert p.stop == []

    def test_custom_values(self) -> None:
        p = GenerationParams(temperature=0.5, max_tokens=100, stop=["\n"])
        assert p.temperature == 0.5
        assert p.max_tokens == 100
        assert p.stop == ["\n"]


# =========================================================================
# CompletionRequest / CompletionResponse
# =========================================================================


class TestCompletionRequest:
    def test_defaults(self) -> None:
        msg = _make_message()
        req = CompletionRequest(messages=[msg], model="m")
        assert req.provider is None
        assert req.stream is False
        assert req.params.temperature == 0.7


class TestCompletionResponse:
    def test_defaults(self) -> None:
        msg = _make_message(role=MessageRole.ASSISTANT)
        resp = CompletionResponse(message=msg)
        assert resp.finish_reason == FinishReason.STOP
        assert resp.model == ""
        assert resp.usage is None


# =========================================================================
# LLMProvider base
# =========================================================================


class TestLLMProviderBase:
    @pytest.mark.asyncio
    async def test_generate_stream_not_implemented(self) -> None:
        class MinimalProvider(LLMProvider):
            provider_id = "test"
            model_id = "m"

            async def generate(self, request: CompletionRequest) -> CompletionResponse:
                raise NotImplementedError

        p = MinimalProvider()
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            async for _ in p.generate_stream(_make_request()):
                pass

    @pytest.mark.asyncio
    async def test_check_health_default(self) -> None:
        class HealthyProvider(LLMProvider):
            provider_id = "test"
            model_id = "m"

            async def generate(self, request: CompletionRequest) -> CompletionResponse:
                raise NotImplementedError

        p = HealthyProvider()
        assert await p.check_health() is True

    @pytest.mark.asyncio
    async def test_list_models_default(self) -> None:
        class FixedProvider(LLMProvider):
            provider_id = "test"
            model_id = "my-model"

            async def generate(self, request: CompletionRequest) -> CompletionResponse:
                raise NotImplementedError

        p = FixedProvider()
        assert await p.list_models() == ["my-model"]

    def test_supports_capability(self) -> None:
        class CapableProvider(LLMProvider):
            provider_id = "test"
            model_id = "m"

            async def generate(self, request: CompletionRequest) -> CompletionResponse:
                raise NotImplementedError

        p = CapableProvider()
        assert p.supports_capability(ModelCapability.CHAT)
        assert p.supports_capability(ModelCapability.STREAMING)
        assert not p.supports_capability(ModelCapability.EMBEDDING)


# =========================================================================
# ProviderRegistry
# =========================================================================


class TestProviderRegistry:
    @pytest.fixture
    def registry(self) -> ProviderRegistry:
        return ProviderRegistry()

    def test_register_and_get(self, registry: ProviderRegistry) -> None:
        p = MagicMock(spec=LLMProvider)
        p.provider_id = "test_provider"
        registry.register(p)
        assert registry.get("test_provider") is p

    def test_get_unknown_raises(self, registry: ProviderRegistry) -> None:
        with pytest.raises(RouterNoProviderError, match="No provider registered for 'unknown'"):
            registry.get("unknown")

    def test_list(self, registry: ProviderRegistry) -> None:
        p1, p2 = MagicMock(spec=LLMProvider), MagicMock(spec=LLMProvider)
        p1.provider_id, p2.provider_id = "p1", "p2"
        registry.register(p1)
        registry.register(p2)
        assert len(registry.list()) == 2

    def test_remove(self, registry: ProviderRegistry) -> None:
        p = MagicMock(spec=LLMProvider)
        p.provider_id = "p"
        registry.register(p)
        registry.remove("p")
        assert registry.is_registered("p") is False

    def test_is_registered(self, registry: ProviderRegistry) -> None:
        p = MagicMock(spec=LLMProvider)
        p.provider_id = "p"
        assert registry.is_registered("p") is False
        registry.register(p)
        assert registry.is_registered("p") is True

    def test_register_overwrites(self, registry: ProviderRegistry) -> None:
        p1, p2 = MagicMock(spec=LLMProvider), MagicMock(spec=LLMProvider)
        p1.provider_id = p2.provider_id = "same"
        registry.register(p1)
        registry.register(p2)
        assert registry.get("same") is p2


# =========================================================================
# _domain_to_ollama_messages
# =========================================================================


class TestDomainToOllamaMessages:
    def test_text_message(self) -> None:
        msg = _make_message(text="hello")
        result = _domain_to_ollama_messages([msg])
        assert result == [{"role": "user", "content": "hello"}]

    def test_tool_call(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role=MessageRole.ASSISTANT,
            content=[ToolCallBlock(tool_call_id="tc1", tool_name="calc", arguments={"x": 1})],
            created_at=datetime.now(UTC),
        )
        result = _domain_to_ollama_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "calc"

    def test_tool_result(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role=MessageRole.TOOL,
            content=[ToolResultBlock(tool_call_id="tc1", tool_name="calc", output="42")],
            created_at=datetime.now(UTC),
        )
        result = _domain_to_ollama_messages([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "42"

    def test_image_block(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role=MessageRole.USER,
            content=[ImageBlock(data_uri="data:image/png;base64,...")],
            created_at=datetime.now(UTC),
        )
        result = _domain_to_ollama_messages([msg])
        assert len(result) == 1
        assert result[0]["content"] == "data:image/png;base64,..."


# =========================================================================
# _domain_to_openai_messages
# =========================================================================


class TestDomainToOpenAIMessages:
    def test_text_message(self) -> None:
        msg = _make_message(text="hello")
        result = _domain_to_openai_messages([msg])
        assert result == [{"role": "user", "content": "hello"}]

    def test_multimodal(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role=MessageRole.USER,
            content=[
                TextBlock(text="what is this?"),
                ImageBlock(data_uri="data:image/png;base64,..."),
            ],
            created_at=datetime.now(UTC),
        )
        result = _domain_to_openai_messages([msg])
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"

    def test_tool_call(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role=MessageRole.ASSISTANT,
            content=[ToolCallBlock(tool_call_id="tc1", tool_name="calc", arguments={"x": 1})],
            created_at=datetime.now(UTC),
        )
        result = _domain_to_openai_messages([msg])
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "calc"

    def test_tool_result(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role=MessageRole.TOOL,
            content=[ToolResultBlock(tool_call_id="tc1", tool_name="calc", output="42")],
            created_at=datetime.now(UTC),
        )
        result = _domain_to_openai_messages([msg])
        assert result[0]["role"] == "tool"


# =========================================================================
# _parse_finish_reason
# =========================================================================


class TestParseFinishReason:
    def test_stop(self) -> None:
        assert _parse_finish_reason("stop") == FinishReason.STOP

    def test_length(self) -> None:
        assert _parse_finish_reason("length") == FinishReason.LENGTH

    def test_tool_calls(self) -> None:
        assert _parse_finish_reason("tool_calls") == FinishReason.TOOL_CALLS

    def test_unknown_defaults_to_stop(self) -> None:
        assert _parse_finish_reason(None) == FinishReason.STOP
        assert _parse_finish_reason("") == FinishReason.STOP
        assert _parse_finish_reason("weird_reason") == FinishReason.STOP


# =========================================================================
# StreamCollector
# =========================================================================


class TestStreamCollector:
    def test_empty_collector(self) -> None:
        c = StreamCollector(conversation_id="c1")
        assert c.finish_reason == FinishReason.NULL
        assert c.usage is None

    def test_text_deltas(self) -> None:
        c = StreamCollector(conversation_id="c1")
        c.feed(TextDeltaEvent(delta="Hello "))
        c.feed(TextDeltaEvent(delta="World"))
        msg = c.build_message()
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "Hello World"

    def test_tool_calls(self) -> None:
        c = StreamCollector(conversation_id="c1")
        c.feed(ToolCallStartEvent(tool_call_id="tc1", tool_name="calc"))
        c.feed(ToolCallDeltaEvent(tool_call_id="tc1", arguments_delta='{"x":'))
        c.feed(ToolCallDeltaEvent(tool_call_id="tc1", arguments_delta=' 1}'))
        c.feed(ToolCallEndEvent(tool_call_id="tc1", tool_name="calc", arguments={"x": 1}))
        msg = c.build_message()
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ToolCallBlock)
        assert msg.content[0].tool_name == "calc"
        assert msg.content[0].arguments == {"x": 1}

    def test_usage(self) -> None:
        c = StreamCollector(conversation_id="c1")
        c.feed(StreamUsageEvent(prompt_tokens=10, completion_tokens=20, total_tokens=30))
        assert c.usage is not None
        assert c.usage.prompt_tokens == 10
        assert c.usage.total_tokens == 30

    def test_finish_reason(self) -> None:
        c = StreamCollector(conversation_id="c1")
        assert c.finish_reason == FinishReason.NULL
        c.feed(StreamDoneEvent(finish_reason=FinishReason.STOP))
        assert c.finish_reason == FinishReason.STOP

    def test_unknown_events_are_ignored(self) -> None:
        c = StreamCollector(conversation_id="c1")
        c.feed(StreamStartEvent())
        assert c.finish_reason == FinishReason.NULL


class TestCollectStream:
    @pytest.mark.asyncio
    async def test_collects_all_events(self) -> None:
        async def event_stream() -> AsyncIterator[Any]:
            yield TextDeltaEvent(delta="hello")
            yield StreamUsageEvent(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            yield StreamDoneEvent(finish_reason=FinishReason.STOP)

        collector = await collect_stream(event_stream(), conversation_id="c1")
        assert len(collector.build_message().content) == 1
        assert collector.usage is not None
        assert collector.finish_reason == FinishReason.STOP


# =========================================================================
# OllamaProvider
# =========================================================================


class TestOllamaProvider:
    @pytest.fixture
    def provider(self) -> OllamaProvider:
        p = OllamaProvider(base_url="http://localhost:11434", model="llama3.1")
        yield p
        # Don't actually close in tests that don't open a real client
        # (the client is patched away in most tests)

    @pytest.mark.asyncio
    async def test_properties(self, provider: OllamaProvider) -> None:
        assert provider.provider_id == "ollama"
        assert provider.model_id == "llama3.1"

    @pytest.mark.asyncio
    async def test_generate_via_stream(self, provider: OllamaProvider) -> None:
        """generate() collects from generate_stream()."""
        ollama_lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " World"}, "done": False}),
            json.dumps({
                "message": {"content": ""},
                "done": True,
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }),
        ]
        mock_client = _mock_stream_response(ollama_lines)
        with patch.object(provider, "_client", mock_client):
            resp = await provider.generate(_make_request())
        assert resp.message is not None
        assert "Hello World" in str(resp.message.content[0].text)
        assert resp.usage is not None
        assert resp.usage.total_tokens == 8

    @pytest.mark.asyncio
    async def test_generate_stream_yields_events(self, provider: OllamaProvider) -> None:
        ollama_lines = [
            json.dumps({"message": {"content": "Hi"}, "done": False}),
            json.dumps({
                "message": {"content": "", "tool_calls": [{"function": {"name": "calc", "arguments": {"x": 1}}}]},
                "done": True,
            }),
        ]
        mock_client = _mock_stream_response(ollama_lines)
        with patch.object(provider, "_client", mock_client):
            events: list[Any] = []
            async for event in provider.generate_stream(_make_request()):
                events.append(event)

        assert len(events) >= 3
        assert any(isinstance(e, StreamMetadataEvent) for e in events)
        assert any(isinstance(e, TextDeltaEvent) and e.delta == "Hi" for e in events)
        assert any(isinstance(e, ToolCallStartEvent) for e in events)
        assert any(isinstance(e, ToolCallEndEvent) for e in events)

    @pytest.mark.asyncio
    async def test_health_ok(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock()
        mock_client.get.return_value.status_code = 200

        with patch.object(provider, "_client", mock_client):
            assert await provider.check_health() is True

    @pytest.mark.asyncio
    async def test_health_fail(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("Connection error"))

        with patch.object(provider, "_client", mock_client):
            assert await provider.check_health() is False

    @pytest.mark.asyncio
    async def test_list_models(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock()
        mock_client.get.return_value.status_code = 200
        mock_client.get.return_value.json = MagicMock(return_value={
            "models": [{"name": "llama3.1"}, {"name": "mistral"}],
        })

        with patch.object(provider, "_client", mock_client):
            models = await provider.list_models()
        assert "llama3.1" in models
        assert "mistral" in models

    @pytest.mark.asyncio
    async def test_list_models_fallback_on_error(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("fail"))

        with patch.object(provider, "_client", mock_client):
            models = await provider.list_models()
        assert models == ["llama3.1"]

    @pytest.mark.asyncio
    async def test_timeout_error(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=__import__("httpx").TimeoutException("timed out"),
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider, "_client", mock_client):
            with pytest.raises(ProviderTimeoutError, match="Ollama stream timed out"):
                async for _ in provider.generate_stream(_make_request()):
                    pass

    @pytest.mark.asyncio
    async def test_connection_error(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=__import__("httpx").ConnectError("connection refused"),
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider, "_client", mock_client):
            with pytest.raises(ProviderConnectionError, match="Could not connect to Ollama"):
                async for _ in provider.generate_stream(_make_request()):
                    pass

    @pytest.mark.asyncio
    async def test_http_status_error_yields_error_event(self, provider: OllamaProvider) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=__import__("httpx").HTTPStatusError(
                "Too Many", request=MagicMock(), response=mock_response,
            ),
        )
        mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider, "_client", mock_client):
            events: list[Any] = []
            async for event in provider.generate_stream(_make_request()):
                events.append(event)
        assert any(isinstance(e, StreamErrorEvent) for e in events)

    @pytest.mark.asyncio
    async def test_close(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        with patch.object(provider, "_client", mock_client):
            await provider.close()
        mock_client.aclose.assert_awaited_once()


# =========================================================================
# LMStudioProvider
# =========================================================================


class TestLMStudioProvider:
    @pytest.fixture
    def provider(self) -> LMStudioProvider:
        return LMStudioProvider(base_url="http://localhost:1234/v1", model="local-model")

    @pytest.mark.asyncio
    async def test_properties(self, provider: LMStudioProvider) -> None:
        assert provider.provider_id == "lm_studio"
        assert provider.model_id == "local-model"

    @pytest.mark.asyncio
    async def test_generate_via_stream(self, provider: LMStudioProvider) -> None:
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": " World"}, "finish_reason": None}]}),
            "data: " + json.dumps({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_response(lines)
        with patch.object(provider, "_client", mock_client):
            resp = await provider.generate(_make_request())
        assert "Hello World" in str(resp.message.content[0].text)
        assert resp.usage is not None
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_stream_tool_calls(self, provider: LMStudioProvider) -> None:
        lines = [
            "data: " + json.dumps({
                "choices": [{
                    "delta": {"tool_calls": [{"index": 0, "id": "tc1", "function": {"name": "calc", "arguments": ""}}]},
                    "finish_reason": None,
                }],
            }),
            "data: " + json.dumps({
                "choices": [{
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x": 1}'}}]},
                    "finish_reason": "tool_calls",
                }],
            }),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_response(lines)
        with patch.object(provider, "_client", mock_client):
            events: list[Any] = []
            async for event in provider.generate_stream(_make_request()):
                events.append(event)
        assert any(isinstance(e, ToolCallStartEvent) for e in events)
        assert any(isinstance(e, ToolCallDeltaEvent) for e in events)
        assert any(isinstance(e, ToolCallEndEvent) for e in events)
        assert any(isinstance(e, StreamDoneEvent) for e in events)

    @pytest.mark.asyncio
    async def test_stream_skips_non_data_lines(self, provider: LMStudioProvider) -> None:
        lines = [
            ": keep-alive comment",
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_response(lines)
        with patch.object(provider, "_client", mock_client):
            events: list[Any] = []
            async for event in provider.generate_stream(_make_request()):
                events.append(event)
        assert any(isinstance(e, TextDeltaEvent) for e in events)

    @pytest.mark.asyncio
    async def test_health_ok(self, provider: LMStudioProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock()
        mock_client.get.return_value.status_code = 200

        with patch.object(provider, "_client", mock_client):
            assert await provider.check_health() is True

    @pytest.mark.asyncio
    async def test_list_models(self, provider: LMStudioProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock()
        mock_client.get.return_value.status_code = 200
        mock_client.get.return_value.json = MagicMock(return_value={
            "data": [{"id": "model1"}, {"id": "model2"}],
        })

        with patch.object(provider, "_client", mock_client):
            models = await provider.list_models()
        assert "model1" in models
        assert "model2" in models

    @pytest.mark.asyncio
    async def test_timeout_error(self, provider: LMStudioProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=__import__("httpx").TimeoutException("timed out"),
        )

        with patch.object(provider, "_client", mock_client):
            with pytest.raises(ProviderTimeoutError, match="LM Studio stream timed out"):
                async for _ in provider.generate_stream(_make_request()):
                    pass

    @pytest.mark.asyncio
    async def test_http_status_error(self, provider: LMStudioProvider) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=__import__("httpx").HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response,
            ),
        )

        with patch.object(provider, "_client", mock_client):
            events: list[Any] = []
            async for event in provider.generate_stream(_make_request()):
                events.append(event)
        assert any(isinstance(e, StreamErrorEvent) for e in events)

    @pytest.mark.asyncio
    async def test_close(self, provider: LMStudioProvider) -> None:
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        with patch.object(provider, "_client", mock_client):
            await provider.close()
        mock_client.aclose.assert_awaited_once()


# =========================================================================
# OpenAICompatibleProvider
# =========================================================================


class TestOpenAICompatibleProvider:
    @pytest.fixture
    def provider(self) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

    @pytest.mark.asyncio
    async def test_properties(self, provider: OpenAICompatibleProvider) -> None:
        assert provider.provider_id == "openai_compatible"
        assert provider.model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_via_stream(self, provider: OpenAICompatibleProvider) -> None:
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}),
            "data: " + json.dumps({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_response(lines)
        with patch.object(provider, "_client", mock_client):
            resp = await provider.generate(_make_request())
        assert resp.message is not None
        assert resp.usage is not None
        assert resp.usage.total_tokens == 5

    @pytest.mark.asyncio
    async def test_generate_stream_with_api_key(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-secret",
            model="gpt-4o",
        )
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_response(lines)
        with patch.object(provider, "_client", mock_client):
            events: list[Any] = []
            async for event in provider.generate_stream(_make_request()):
                events.append(event)
        assert any(isinstance(e, TextDeltaEvent) for e in events)

    @pytest.mark.asyncio
    async def test_without_api_key(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key=None,
            model="gpt-4o",
        )
        assert provider is not None

    @pytest.mark.asyncio
    async def test_health_ok(self, provider: OpenAICompatibleProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock()
        mock_client.get.return_value.status_code = 200

        with patch.object(provider, "_client", mock_client):
            assert await provider.check_health() is True

    @pytest.mark.asyncio
    async def test_list_models(self, provider: OpenAICompatibleProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock()
        mock_client.get.return_value.status_code = 200
        mock_client.get.return_value.json = MagicMock(return_value={
            "data": [{"id": "gpt-4"}, {"id": "gpt-3.5"}],
        })

        with patch.object(provider, "_client", mock_client):
            models = await provider.list_models()
        assert "gpt-4" in models

    @pytest.mark.asyncio
    async def test_timeout_error(self, provider: OpenAICompatibleProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=__import__("httpx").TimeoutException("timed out"),
        )

        with patch.object(provider, "_client", mock_client):
            with pytest.raises(ProviderTimeoutError, match="Provider stream timed out"):
                async for _ in provider.generate_stream(_make_request()):
                    pass

    @pytest.mark.asyncio
    async def test_connection_error(self, provider: OpenAICompatibleProvider) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=__import__("httpx").ConnectError("connection refused"),
        )

        with patch.object(provider, "_client", mock_client):
            with pytest.raises(ProviderConnectionError, match="Could not connect to"):
                async for _ in provider.generate_stream(_make_request()):
                    pass

    @pytest.mark.asyncio
    async def test_close(self, provider: OpenAICompatibleProvider) -> None:
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        with patch.object(provider, "_client", mock_client):
            await provider.close()
        mock_client.aclose.assert_awaited_once()


# =========================================================================
# Factory
# =========================================================================


class TestCreateProviderAdapter:
    def test_ollama(self) -> None:
        spec = _make_spec(ProviderType.OLLAMA, "ollama1", models=["llama3.1"])
        settings = _settings()
        adapter = create_provider_adapter(spec, settings)
        assert isinstance(adapter, OllamaProvider)
        assert adapter.provider_id == "ollama1"

    def test_lm_studio(self) -> None:
        spec = _make_spec(ProviderType.LM_STUDIO, "ls1", models=["local-model"])
        settings = _settings()
        adapter = create_provider_adapter(spec, settings)
        assert isinstance(adapter, LMStudioProvider)
        assert adapter.provider_id == "ls1"

    def test_openai_compatible(self) -> None:
        spec = _make_spec(ProviderType.OPENAI_COMPATIBLE, "oai1", models=["gpt-4"])
        settings = _settings()
        adapter = create_provider_adapter(spec, settings)
        assert isinstance(adapter, OpenAICompatibleProvider)
        assert adapter.provider_id == "oai1"

    def test_unknown_type_raises(self) -> None:
        spec = MagicMock(spec=ProviderSpec)
        spec.provider_type = "unknown"
        spec.id = "bad"
        spec.base_url = None
        spec.api_key = None
        spec.models = []
        settings = _settings()
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_provider_adapter(spec, settings)

    def test_fallback_base_url(self) -> None:
        spec = _make_spec(ProviderType.OLLAMA, "o1", base_url=None)
        settings = _settings()
        adapter = create_provider_adapter(spec, settings)
        assert isinstance(adapter, OllamaProvider)

    def test_custom_timeout(self) -> None:
        spec = _make_spec(ProviderType.OLLAMA, "o1")
        settings = _settings()
        adapter = create_provider_adapter(spec, settings, timeout_seconds=30.0)
        assert isinstance(adapter, OllamaProvider)


class TestProviderTypeToDefaultBaseUrl:
    def test_ollama(self) -> None:
        settings = _settings()
        url = provider_type_to_default_base_url(ProviderType.OLLAMA, settings)
        assert url == "http://localhost:11434"

    def test_lm_studio(self) -> None:
        settings = _settings()
        url = provider_type_to_default_base_url(ProviderType.LM_STUDIO, settings)
        assert url == "http://localhost:1234/v1"

    def test_unknown_returns_none(self) -> None:
        settings = _settings()
        url = provider_type_to_default_base_url(ProviderType.ANTHROPIC, settings)
        assert url is None


# =========================================================================
# LLMRouter
# =========================================================================


def _mock_provider(provider_id: str = "test_provider",
                   model_id: str = "test-model") -> MagicMock:
    p = MagicMock(spec=LLMProvider)
    p.provider_id = provider_id
    p.model_id = model_id
    p.generate = AsyncMock(return_value=CompletionResponse(
        message=_make_message(role=MessageRole.ASSISTANT),
        finish_reason=FinishReason.STOP,
        model=model_id,
    ))
    p.generate_stream = MagicMock()
    p.check_health = AsyncMock(return_value=True)
    return p


class TestLLMRouter:
    @pytest.fixture
    def router(self) -> LLMRouter:
        return LLMRouter(
            settings=_settings(),
            default_provider_id="default_provider",
            provider_preference=["preferred_provider"],
            max_retries=2,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_seconds=3600,
        )

    def test_default_provider_id(self) -> None:
        router = LLMRouter(settings=_settings())
        assert router._default_provider_id == "ollama"

    # ------------------------------------------------------------------
    # Provider adapter lifecycle
    # ------------------------------------------------------------------

    def test_register_adapter(self, router: LLMRouter) -> None:
        spec = _make_spec(ProviderType.OLLAMA, "test_adapter")
        adapter = router.register_adapter(spec)
        assert isinstance(adapter, OllamaProvider)
        assert router._registry.is_registered("test_adapter")

    def test_remove_adapter(self, router: LLMRouter) -> None:
        spec = _make_spec(ProviderType.OLLAMA, "to_remove")
        router.register_adapter(spec)
        router.remove_adapter("to_remove")
        assert not router._registry.is_registered("to_remove")

    def test_remove_adapter_clears_failure_counts(self, router: LLMRouter) -> None:
        router._failure_counts["p1"] = 5
        router._circuit_open_until["p1"] = 999.0
        router._registry.register(_mock_provider("p1"))
        router.remove_adapter("p1")
        assert "p1" not in router._failure_counts
        assert "p1" not in router._circuit_open_until

    def test_refresh_adapter(self, router: LLMRouter) -> None:
        spec = _make_spec(ProviderType.OLLAMA, "refreshable")
        router.register_adapter(spec)
        new_spec = _make_spec(ProviderType.LM_STUDIO, "refreshable")
        adapter = router.refresh_adapter(new_spec)
        assert isinstance(adapter, LMStudioProvider)

    # ------------------------------------------------------------------
    # load_providers_from_db
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_load_providers_from_db(self, router: LLMRouter) -> None:
        specs = [
            _make_spec(ProviderType.OLLAMA, "p1"),
            _make_spec(ProviderType.LM_STUDIO, "p2"),
        ]
        await router.load_providers_from_db(specs)
        assert router._registry.is_registered("p1")
        assert router._registry.is_registered("p2")

    @pytest.mark.asyncio
    async def test_load_providers_skips_disabled(self, router: LLMRouter) -> None:
        disabled = ProviderSpec(
            id="disabled_p",
            provider_type=ProviderType.OLLAMA,
            display_name="disabled",
            models=["m"],
            is_enabled=False,
        )
        await router.load_providers_from_db([disabled])
        assert not router._registry.is_registered("disabled_p")

    @pytest.mark.asyncio
    async def test_load_providers_removes_stale(self, router: LLMRouter) -> None:
        router._registry.register(_mock_provider("stale"))
        await router.load_providers_from_db([])
        assert not router._registry.is_registered("stale")

    # ------------------------------------------------------------------
    # generate (non-streaming)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_success(self, router: LLMRouter) -> None:
        p = _mock_provider("primary")
        router._registry.register(p)
        resp = await router.generate(_make_request(provider="primary"))
        assert resp.finish_reason == FinishReason.STOP
        p.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_fallback_on_timeout(self, router: LLMRouter) -> None:
        primary = _mock_provider("primary")
        primary.generate = AsyncMock(side_effect=ProviderTimeoutError("Timeout"))
        router._registry.register(primary)

        fallback = _mock_provider("default_provider")
        router._registry.register(fallback)

        resp = await router.generate(_make_request(provider="primary"))
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_all_fallbacks_exhausted(self, router: LLMRouter) -> None:
        """When all providers fail, RouterNoProviderError is raised."""
        p = _mock_provider("primary")
        p.generate = AsyncMock(side_effect=ProviderTimeoutError("Always fails"))
        router._registry.register(p)
        with pytest.raises(RouterNoProviderError):
            await router.generate(_make_request(provider="primary"))

    @pytest.mark.asyncio
    async def test_generate_authentication_error_breaks(self, router: LLMRouter) -> None:
        p = _mock_provider("primary")
        p.generate = AsyncMock(side_effect=ProviderAuthenticationError("Bad key"))
        router._registry.register(p)
        with pytest.raises(GenerationError):
            await router.generate(_make_request(provider="primary"))
        # Should not retry authentication errors
        assert p.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_generate_no_provider_available(self, router: LLMRouter) -> None:
        with pytest.raises(RouterNoProviderError):
            await router.generate(_make_request(provider="nonexistent"))

    @pytest.mark.asyncio
    async def test_generate_fallback_to_default(self, router: LLMRouter) -> None:
        default_p = _mock_provider("default_provider")
        router._registry.register(default_p)
        resp = await router.generate(_make_request(provider=None))
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_fallback_chain(self, router: LLMRouter) -> None:
        preferred = _mock_provider("preferred_provider")
        router._registry.register(preferred)
        resp = await router.generate(_make_request(provider=None))
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_generate_skips_circuit_open(self, router: LLMRouter) -> None:
        p = _mock_provider("primary")
        router._registry.register(p)
        router._circuit_open_until["primary"] = 1e12  # far in the future
        with pytest.raises(RouterNoProviderError):
            await router.generate(_make_request(provider="primary"))

    @pytest.mark.asyncio
    async def test_generate_skips_already_tried(self, router: LLMRouter) -> None:
        p = _mock_provider("primary")
        p.generate = AsyncMock(side_effect=ProviderTimeoutError("Fail"))
        router._registry.register(p)
        with pytest.raises(RouterNoProviderError):
            await router.generate(_make_request(provider="primary"))

    # ------------------------------------------------------------------
    # generate_stream
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_stream_success(self, router: LLMRouter) -> None:
        p = _mock_provider("primary")

        async def stream_gen(request: CompletionRequest) -> AsyncIterator[Any]:
            yield TextDeltaEvent(delta="hello")
            yield StreamDoneEvent(finish_reason=FinishReason.STOP)

        p.generate_stream = MagicMock(return_value=stream_gen(None))
        router._registry.register(p)

        events: list[Any] = []
        async for event in router.generate_stream(_make_request(provider="primary")):
            events.append(event)
        assert any(isinstance(e, TextDeltaEvent) for e in events)
        assert any(isinstance(e, StreamDoneEvent) for e in events)

    @pytest.mark.asyncio
    async def test_generate_stream_fallback_on_timeout_before_start(self, router: LLMRouter) -> None:
        primary = _mock_provider("primary")
        primary.generate_stream = MagicMock(side_effect=ProviderTimeoutError("Timeout"))
        router._registry.register(primary)

        fallback = _mock_provider("default_provider")
        async def _ok_gen(_request: CompletionRequest) -> AsyncIterator[Any]:
            yield TextDeltaEvent(delta="ok")
            yield StreamDoneEvent(finish_reason=FinishReason.STOP)
        fallback.generate_stream = _ok_gen
        router._registry.register(fallback)

        events: list[Any] = []
        async for event in router.generate_stream(_make_request(provider="primary")):
            events.append(event)
        assert any(isinstance(e, TextDeltaEvent) for e in events)

    @pytest.mark.asyncio
    async def test_generate_stream_error_after_start_propagates(self, router: LLMRouter) -> None:
        p = _mock_provider("primary")

        async def stream_gen(request: CompletionRequest) -> AsyncIterator[Any]:
            yield TextDeltaEvent(delta="partial")
            raise ProviderTimeoutError("Stream error after start")

        p.generate_stream = MagicMock(return_value=stream_gen(None))
        router._registry.register(p)

        with pytest.raises(ProviderTimeoutError):
            async for event in router.generate_stream(_make_request(provider="primary")):
                pass

    @pytest.mark.asyncio
    async def test_generate_stream_no_provider(self, router: LLMRouter) -> None:
        with pytest.raises(GenerationError):
            async for _ in router.generate_stream(_make_request(provider="nonexistent")):
                pass

    # ------------------------------------------------------------------
    # check_health
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_health_all(self, router: LLMRouter) -> None:
        p1, p2 = _mock_provider("p1"), _mock_provider("p2")
        router._registry.register(p1)
        router._registry.register(p2)
        result = await router.check_health()
        assert result == {"p1": True, "p2": True}

    @pytest.mark.asyncio
    async def test_check_health_single(self, router: LLMRouter) -> None:
        p = _mock_provider("p1")
        router._registry.register(p)
        result = await router.check_health(provider_id="p1")
        assert result == {"p1": True}

    @pytest.mark.asyncio
    async def test_check_health_unknown_returns_false(self, router: LLMRouter) -> None:
        result = await router.check_health(provider_id="missing")
        assert result == {"missing": False}

    @pytest.mark.asyncio
    async def test_check_health_circuit_open(self, router: LLMRouter) -> None:
        p = _mock_provider("p1")
        router._registry.register(p)
        router._circuit_open_until["p1"] = 1e12
        result = await router.check_health()
        assert result == {"p1": False}

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def test_circuit_breaker_opens_after_threshold(self, router: LLMRouter) -> None:
        router._record_failure("p1")  # count = 1
        router._record_failure("p1")  # count = 2, threshold = 2 -> open
        assert router._is_circuit_open("p1") is True

    def test_circuit_breaker_half_opens_after_reset(self) -> None:
        router = LLMRouter(
            settings=_settings(),
            circuit_breaker_threshold=2,
            circuit_breaker_reset_seconds=0,
        )
        router._record_failure("p1")
        router._record_failure("p1")
        # Since reset_seconds is 0, the circuit is already half-open
        assert router._is_circuit_open("p1") is False
        assert router._failure_counts["p1"] == 0

    def test_circuit_breaker_success_resets(self, router: LLMRouter) -> None:
        router._failure_counts["p1"] = 5
        router._record_success("p1")
        assert router._failure_counts["p1"] == 0
        assert router._is_circuit_open("p1") is False

    def test_circuit_breaker_no_open_if_below_threshold(self, router: LLMRouter) -> None:
        router._record_failure("p1")
        assert router._is_circuit_open("p1") is False

    # ------------------------------------------------------------------
    # Provider selection internals
    # ------------------------------------------------------------------

    def test_build_candidate_chain_with_explicit(self, router: LLMRouter) -> None:
        chain = router._build_candidate_chain("explicit_p")
        assert chain[0] == "explicit_p"
        assert "default_provider" in chain
        assert "preferred_provider" in chain

    def test_build_candidate_chain_deduplicates(self, router: LLMRouter) -> None:
        router._provider_preference = ["default_provider"]
        chain = router._build_candidate_chain("default_provider")
        # Should appear only once
        assert chain.count("default_provider") == 1

    def test_acquire_provider_returns_none_when_all_excluded(self, router: LLMRouter) -> None:
        p = _mock_provider("p1")
        router._registry.register(p)
        result = router._acquire_provider(
            _make_request(provider="p1"), exclude={"p1"},
        )
        assert result is None

    def test_sleep_is_overridable(self, router: LLMRouter) -> None:
        """The _sleep method is a testability hook."""
        assert router._sleep is not None
