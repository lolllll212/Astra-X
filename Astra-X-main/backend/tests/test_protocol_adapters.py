"""Unit tests for the transport layer and protocol adapters.

Covers:
- Transport: request, stream, error mapping, close
- ChatCompletionsAdapter: build_request, parse_response, parse_stream_chunk
- ResponsesApiAdapter: build_request, parse_response, parse_stream_chunk
- OpenAICompatibleProvider integration with both protocols
- ProviderSpec capability defaults
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domain.enums import ModelCapability, OpenAIProtocol, ProviderType
from app.domain.message import (
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from app.domain.provider import ProviderSpec
from app.domain.stream import (
    StreamDoneEvent,
    StreamUsageEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from app.llm.exceptions import (
    GenerationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
)
from app.llm.models import (
    CompletionRequest,
    FinishReason,
    GenerationParams,
)
from app.llm.providers.adapters.chat_completions import (
    ChatCompletionsAdapter,
    domain_to_openai_messages,
    parse_finish_reason,
)
from app.llm.providers.adapters.responses_api import (
    ResponsesApiAdapter,
    domain_to_responses_input,
)
from app.llm.providers.openai_compatible import OpenAICompatibleProvider
from app.llm.providers.transport import Transport

# =========================================================================
# Helpers
# =========================================================================


def _make_message(text: str = "hello", role: str = "user",
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


async def _async_iter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


def _mock_stream_client(lines: list[str]) -> MagicMock:
    """Create a mock httpx client for Transport.stream()."""
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


def _mock_request_client(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Create a mock httpx client for Transport.request()."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json = MagicMock(return_value=json_data or {})

    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=mock_response)

    return mock_client


# =========================================================================
# Transport
# =========================================================================


class TestTransport:
    @pytest.mark.asyncio
    async def test_request_success(self) -> None:
        transport = Transport(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            timeout=30.0,
        )
        mock_client = _mock_request_client(200, {"id": "resp1"})
        with patch.object(transport, "_client", mock_client):
            result = await transport.request("GET", "/models")

        assert result == {"id": "resp1"}
        mock_client.request.assert_awaited_once_with("GET", "/models", json=None)

    @pytest.mark.asyncio
    async def test_request_with_json_body(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        mock_client = _mock_request_client(200, {"choices": [{}]})
        with patch.object(transport, "_client", mock_client):
            result = await transport.request("POST", "/chat", json_data={"model": "gpt-4"})

        assert result == {"choices": [{}]}
        mock_client.request.assert_awaited_once_with("POST", "/chat", json={"model": "gpt-4"})

    @pytest.mark.asyncio
    async def test_request_timeout_error(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        mock_client = MagicMock()
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with patch.object(transport, "_client", mock_client):
            with pytest.raises(ProviderTimeoutError, match="timed out"):
                await transport.request("GET", "/models")

    @pytest.mark.asyncio
    async def test_request_connection_error(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        mock_client = MagicMock()
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with patch.object(transport, "_client", mock_client):
            with pytest.raises(ProviderConnectionError, match="Could not connect"):
                await transport.request("GET", "/models")

    @pytest.mark.asyncio
    async def test_request_authentication_error(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("401", request=MagicMock(), response=mock_response),
        )
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_client", mock_client):
            with pytest.raises(ProviderAuthenticationError, match="401 Unauthorized"):
                await transport.request("GET", "/models")

    @pytest.mark.asyncio
    async def test_request_generation_error(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response),
        )
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch.object(transport, "_client", mock_client):
            with pytest.raises(GenerationError, match="429"):
                await transport.request("GET", "/models")

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        lines = [
            "data: " + json.dumps({"type": "delta", "content": "Hello"}),
            "data: " + json.dumps({"type": "delta", "content": " World"}),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_client(lines)
        with patch.object(transport, "_client", mock_client):
            chunks: list[dict] = []
            async for chunk in transport.stream("POST", "/chat", json_data={}):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0]["content"] == "Hello"
        assert chunks[1]["content"] == " World"

    @pytest.mark.asyncio
    async def test_stream_skips_non_data_lines(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        lines = [
            ": keep-alive comment",
            "data: " + json.dumps({"content": "actual data"}),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_client(lines)
        with patch.object(transport, "_client", mock_client):
            chunks: list[dict] = []
            async for chunk in transport.stream("POST", "/chat", json_data={}):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["content"] == "actual data"

    @pytest.mark.asyncio
    async def test_stream_timeout(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(
            side_effect=httpx.TimeoutException("stream timed out"),
        )

        with patch.object(transport, "_client", mock_client):
            with pytest.raises(ProviderTimeoutError):
                async for _ in transport.stream("POST", "/chat", json_data={}):
                    pass

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        transport = Transport(base_url="https://api.example.com/v1")
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        with patch.object(transport, "_client", mock_client):
            await transport.close()
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_client_injection(self) -> None:
        custom_client = MagicMock()
        transport = Transport(
            base_url="https://api.example.com/v1",
            client=custom_client,
        )
        assert transport._client is custom_client


# =========================================================================
# ChatCompletionsAdapter
# =========================================================================


class TestChatCompletionsAdapter:
    @pytest.fixture
    def adapter(self) -> ChatCompletionsAdapter:
        return ChatCompletionsAdapter()

    def test_protocol(self, adapter: ChatCompletionsAdapter) -> None:
        assert adapter.protocol == OpenAIProtocol.CHAT_COMPLETIONS

    def test_endpoint(self, adapter: ChatCompletionsAdapter) -> None:
        assert adapter.endpoint == "/chat/completions"

    def test_build_request_basic(self, adapter: ChatCompletionsAdapter) -> None:
        request = _make_request(text="Hello", model="gpt-4")
        payload = adapter.build_request(request, "gpt-4")

        assert payload["model"] == "gpt-4"
        assert payload["stream"] is True
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] is None

    def test_build_request_with_params(self, adapter: ChatCompletionsAdapter) -> None:
        request = CompletionRequest(
            messages=[_make_message(text="Hi")],
            model="gpt-4",
            params=GenerationParams(temperature=0.3, max_tokens=100, stop=["\n"]),
        )
        payload = adapter.build_request(request, "gpt-4")

        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 100
        assert payload["stop"] == ["\n"]

    def test_build_request_with_tools(self, adapter: ChatCompletionsAdapter) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role="assistant",
            content=[ToolCallBlock(tool_call_id="tc1", tool_name="calc", arguments={"x": 1})],
            created_at=datetime.now(UTC),
        )
        request = CompletionRequest(
            messages=[msg],
            model="gpt-4",
        )
        payload = adapter.build_request(request, "gpt-4")

        assert "tool_calls" in payload["messages"][0]
        assert payload["messages"][0]["tool_calls"][0]["function"]["name"] == "calc"

    def test_parse_response_text(self, adapter: ChatCompletionsAdapter) -> None:
        raw = {
            "choices": [{
                "message": {"content": "Hello world"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        resp = adapter.parse_response(raw, "gpt-4")

        assert resp.finish_reason == FinishReason.STOP
        assert resp.model == "gpt-4"
        assert len(resp.message.content) == 1
        assert isinstance(resp.message.content[0], TextBlock)
        assert resp.message.content[0].text == "Hello world"
        assert resp.usage is not None
        assert resp.usage.total_tokens == 8

    def test_parse_response_tool_calls(self, adapter: ChatCompletionsAdapter) -> None:
        raw = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "tc1",
                        "function": {"name": "calculator", "arguments": '{"x": 1, "y": 2}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        resp = adapter.parse_response(raw, "gpt-4")

        assert resp.finish_reason == FinishReason.TOOL_CALLS
        assert len(resp.message.content) == 1
        assert isinstance(resp.message.content[0], ToolCallBlock)
        assert resp.message.content[0].tool_name == "calculator"
        assert resp.message.content[0].arguments == {"x": 1, "y": 2}

    def test_parse_stream_chunk_text_delta(self, adapter: ChatCompletionsAdapter) -> None:
        chunk = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        events, state = adapter.parse_stream_chunk(chunk)

        assert len(events) == 1
        assert isinstance(events[0], TextDeltaEvent)
        assert events[0].delta == "Hello"

    def test_parse_stream_chunk_done(self, adapter: ChatCompletionsAdapter) -> None:
        chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        events, state = adapter.parse_stream_chunk(chunk)

        assert any(isinstance(e, StreamDoneEvent) for e in events)
        assert not any(isinstance(e, ToolCallEndEvent) for e in events)

    def test_parse_stream_chunk_with_usage(self, adapter: ChatCompletionsAdapter) -> None:
        chunk = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        events, state = adapter.parse_stream_chunk(chunk)

        assert any(isinstance(e, StreamUsageEvent) for e in events)
        assert any(e.total_tokens == 30 for e in events if isinstance(e, StreamUsageEvent))

    def test_parse_stream_chunk_tool_call_start(self, adapter: ChatCompletionsAdapter) -> None:
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{"index": 0, "id": "tc1", "function": {"name": "calc", "arguments": ""}}],
                },
                "finish_reason": None,
            }],
        }
        events, state = adapter.parse_stream_chunk(chunk)

        assert any(isinstance(e, ToolCallStartEvent) for e in events)
        assert 0 in state
        assert state[0]["id"] == "tc1"

    def test_parse_stream_chunk_tool_call_delta(self, adapter: ChatCompletionsAdapter) -> None:
        initial_state = {0: {"id": "tc1", "name": "calc", "arguments": ""}}
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{"index": 0, "function": {"arguments": '{"x": 1}'}}],
                },
                "finish_reason": None,
            }],
        }
        events, state = adapter.parse_stream_chunk(chunk, initial_state)

        assert any(isinstance(e, ToolCallDeltaEvent) for e in events)
        assert state[0]["arguments"] == '{"x": 1}'

    def test_parse_stream_chunk_tool_call_finish(self, adapter: ChatCompletionsAdapter) -> None:
        initial_state = {0: {"id": "tc1", "name": "calc", "arguments": '{"x": 1}'}}
        chunk = {
            "choices": [{
                "delta": {},
                "finish_reason": "tool_calls",
            }],
        }
        events, state = adapter.parse_stream_chunk(chunk, initial_state)

        assert any(isinstance(e, ToolCallEndEvent) for e in events)
        end_event = next(e for e in events if isinstance(e, ToolCallEndEvent))
        assert end_event.tool_call_id == "tc1"
        assert end_event.arguments == {"x": 1}
        assert any(isinstance(e, StreamDoneEvent) for e in events)

    def test_parse_stream_chunk_empty_choices(self, adapter: ChatCompletionsAdapter) -> None:
        chunk = {"choices": []}
        events, state = adapter.parse_stream_chunk(chunk)
        assert events == []

    def test_parse_stream_chunk_done_returns_false(self, adapter: ChatCompletionsAdapter) -> None:
        assert adapter.parse_stream_chunk_done({"choices": []}) is False

    def test_parse_stream_chunk_no_tool_calls(self, adapter: ChatCompletionsAdapter) -> None:
        msg = _make_message(text="hello")
        request = _make_request(text="hello")
        payload = adapter.build_request(request, "gpt-4")
        assert payload["messages"][0]["content"] == "hello"


# =========================================================================
# ResponsesApiAdapter
# =========================================================================


class TestResponsesApiAdapter:
    @pytest.fixture
    def adapter(self) -> ResponsesApiAdapter:
        return ResponsesApiAdapter()

    def test_protocol(self, adapter: ResponsesApiAdapter) -> None:
        assert adapter.protocol == OpenAIProtocol.RESPONSES

    def test_endpoint(self, adapter: ResponsesApiAdapter) -> None:
        assert adapter.endpoint == "/responses"

    def test_build_request_basic(self, adapter: ResponsesApiAdapter) -> None:
        request = _make_request(text="Hello", model="gpt-4o")
        payload = adapter.build_request(request, "gpt-4o")

        assert payload["model"] == "gpt-4o"
        assert payload["stream"] is True
        assert payload["input"][0]["content"] == "Hello"
        assert payload["temperature"] == 0.7

    def test_build_request_with_vision(self, adapter: ResponsesApiAdapter) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role="user",
            content=[
                TextBlock(text="what is this?"),
                ImageBlock(data_uri="data:image/png;base64,..."),
            ],
            created_at=datetime.now(UTC),
        )
        request = CompletionRequest(messages=[msg], model="gpt-4o")
        payload = adapter.build_request(request, "gpt-4o")

        assert isinstance(payload["input"][0]["content"], list)
        assert payload["input"][0]["content"][0]["type"] == "input_text"
        assert payload["input"][0]["content"][1]["type"] == "input_image"

    def test_parse_response_text(self, adapter: ResponsesApiAdapter) -> None:
        raw = {
            "id": "resp1",
            "output": [{"type": "message", "content": "Hello world"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        resp = adapter.parse_response(raw, "gpt-4o")

        assert resp.finish_reason == FinishReason.STOP
        assert len(resp.message.content) == 1
        assert isinstance(resp.message.content[0], TextBlock)
        assert resp.message.content[0].text == "Hello world"
        assert resp.usage is not None

    def test_parse_response_tool_calls(self, adapter: ResponsesApiAdapter) -> None:
        raw = {
            "id": "resp1",
            "output": [{
                "type": "function_call",
                "id": "fc1",
                "function": {"name": "calculator", "arguments": '{"x": 1}'},
            }],
        }
        resp = adapter.parse_response(raw, "gpt-4o")

        assert len(resp.message.content) == 1
        assert isinstance(resp.message.content[0], ToolCallBlock)
        assert resp.message.content[0].tool_name == "calculator"

    def test_parse_response_incomplete_status(self, adapter: ResponsesApiAdapter) -> None:
        raw = {
            "id": "resp1",
            "output": [],
            "status": "incomplete",
        }
        resp = adapter.parse_response(raw, "gpt-4o")
        assert resp.finish_reason == FinishReason.LENGTH

    def test_parse_response_failed_status(self, adapter: ResponsesApiAdapter) -> None:
        raw = {
            "id": "resp1",
            "output": [],
            "status": "failed",
        }
        resp = adapter.parse_response(raw, "gpt-4o")
        assert resp.finish_reason == FinishReason.ERROR

    def test_parse_stream_chunk_text_delta(self, adapter: ResponsesApiAdapter) -> None:
        chunk = {"type": "response.output_text.delta", "delta": "Hello", "item_id": "msg1"}
        events, state = adapter.parse_stream_chunk(chunk)

        assert len(events) == 1
        assert isinstance(events[0], TextDeltaEvent)
        assert events[0].delta == "Hello"

    def test_parse_stream_chunk_function_call_delta(self, adapter: ResponsesApiAdapter) -> None:
        chunk = {
            "type": "response.function_call_arguments.delta",
            "delta": '{"x":',
            "item_id": "fc1",
        }
        events, state = adapter.parse_stream_chunk(chunk)

        assert len(events) == 1
        assert isinstance(events[0], ToolCallDeltaEvent)
        assert events[0].tool_call_id == "fc1"
        assert events[0].arguments_delta == '{"x":'

    def test_parse_stream_chunk_function_call_done(self, adapter: ResponsesApiAdapter) -> None:
        chunk = {
            "type": "response.function_call_arguments.done",
            "item_id": "fc1",
            "name": "calculator",
            "arguments": '{"x": 1}',
        }
        events, state = adapter.parse_stream_chunk(chunk)

        assert len(events) == 1
        assert isinstance(events[0], ToolCallEndEvent)
        assert events[0].tool_call_id == "fc1"
        assert events[0].tool_name == "calculator"
        assert events[0].arguments == {"x": 1}

    def test_parse_stream_chunk_completed(self, adapter: ResponsesApiAdapter) -> None:
        chunk = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        }
        events, state = adapter.parse_stream_chunk(chunk)

        assert any(isinstance(e, StreamDoneEvent) for e in events)
        assert any(isinstance(e, StreamUsageEvent) for e in events)

    def test_parse_stream_chunk_completed_incomplete(self, adapter: ResponsesApiAdapter) -> None:
        chunk = {
            "type": "response.completed",
            "response": {
                "status": "incomplete",
            },
        }
        events, state = adapter.parse_stream_chunk(chunk)

        done_event = next(e for e in events if isinstance(e, StreamDoneEvent))
        assert done_event.finish_reason == FinishReason.LENGTH

    def test_parse_stream_chunk_completed_failed(self, adapter: ResponsesApiAdapter) -> None:
        chunk = {
            "type": "response.completed",
            "response": {
                "status": "failed",
            },
        }
        events, state = adapter.parse_stream_chunk(chunk)

        done_event = next(e for e in events if isinstance(e, StreamDoneEvent))
        assert done_event.finish_reason == FinishReason.ERROR

    def test_parse_stream_chunk_in_progress(self, adapter: ResponsesApiAdapter) -> None:
        chunk = {"type": "response.in_progress"}
        events, state = adapter.parse_stream_chunk(chunk)
        assert events == []

    def test_parse_stream_chunk_done_true(self, adapter: ResponsesApiAdapter) -> None:
        assert adapter.parse_stream_chunk_done({"type": "response.completed"}) is True

    def test_parse_stream_chunk_done_false(self, adapter: ResponsesApiAdapter) -> None:
        assert adapter.parse_stream_chunk_done({"type": "response.output_text.delta"}) is False

    def test_build_request_no_max_tokens(self, adapter: ResponsesApiAdapter) -> None:
        request = CompletionRequest(
            messages=[_make_message(text="Hi")],
            model="gpt-4o",
            params=GenerationParams(max_tokens=None),
        )
        payload = adapter.build_request(request, "gpt-4o")

        assert payload["model"] == "gpt-4o"
        assert payload["max_output_tokens"] is None

    def test_build_request_with_max_tokens(self, adapter: ResponsesApiAdapter) -> None:
        request = CompletionRequest(
            messages=[_make_message(text="Hi")],
            model="gpt-4o",
            params=GenerationParams(max_tokens=200),
        )
        payload = adapter.build_request(request, "gpt-4o")

        assert payload["max_output_tokens"] == 200


# =========================================================================
# domain_to_openai_messages
# =========================================================================


class TestDomainToOpenAIMessages:
    def test_text_message(self) -> None:
        msg = _make_message(text="hello")
        result = domain_to_openai_messages([msg])
        assert result == [{"role": "user", "content": "hello"}]

    def test_multimodal(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role="user",
            content=[
                TextBlock(text="what is this?"),
                ImageBlock(data_uri="data:image/png;base64,..."),
            ],
            created_at=datetime.now(UTC),
        )
        result = domain_to_openai_messages([msg])
        assert len(result) == 1
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"

    def test_tool_call(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role="assistant",
            content=[ToolCallBlock(tool_call_id="tc1", tool_name="calc", arguments={"x": 1})],
            created_at=datetime.now(UTC),
        )
        result = domain_to_openai_messages([msg])
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "calc"

    def test_tool_result(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role="tool",
            content=[ToolResultBlock(tool_call_id="tc1", tool_name="calc", output="42")],
            created_at=datetime.now(UTC),
        )
        result = domain_to_openai_messages([msg])
        assert result[0]["role"] == "tool"


# =========================================================================
# parse_finish_reason
# =========================================================================


class TestParseFinishReason:
    def test_stop(self) -> None:
        assert parse_finish_reason("stop") == FinishReason.STOP

    def test_length(self) -> None:
        assert parse_finish_reason("length") == FinishReason.LENGTH

    def test_tool_calls(self) -> None:
        assert parse_finish_reason("tool_calls") == FinishReason.TOOL_CALLS

    def test_defaults_to_stop(self) -> None:
        assert parse_finish_reason(None) == FinishReason.STOP
        assert parse_finish_reason("") == FinishReason.STOP
        assert parse_finish_reason("weird_reason") == FinishReason.STOP


# =========================================================================
# domain_to_responses_input
# =========================================================================


class TestDomainToResponsesInput:
    def test_text_message(self) -> None:
        msg = _make_message(text="hello")
        result = domain_to_responses_input([msg])
        assert result[0]["content"] == "hello"

    def test_multimodal(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role="user",
            content=[
                TextBlock(text="what is this?"),
                ImageBlock(data_uri="data:image/png;base64,..."),
            ],
            created_at=datetime.now(UTC),
        )
        result = domain_to_responses_input([msg])
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][1]["type"] == "input_image"

    def test_tool_result(self) -> None:
        msg = Message(
            id="m1", conversation_id="c1", role="tool",
            content=[ToolResultBlock(tool_call_id="tc1", tool_name="calc", output="42")],
            created_at=datetime.now(UTC),
        )
        result = domain_to_responses_input([msg])
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "42"


# =========================================================================
# OpenAICompatibleProvider — Protocol Integration
# =========================================================================


class TestOpenAICompatibleProviderProtocol:
    @pytest.mark.asyncio
    async def test_default_protocol_is_chat_completions(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            model="gpt-4o",
        )
        assert provider.protocol == OpenAIProtocol.CHAT_COMPLETIONS

    @pytest.mark.asyncio
    async def test_chat_completions_protocol(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            model="gpt-4o",
            protocol=OpenAIProtocol.CHAT_COMPLETIONS,
        )
        assert provider.protocol == OpenAIProtocol.CHAT_COMPLETIONS
        assert provider._adapter.protocol == OpenAIProtocol.CHAT_COMPLETIONS

    @pytest.mark.asyncio
    async def test_responses_protocol(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            model="gpt-4o",
            protocol=OpenAIProtocol.RESPONSES,
        )
        assert provider.protocol == OpenAIProtocol.RESPONSES
        assert provider._adapter.protocol == OpenAIProtocol.RESPONSES

    @pytest.mark.asyncio
    async def test_non_streaming_generate_with_chat_completions(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            model="gpt-4o",
            protocol=OpenAIProtocol.CHAT_COMPLETIONS,
        )
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}),
            "data: " + json.dumps({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }),
            "data: [DONE]",
        ]
        mock_client = _mock_stream_client(lines)
        with patch.object(provider._transport, "_client", mock_client):
            resp = await provider.generate(_make_request())

        assert resp.message is not None
        assert "Hi" in str(resp.message.content[0].text)
        assert resp.usage is not None
        assert resp.usage.total_tokens == 5

    @pytest.mark.asyncio
    async def test_non_streaming_generate_with_responses_api(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            model="gpt-4o",
            protocol=OpenAIProtocol.RESPONSES,
        )
        lines = [
            "data: " + json.dumps({"type": "response.output_text.delta", "delta": "Hello", "item_id": "msg1"}),
            "data: " + json.dumps({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            }),
        ]
        mock_client = _mock_stream_client(lines)
        with patch.object(provider._transport, "_client", mock_client):
            resp = await provider.generate(_make_request())

        assert resp.message is not None
        assert "Hello" in str(resp.message.content[0].text)
        assert resp.usage is not None

    @pytest.mark.asyncio
    async def test_supports_capability(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            model="gpt-4o",
        )
        assert provider.supports_capability(ModelCapability.CHAT)
        assert provider.supports_capability(ModelCapability.STREAMING)
        assert not provider.supports_capability(ModelCapability.EMBEDDING)


# =========================================================================
# ProviderSpec Capability Defaults
# =========================================================================


class TestProviderSpecCapabilities:
    def test_default_protocol(self) -> None:
        spec = ProviderSpec(
            id="test",
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            display_name="test",
        )
        assert spec.protocol == OpenAIProtocol.CHAT_COMPLETIONS
        assert spec.supports_streaming is True
        assert spec.supports_vision is False
        assert spec.supports_tool_calling is True
        assert spec.supports_parallel_tool_calls is True
        assert spec.max_tool_calls_per_request == 10

    def test_responses_protocol(self) -> None:
        spec = ProviderSpec(
            id="test",
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            display_name="test",
            protocol=OpenAIProtocol.RESPONSES,
            supports_responses_api=True,
        )
        assert spec.protocol == OpenAIProtocol.RESPONSES
        assert spec.supports_responses_api is True

    def test_custom_capabilities(self) -> None:
        spec = ProviderSpec(
            id="test",
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            display_name="test",
            supports_vision=True,
            supports_structured_output=True,
            max_tool_calls_per_request=5,
        )
        assert spec.supports_vision is True
        assert spec.supports_structured_output is True
        assert spec.max_tool_calls_per_request == 5
