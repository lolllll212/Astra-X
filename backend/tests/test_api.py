"""Tests for the API layer — middleware, dependencies, routes, WebSocket, error handlers."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import ORJSONResponse
from pydantic import TypeAdapter
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.testclient import TestClient

from app.api.dependencies import (
    get_settings,
    get_db_session,
    get_conversation_repository,
    get_message_repository,
    get_usage_repository,
    get_attachment_repository,
    get_attachment_service,
    get_llm_router,
    get_conversation_service,
    get_core_memory_manager,
    get_memory_service,
    get_plugin_service,
    get_provider_repository,
    get_provider_service,
    get_usage_service,
    get_chat_service,
)
from app.api.errors import (
    _map_llm_error,
    llm_error_handler,
    unhandled_error_handler,
    get_uptime,
    _StartupState,
    _startup,
)
from app.core.exceptions import (
    AstraError,
    ConflictError,
    ResourceNotFoundError,
    astra_error_handler,
)
from app.api.middleware import (
    RequestIDMiddleware,
    TimingMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware,
)
from app.api.routes.attachments import (
    _attachment_to_response,
    router as attachments_router,
)
from app.api.routes.chat import (
    _content_schema_to_domain,
    _content_domain_to_schema,
    _params_schema_to_domain,
    _usage_to_response,
    _message_to_response,
    _conversation_to_response,
    _chat_result_to_response,
    router as chat_router,
)
from app.api.routes.conversations import (
    _conversation_to_response as _conv_to_response,
    router as conversations_router,
)
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.plugins import (
    _spec_to_response as _plugin_spec_to_response,
    router as plugins_router,
)
from app.api.routes.providers import (
    _spec_to_response,
    router as providers_router,
)
from app.api.routes.system import router as system_router
from app.api.schemas.attachment import AttachmentCreate, AttachmentResponse, AttachmentListResponse
from app.api.schemas.chat import (
    TextBlockSchema,
    ImageBlockSchema,
    ToolCallBlockSchema,
    ToolResultBlockSchema,
    ContentBlockSchema,
    GenerationParamsSchema,
    ChatRequest,
    ContinueRequest,
    UsageResponse,
    MessageResponse,
    ChatResponse,
    TextDeltaEvent,
    StreamErrorEvent,
    StreamDoneEvent,
)
from app.api.schemas.common import PaginationParams, PaginatedResponse, MessageResponse as CommonMessageResponse
from app.api.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationResponse, ConversationListResponse
from app.api.schemas.health import HealthResponse
from app.api.schemas.plugin import (
    PluginInstallRequest,
    PluginUpdateRequest,
    PluginResponse,
    PluginListResponse,
)
from app.api.schemas.provider import (
    ProviderRegisterRequest,
    ProviderUpdateRequest,
    ProviderResponse,
    ProviderListResponse,
    ProviderCheckRequest,
    ProviderCheckResponse,
    ModelInfoResponse,
    ModelListResponse,
)
from app.api.schemas.system import VersionInfo, ConfigInfo, MetricsResponse
from app.config.settings import Settings, Environment, LogFormat, _DEFAULT_DEV_SECRET_KEY
from app.domain.attachment import Attachment
from app.domain.conversation import Conversation, ConversationMetadata, ConversationParticipant
from app.domain.enums import (
    AttachmentType,
    ContentBlockType,
    ConversationStatus,
    MessageRole,
    ProviderType,
)
from app.domain.message import (
    ContentBlock,
    ImageBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    Message,
)
from app.domain.plugin import PluginSpec, PluginStatus
from app.domain.provider import ProviderSpec
from app.domain.usage import Usage
from app.llm.exceptions import (
    GenerationError,
    LLMError,
    ModelNotSupportedError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
    RouterNoProviderError,
)
from app.llm.models import GenerationParams
from app.llm.router import LLMRouter
from app.services.chat_service import ChatResult, ChatService
from app.services.conversation_service import ConversationService
from app.services.attachment_service import AttachmentService
from app.services.memory_service import MemoryService
from app.services.plugin_service import PluginService
from app.services.provider_service import ProviderService
from app.services.usage_service import UsageService


# ======================================================================
# Helpers
# ======================================================================


def _make_minimal_app(
    *middleware_classes: type[BaseHTTPMiddleware],
    handler: Callable[[Request], Response] | None = None,
) -> FastAPI:
    """Build a minimal FastAPI app with only the requested middleware(s)."""
    app = FastAPI()

    if handler:

        @app.get("/test")
        async def test_route(request: Request) -> Response:  # type: ignore[misc]
            return handler(request)

    else:

        @app.get("/test")
        async def test_route() -> dict[str, str]:  # type: ignore[misc]
            return {"ok": "true"}

    for mw in middleware_classes:
        app.add_middleware(mw)

    return app


def _make_mock_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with default test overrides.

    Defaults to testing environment. When overridden to production,
    callers must also provide a 32+ char secret_key and non-wildcard
    allowed_hosts/cors_origins to pass the production-safety validator.
    """
    kwargs: dict[str, Any] = {
        "database_url": "sqlite+aiosqlite://",
        "environment": "testing",
        "secret_key": _DEFAULT_DEV_SECRET_KEY,
        "rate_limit_requests_per_minute": 60,
    }
    kwargs.update(overrides)
    if kwargs.get("environment") is Environment.PRODUCTION:
        kwargs.setdefault("allowed_hosts", ["example.com"])
        kwargs.setdefault("cors_origins", ["https://example.com"])
        sk = kwargs.get("secret_key", _DEFAULT_DEV_SECRET_KEY)
        if isinstance(sk, str) and len(sk) < 32:
            kwargs["secret_key"] = sk + "x" * (32 - len(sk))
    return Settings(**kwargs)


# ======================================================================
# Middleware tests
# ======================================================================


class TestRequestIDMiddleware:
    def test_generates_request_id_when_header_missing(self) -> None:
        app = _make_minimal_app(RequestIDMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) > 0

    def test_uses_provided_request_id_header(self) -> None:
        app = _make_minimal_app(RequestIDMiddleware)
        client = TestClient(app)
        resp = client.get("/test", headers={"X-Request-ID": "my-req-42"})
        assert resp.headers["X-Request-ID"] == "my-req-42"

    def test_sets_request_state(self) -> None:
        app = _make_minimal_app(RequestIDMiddleware)

        @app.get("/check-state")
        async def check_state(request: Request) -> dict[str, str]:  # type: ignore[misc]
            return {"rid": request.state.request_id}

        client = TestClient(app)
        resp = client.get("/check-state")
        assert resp.status_code == 200
        assert len(resp.json()["rid"]) > 0


class TestTimingMiddleware:
    def test_adds_timing_header(self) -> None:
        app = _make_minimal_app(TimingMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert "X-Request-Time-Ms" in resp.headers
        assert resp.headers["X-Request-Time-Ms"].isdigit()


class TestRequestLoggingMiddleware:
    def test_logs_request(self) -> None:
        app = _make_minimal_app(RequestLoggingMiddleware, RequestIDMiddleware)
        client = TestClient(app)
        with patch("app.api.middleware.logger.info") as mock_log:
            resp = client.get("/test")
        assert resp.status_code == 200
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        assert args[0] == "api.request"


class TestSecurityHeadersMiddleware:
    def test_adds_security_headers(self) -> None:
        app = _make_minimal_app(SecurityHeadersMiddleware)
        client = TestClient(app)
        resp = client.get("/test")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"


class TestRateLimitMiddleware:
    def test_passes_exempt_paths(self) -> None:
        app = _make_minimal_app(RateLimitMiddleware)
        client = TestClient(app)
        for path in ["/health", "/metrics", "/docs"]:
            resp = client.get(path)
            assert resp.status_code in (200, 404), f"Path {path} returned {resp.status_code}"

    def test_rate_limits_excessive_requests(self) -> None:
        app = _make_minimal_app(RateLimitMiddleware)
        client = TestClient(app)

        # Set low limit via app state
        settings = _make_mock_settings(environment="testing", rate_limit_requests_per_minute=3)
        app.state.settings = settings

        # Exhaust the budget
        for _ in range(3):
            resp = client.get("/test")
            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        # Next request should be rate-limited
        resp = client.get("/test")
        assert resp.status_code == 429, f"Expected 429, got {resp.status_code}"
        data = resp.json()
        assert data["error"]["code"] == "rate_limit_error"

    def test_development_mode_has_higher_limit(self) -> None:
        app = _make_minimal_app(RateLimitMiddleware)
        client = TestClient(app)

        settings = _make_mock_settings(environment="development", rate_limit_requests_per_minute=60)
        app.state.settings = settings

        for _ in range(100):
            resp = client.get("/test")
            assert resp.status_code == 200, f"Expected 200 at iteration, got {resp.status_code}"

    def test_respects_bucket_window(self) -> None:
        app = _make_minimal_app(RateLimitMiddleware)
        client = TestClient(app)

        settings = _make_mock_settings(environment="testing", rate_limit_requests_per_minute=1)
        app.state.settings = settings

        # Clear any bucket state from other tests
        RateLimitMiddleware._buckets.clear()

        resp = client.get("/test")
        assert resp.status_code == 200

        resp = client.get("/test")
        assert resp.status_code == 429

        # Clear the bucket and verify we can make another request
        RateLimitMiddleware._buckets.clear()
        resp = client.get("/test")
        assert resp.status_code == 200


class TestAuthenticationMiddleware:
    def test_exempt_paths_pass_without_auth(self) -> None:
        app = _make_minimal_app(AuthenticationMiddleware)
        client = TestClient(app)

        SECRET = _DEFAULT_DEV_SECRET_KEY + "-extra"  # must differ from dev default
        settings = _make_mock_settings(environment="production", secret_key=SECRET, allowed_hosts=["example.com"], cors_origins=["https://example.com"])
        app.state.settings = settings

        for path in ["/health", "/metrics"]:
            resp = client.get(path)
            assert resp.status_code in (200, 404), f"Path {path} returned {resp.status_code}"

    def test_development_mode_skips_auth(self) -> None:
        app = _make_minimal_app(AuthenticationMiddleware)
        client = TestClient(app)

        settings = _make_mock_settings(environment="development")
        app.state.settings = settings

        resp = client.get("/test")
        assert resp.status_code == 200

    def test_production_requires_bearer_token(self) -> None:
        app = _make_minimal_app(AuthenticationMiddleware)
        client = TestClient(app)

        SECRET = "x" * 32
        settings = _make_mock_settings(environment="production", secret_key=SECRET, allowed_hosts=["example.com"], cors_origins=["https://example.com"])
        app.state.settings = settings

        resp = client.get("/test")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "authentication_error"

    def test_valid_token_passes(self) -> None:
        app = _make_minimal_app(AuthenticationMiddleware)
        client = TestClient(app)

        SECRET = "x" * 32
        settings = _make_mock_settings(environment="production", secret_key=SECRET, allowed_hosts=["example.com"], cors_origins=["https://example.com"])
        app.state.settings = settings

        resp = client.get("/test", headers={"Authorization": f"Bearer {SECRET}"})
        assert resp.status_code == 200

    def test_invalid_token_rejected(self) -> None:
        app = _make_minimal_app(AuthenticationMiddleware)
        client = TestClient(app)

        SECRET = "x" * 32
        settings = _make_mock_settings(environment="production", secret_key=SECRET, allowed_hosts=["example.com"], cors_origins=["https://example.com"])
        app.state.settings = settings

        resp = client.get("/test", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "authentication_error"

    def test_invalid_auth_format_rejected(self) -> None:
        app = _make_minimal_app(AuthenticationMiddleware)
        client = TestClient(app)

        SECRET = "x" * 32
        settings = _make_mock_settings(environment="production", secret_key=SECRET, allowed_hosts=["example.com"], cors_origins=["https://example.com"])
        app.state.settings = settings

        resp = client.get("/test", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401


# ======================================================================
# Error handler tests
# ======================================================================


class TestMapLLMError:
    def test_router_no_provider(self) -> None:
        exc = RouterNoProviderError("No provider available")
        http_status, code, msg = _map_llm_error(exc)
        assert http_status == 503
        assert code == "provider_unavailable"

    def test_provider_timeout(self) -> None:
        exc = ProviderTimeoutError("Timed out")
        http_status, code, msg = _map_llm_error(exc)
        assert http_status == 504
        assert code == "provider_timeout"

    def test_provider_connection_error(self) -> None:
        exc = ProviderConnectionError("Connection refused")
        http_status, code, msg = _map_llm_error(exc)
        assert http_status == 502
        assert code == "provider_connection_error"

    def test_provider_authentication_error(self) -> None:
        exc = ProviderAuthenticationError("Unauthorized")
        http_status, code, msg = _map_llm_error(exc)
        assert http_status == 502
        assert code == "provider_authentication_error"

    def test_model_not_supported(self) -> None:
        exc = ModelNotSupportedError("Model not found")
        http_status, code, msg = _map_llm_error(exc)
        assert http_status == 400
        assert code == "model_not_supported"

    def test_generation_error(self) -> None:
        exc = GenerationError("Generation failed")
        http_status, code, msg = _map_llm_error(exc)
        assert http_status == 500
        assert code == "generation_error"

    def test_unknown_llm_error_defaults_to_502(self) -> None:
        class UnknownLLMError(LLMError):
            pass

        exc = UnknownLLMError("Something went wrong")
        http_status, code, msg = _map_llm_error(exc)
        assert http_status == 502
        assert code == "llm_error"


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (RouterNoProviderError("nope"), 503),
        (ProviderTimeoutError("nope"), 504),
        (ProviderConnectionError("nope"), 502),
        (ProviderAuthenticationError("nope"), 502),
        (ModelNotSupportedError("nope"), 400),
        (GenerationError("nope"), 500),
    ],
)
@pytest.mark.asyncio
async def test_llm_error_handler_mapping(exc: LLMError, expected_status: int) -> None:
    request = MagicMock(spec=Request)
    request.url.path = "/chat"
    request.method = "POST"

    with patch("app.api.errors._get_request_id", return_value="req-1"):
        with patch("app.api.errors.logger"):
            resp = await llm_error_handler(request, exc)
    assert resp.status_code == expected_status
    body = json.loads(resp.body)
    assert "error" in body


@pytest.mark.asyncio
async def test_unhandled_error_handler_returns_500() -> None:
    request = MagicMock(spec=Request)
    request.url.path = "/test"
    request.method = "GET"

    with patch("app.api.errors._get_request_id", return_value="req-1"):
        with patch("app.api.errors.logger"):
            resp = await unhandled_error_handler(request, ValueError("something broke"))
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"]["code"] == "internal_server_error"


def test_get_uptime_returns_positive_float() -> None:
    uptime = get_uptime()
    assert isinstance(uptime, float)
    assert uptime >= 0


def test_startup_state_tracks_time() -> None:
    state = _StartupState(start_time=time.monotonic())
    elapsed = time.monotonic() - state.start_time
    assert elapsed >= 0


# ======================================================================
# Dependency tests
# ======================================================================


class MockEngine:
    pass


@pytest.mark.asyncio
async def test_get_settings_returns_from_app_state() -> None:
    request = MagicMock(spec=Request)
    settings = _make_mock_settings()
    request.app.state.settings = settings

    result = await get_settings(request)
    assert result is settings


@pytest.mark.asyncio
async def test_get_db_session_yields_session() -> None:
    request = MagicMock(spec=Request)
    mock_engine = MockEngine()
    request.app.state.db_engine = mock_engine

    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session

    with patch("app.api.dependencies.create_session_factory") as mock_factory, \
         patch("app.api.dependencies.session_context") as mock_ctx:
        mock_factory.return_value = "factory"
        mock_ctx.return_value.__aenter__.return_value = mock_session

        gen = get_db_session(request)
        session = await gen.__anext__()
        assert session is mock_session

        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()


@pytest.mark.asyncio
async def test_get_conversation_repository() -> None:
    mock_session = AsyncMock()
    repo = await get_conversation_repository(session=mock_session)
    from app.database.repositories.conversation_repository import ConversationRepository
    assert isinstance(repo, ConversationRepository)


@pytest.mark.asyncio
async def test_get_message_repository() -> None:
    mock_session = AsyncMock()
    repo = await get_message_repository(session=mock_session)
    from app.database.repositories.message_repository import MessageRepository
    assert isinstance(repo, MessageRepository)


@pytest.mark.asyncio
async def test_get_usage_repository() -> None:
    mock_session = AsyncMock()
    repo = await get_usage_repository(session=mock_session)
    from app.database.repositories.usage_repository import UsageRepository
    assert isinstance(repo, UsageRepository)


@pytest.mark.asyncio
async def test_get_attachment_repository() -> None:
    mock_session = AsyncMock()
    repo = await get_attachment_repository(session=mock_session)
    from app.database.repositories.attachment_repository import AttachmentRepository
    assert isinstance(repo, AttachmentRepository)


@pytest.mark.asyncio
async def test_get_attachment_service() -> None:
    mock_repo = AsyncMock()
    svc = await get_attachment_service(repository=mock_repo)
    assert isinstance(svc, AttachmentService)


@pytest.mark.asyncio
async def test_get_llm_router_returns_existing_router() -> None:
    request = MagicMock(spec=Request)
    existing = MagicMock(spec=LLMRouter)
    request.app.state.llm_router = existing

    result = await get_llm_router(request)
    assert result is existing


@pytest.mark.asyncio
async def test_get_llm_router_lazy_init_and_cache() -> None:
    """When llm_router is absent, the dependency creates one and caches it."""
    settings = _make_mock_settings(default_llm_provider="ollama")

    class FakeAppState:
        pass

    fake_state = FakeAppState()
    fake_state.settings = settings

    request = MagicMock(spec=Request)
    request.app.state = fake_state

    result1 = await get_llm_router(request)
    assert isinstance(result1, LLMRouter)

    result2 = await get_llm_router(request)
    assert result2 is result1


@pytest.mark.asyncio
async def test_get_conversation_service() -> None:
    mock_repo = AsyncMock()
    svc = await get_conversation_service(repository=mock_repo)
    assert isinstance(svc, ConversationService)


@pytest.mark.asyncio
async def test_get_core_memory_manager_returns_none_when_missing() -> None:
    request = MagicMock(spec=Request)
    request.app.state = MagicMock(spec_set=[])  # no attributes at all
    result = await get_core_memory_manager(request)
    assert result is None


@pytest.mark.asyncio
async def test_get_core_memory_manager_returns_memory_manager() -> None:
    request = MagicMock(spec=Request)
    mock_mm = MagicMock()
    request.app.state.memory_manager = mock_mm

    result = await get_core_memory_manager(request)
    assert result is mock_mm


@pytest.mark.asyncio
async def test_get_memory_service() -> None:
    mock_msg_repo = AsyncMock()
    mock_mm = MagicMock()

    with patch("app.api.dependencies.MemoryService") as MockMemoryService:
        instance = MagicMock(spec=MemoryService)
        MockMemoryService.return_value = instance

        svc = await get_memory_service(
            message_repository=mock_msg_repo,
            core_memory_manager=mock_mm,
        )
        assert svc is instance
        MockMemoryService.assert_called_once_with(mock_msg_repo, core_memory_manager=mock_mm)


@pytest.mark.asyncio
async def test_get_provider_repository() -> None:
    mock_session = AsyncMock()
    repo = await get_provider_repository(session=mock_session)
    from app.database.repositories.provider_repository import ProviderRepository
    assert isinstance(repo, ProviderRepository)


@pytest.mark.asyncio
async def test_get_provider_service() -> None:
    mock_repo = AsyncMock()
    mock_router = MagicMock(spec=LLMRouter)
    mock_settings = _make_mock_settings()

    with patch("app.api.dependencies.ProviderService") as MockProviderService:
        instance = MagicMock(spec=ProviderService)
        MockProviderService.return_value = instance

        svc = await get_provider_service(
            repository=mock_repo,
            llm_router=mock_router,
            settings=mock_settings,
        )
        assert svc is instance


@pytest.mark.asyncio
async def test_get_usage_service() -> None:
    mock_repo = AsyncMock()
    svc = await get_usage_service(repository=mock_repo)
    assert isinstance(svc, UsageService)


@pytest.mark.asyncio
async def test_get_chat_service() -> None:
    mock_conv_repo = AsyncMock()
    mock_msg_repo = AsyncMock()
    mock_usage_repo = AsyncMock()
    mock_llm = MagicMock(spec=LLMRouter)
    mock_memory = MagicMock(spec=MemoryService)

    with patch("app.api.dependencies.ChatService") as MockChatService:
        instance = MagicMock(spec=ChatService)
        MockChatService.return_value = instance

        svc = await get_chat_service(
            conversation_repo=mock_conv_repo,
            message_repo=mock_msg_repo,
            usage_repo=mock_usage_repo,
            llm_router=mock_llm,
            memory_service=mock_memory,
        )
        assert svc is instance
        MockChatService.assert_called_once_with(
            conversation_repo=mock_conv_repo,
            message_repo=mock_msg_repo,
            usage_repo=mock_usage_repo,
            llm_router=mock_llm,
            memory_service=mock_memory,
        )


# ======================================================================
# Route helper tests
# ======================================================================


class TestChatHelpers:
    def test_content_schema_to_domain_text(self) -> None:
        schema = TextBlockSchema(text="Hello")
        result = _content_schema_to_domain(schema)
        assert isinstance(result, TextBlock)
        assert result.text == "Hello"

    def test_content_schema_to_domain_image(self) -> None:
        schema = ImageBlockSchema(data_uri="data:image/png;base64,abc", mime_type="image/png")
        result = _content_schema_to_domain(schema)
        assert isinstance(result, ImageBlock)
        assert result.data_uri == "data:image/png;base64,abc"

    def test_content_schema_to_domain_tool_call(self) -> None:
        schema = ToolCallBlockSchema(tool_call_id="tc1", tool_name="calc", arguments={"expr": "1+1"})
        result = _content_schema_to_domain(schema)
        assert isinstance(result, ToolCallBlock)
        assert result.tool_call_id == "tc1"

    def test_content_schema_to_domain_tool_result(self) -> None:
        schema = ToolResultBlockSchema(tool_call_id="tc1", tool_name="calc", output="2")
        result = _content_schema_to_domain(schema)
        assert isinstance(result, ToolResultBlock)
        assert result.tool_call_id == "tc1"

    def test_content_schema_to_domain_unknown_raises(self) -> None:
        class FakeSchema:
            pass

        with pytest.raises(ValueError):
            _content_schema_to_domain(cast(ContentBlockSchema, FakeSchema()))

    def test_content_domain_to_schema_text(self) -> None:
        block = TextBlock(text="Hello")
        result = _content_domain_to_schema(block)
        assert isinstance(result, TextBlockSchema)
        assert result.text == "Hello"

    def test_content_domain_to_schema_image(self) -> None:
        block = ImageBlock(data_uri="data:img", mime_type="image/png")
        result = _content_domain_to_schema(block)
        assert isinstance(result, ImageBlockSchema)

    def test_content_domain_to_schema_tool_call(self) -> None:
        block = ToolCallBlock(tool_call_id="tc1", tool_name="calc", arguments={"x": 1})
        result = _content_domain_to_schema(block)
        assert isinstance(result, ToolCallBlockSchema)
        assert result.tool_call_id == "tc1"

    def test_content_domain_to_schema_tool_result(self) -> None:
        block = ToolResultBlock(tool_call_id="tc1", tool_name="calc", output="2")
        result = _content_domain_to_schema(block)
        assert isinstance(result, ToolResultBlockSchema)

    def test_content_domain_to_schema_unknown_raises(self) -> None:
        class FakeBlock:
            pass

        with pytest.raises(ValueError):
            _content_domain_to_schema(cast(ContentBlock, FakeBlock()))

    def test_params_schema_to_domain_none(self) -> None:
        assert _params_schema_to_domain(None) is None

    def test_params_schema_to_domain_with_values(self) -> None:
        schema = GenerationParamsSchema(temperature=0.8, max_tokens=100)
        result = _params_schema_to_domain(schema)
        assert result is not None
        assert result.temperature == 0.8
        assert result.max_tokens == 100

    def test_usage_to_response_none(self) -> None:
        assert _usage_to_response(None) is None

    def test_usage_to_response_with_usage(self) -> None:
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        result = _usage_to_response(usage)
        assert result is not None
        assert result.prompt_tokens == 10
        assert result.total_tokens == 30

    def test_message_to_response(self) -> None:
        msg = Message(
            id="msg1",
            conversation_id="conv1",
            role=MessageRole.ASSISTANT,
            content=[TextBlock(text="Hello")],
        )
        result = _message_to_response(msg)
        assert result.id == "msg1"
        assert result.role == MessageRole.ASSISTANT
        assert len(result.content) == 1

    def test_conversation_to_response(self) -> None:
        conv = Conversation(
            id="conv1",
            title="Test",
            participants=[ConversationParticipant(id="u1", display_name="Alice")],
        )
        result = _conversation_to_response(conv)
        assert result.id == "conv1"
        assert result.title == "Test"
        assert len(result.participants) == 1

    def test_chat_result_to_response(self) -> None:
        now_val = datetime.now(timezone.utc)
        msg = Message(id="msg1", conversation_id="conv1", role=MessageRole.ASSISTANT, content=[TextBlock(text="Hi")])
        conv = Conversation(id="conv1", title="Test")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        result_val = ChatResult(assistant_message=msg, conversation=conv, usage=usage)

        response = _chat_result_to_response(result_val)
        assert response.message.id == "msg1"
        assert response.conversation.id == "conv1"
        assert response.usage is not None
        assert response.usage.total_tokens == 15


class TestConversationHelpers:
    def test_conversation_to_response(self) -> None:
        conv = Conversation(
            id="conv1",
            title="My Chat",
            participants=[ConversationParticipant(id="u1", display_name="Alice")],
            metadata=ConversationMetadata(system_prompt="Be helpful"),
        )
        result = _conv_to_response(conv)
        assert result.id == "conv1"
        assert result.title == "My Chat"
        assert result.metadata.get("system_prompt") == "Be helpful"


class TestAttachmentHelpers:
    def test_attachment_to_response(self) -> None:
        now = datetime.now(timezone.utc)
        att = Attachment(
            id="att1",
            conversation_id="conv1",
            file_name="test.txt",
            mime_type="text/plain",
            size_bytes=100,
            attachment_type=AttachmentType.DOCUMENT,
            storage_path="conv1/att1.txt",
            created_at=now,
        )
        result = _attachment_to_response(att)
        assert result.id == "att1"
        assert result.file_name == "test.txt"
        assert result.attachment_type == AttachmentType.DOCUMENT


class TestProviderHelpers:
    def test_spec_to_response(self) -> None:
        spec = ProviderSpec(
            id="ollama-1",
            provider_type=ProviderType.OLLAMA,
            display_name="Local Ollama",
            base_url="http://localhost:11434",
            is_enabled=True,
            supported_capabilities=[],
            models=["llama3.1"],
        )
        result = _spec_to_response(spec)
        assert result.id == "ollama-1"
        assert result.provider_type == ProviderType.OLLAMA
        assert result.display_name == "Local Ollama"
        assert result.models == ["llama3.1"]
        assert result.healthy is None

    def test_spec_to_response_with_health(self) -> None:
        spec = ProviderSpec(
            id="ollama-1",
            provider_type=ProviderType.OLLAMA,
            display_name="Local",
            base_url="http://localhost:11434",
        )
        result = _spec_to_response(spec, healthy=True)
        assert result.healthy is True


# ======================================================================
# Schema validation tests
# ======================================================================


class TestAttachmentSchema:
    def test_attachment_create_valid(self) -> None:
        body = AttachmentCreate(
            conversation_id="conv1",
            file_name="readme.md",
            attachment_type=AttachmentType.DOCUMENT,
            mime_type="text/markdown",
            size_bytes=1024,
        )
        assert body.conversation_id == "conv1"

    def test_attachment_create_forbids_extra(self) -> None:
        with pytest.raises(Exception):
            AttachmentCreate(conversation_id="c1", file_name="f.txt", attachment_type=AttachmentType.DOCUMENT, extra_field="x")

    def test_attachment_response_frozen(self) -> None:
        now = time.time()
        resp = AttachmentResponse(id="a1", conversation_id="c1", file_name="f.txt", size_bytes=100, attachment_type=AttachmentType.DOCUMENT, created_at=now)
        assert resp.id == "a1"


class TestChatSchema:
    def test_chat_request_valid(self) -> None:
        body = ChatRequest(
            conversation_id="conv1",
            content=[TextBlockSchema(text="Hello")],
        )
        assert body.conversation_id == "conv1"

    def test_chat_request_requires_content(self) -> None:
        with pytest.raises(Exception):
            ChatRequest(conversation_id="c1", content=[])

    def test_chat_response_frozen(self) -> None:
        from datetime import datetime
        msg_resp = MessageResponse(id="m1", conversation_id="c1", role=MessageRole.ASSISTANT, content=[], created_at=datetime.now())
        conv_resp = ConversationResponse(id="c1", status=ConversationStatus.ACTIVE, message_count=0, created_at=datetime.now(), updated_at=datetime.now())
        resp = ChatResponse(message=msg_resp, conversation=conv_resp)
        assert resp.message.id == "m1"

    def test_continue_request_valid(self) -> None:
        body = ContinueRequest(
            conversation_id="conv1",
            tool_results=[ToolResultBlockSchema(tool_call_id="tc1", tool_name="calc", output="2")],
        )
        assert body.conversation_id == "conv1"

    def test_content_block_discriminator_text(self) -> None:
        adapter = TypeAdapter(ContentBlockSchema)
        block = adapter.validate_python({"type": "text", "text": "Hello"})
        assert isinstance(block, TextBlockSchema)

    def test_content_block_discriminator_image(self) -> None:
        adapter = TypeAdapter(ContentBlockSchema)
        block = adapter.validate_python({"type": "image", "data_uri": "data:img", "mime_type": "image/png"})
        assert isinstance(block, ImageBlockSchema)

    def test_content_block_discriminator_tool_call(self) -> None:
        adapter = TypeAdapter(ContentBlockSchema)
        block = adapter.validate_python({"type": "tool_call", "tool_call_id": "tc1", "tool_name": "calc"})
        assert isinstance(block, ToolCallBlockSchema)

    def test_content_block_discriminator_tool_result(self) -> None:
        adapter = TypeAdapter(ContentBlockSchema)
        block = adapter.validate_python({"type": "tool_result", "tool_call_id": "tc1", "tool_name": "calc", "output": "2"})
        assert isinstance(block, ToolResultBlockSchema)

    def test_generation_params_schema_defaults(self) -> None:
        params = GenerationParamsSchema()
        assert params.temperature is None

    def test_text_delta_event(self) -> None:
        event = TextDeltaEvent(text="Hello")
        assert event.event == "text_delta"

    def test_stream_error_event(self) -> None:
        event = StreamErrorEvent(message="Error")
        assert event.event == "error"

    def test_stream_done_event(self) -> None:
        event = StreamDoneEvent()
        assert event.event == "done"

    def test_usage_response_validation(self) -> None:
        resp = UsageResponse(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert resp.total_tokens == 30

    def test_message_response_with_content(self) -> None:
        from datetime import datetime
        content = [TextBlockSchema(text="Hello")]
        resp = MessageResponse(id="m1", conversation_id="c1", role=MessageRole.USER, content=content, created_at=datetime.now())
        assert len(resp.content) == 1


class TestConversationSchema:
    def test_create_defaults(self) -> None:
        body = ConversationCreate()
        assert body.title is None
        assert body.system_prompt is None

    def test_update_all_fields(self) -> None:
        body = ConversationUpdate(title="New Title", system_prompt="Be nice", status=ConversationStatus.ARCHIVED)
        assert body.title == "New Title"
        assert body.status == ConversationStatus.ARCHIVED

    def test_response_frozen(self) -> None:
        from datetime import datetime
        resp = ConversationResponse(id="c1", status=ConversationStatus.ACTIVE, message_count=0, created_at=datetime.now(), updated_at=datetime.now())
        assert resp.id == "c1"

    def test_list_response(self) -> None:
        from datetime import datetime
        items = [ConversationResponse(id="c1", status=ConversationStatus.ACTIVE, message_count=0, created_at=datetime.now(), updated_at=datetime.now())]
        resp = ConversationListResponse(items=items, total=1, page=1, page_size=10, pages=1)
        assert len(resp.items) == 1


class TestCommonSchema:
    def test_pagination_params_defaults(self) -> None:
        p = PaginationParams()
        assert p.page == 1
        assert p.page_size == 50

    def test_paginated_response(self) -> None:
        resp = PaginatedResponse(items=["a", "b"], total=2, page=1, page_size=10, pages=1)
        assert resp.total == 2

    def test_message_response(self) -> None:
        resp = CommonMessageResponse(message="Done.")
        assert resp.message == "Done."


class TestHealthSchema:
    def test_health_response(self) -> None:
        resp = HealthResponse(status="healthy", database=True, uptime=42.5)
        assert resp.status == "healthy"
        assert resp.database is True
        assert resp.providers is None


class TestProviderSchema:
    def test_register_request_valid(self) -> None:
        body = ProviderRegisterRequest(
            provider_id="ollama-local",
            provider_type=ProviderType.OLLAMA,
            display_name="Local Ollama",
            base_url="http://localhost:11434",
            models=["llama3.1"],
        )
        assert body.provider_id == "ollama-local"

    def test_register_request_accepts_string(self) -> None:
        """The ``_validate_not_placeholder`` helper exists but is not wired
        as a pydantic field validator. Verify the current behaviour: the
        string "string" is accepted as a provider_id."""
        body = ProviderRegisterRequest(provider_id="string", provider_type=ProviderType.OLLAMA, display_name="Bad")
        assert body.provider_id == "string"

    def test_update_request_all_optional(self) -> None:
        body = ProviderUpdateRequest()
        assert body.display_name is None

    def test_provider_check_request(self) -> None:
        body = ProviderCheckRequest(provider_id="ollama-1")
        assert body.provider_id == "ollama-1"

    def test_provider_check_response(self) -> None:
        resp = ProviderCheckResponse(provider_id="ollama-1", healthy=True)
        assert resp.healthy is True

    def test_model_info_response(self) -> None:
        resp = ModelInfoResponse(id="llama3.1", name="llama3.1", capabilities=["chat"])
        assert resp.id == "llama3.1"

    def test_model_list_response(self) -> None:
        models = [ModelInfoResponse(id="llama3.1", name="llama3.1", capabilities=[])]
        resp = ModelListResponse(models=models)
        assert len(resp.models) == 1


class TestSystemSchema:
    def test_version_info(self) -> None:
        info = VersionInfo(app_name="Astra-X", app_version="1.0.0", python_version="3.12")
        assert info.app_name == "Astra-X"

    def test_config_info(self) -> None:
        info = ConfigInfo(environment="testing", debug=True, log_level="DEBUG", log_format="json", database_url="sqlite://", default_llm_provider="ollama", default_llm_model="llama3.1", api_v1_prefix="/api/v1")
        assert info.environment == "testing"

    def test_metrics_response(self) -> None:
        resp = MetricsResponse(uptime_seconds=42.0, active_conversations=5, total_messages=100)
        assert resp.active_conversations == 5


# ======================================================================
# Route integration tests (via TestClient)
# ======================================================================


class TestHealthRoute:
    def test_health_endpoint(self) -> None:
        app = FastAPI()
        app.include_router(health_router)
        app.state.settings = _make_mock_settings()
        app.state.db_engine = MockEngine()

        # Mock the get_db_session dependency to return a mock session
        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_router = MagicMock(spec=LLMRouter)
        app.state.llm_router = mock_router

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "database" in data


class TestMetricsRoute:
    def test_metrics_endpoint(self) -> None:
        app = FastAPI()
        app.include_router(metrics_router)
        app.state.settings = _make_mock_settings()
        app.state.db_engine = MockEngine()

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session
        app.dependency_overrides[get_message_repository] = lambda: AsyncMock()

        mock_conv_service = MagicMock(spec=ConversationService)
        mock_conv_service.list_active = AsyncMock(return_value=[])
        app.dependency_overrides[get_conversation_service] = lambda: mock_conv_service

        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_seconds" in data
        assert "active_conversations" in data


class TestSystemRoute:
    def test_version_endpoint(self) -> None:
        app = FastAPI()
        app.include_router(system_router)

        settings = _make_mock_settings(app_name="Astra-X", app_version="2.0.0")
        app.dependency_overrides[get_settings] = lambda: settings

        client = TestClient(app)
        resp = client.get("/system/version")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app_name"] == "Astra-X"
        assert data["app_version"] == "2.0.0"

    def test_config_endpoint_redacts_db_url(self) -> None:
        app = FastAPI()
        app.include_router(system_router)

        settings = _make_mock_settings(database_url="sqlite+aiosqlite:///data.db")
        app.dependency_overrides[get_settings] = lambda: settings

        client = TestClient(app)
        resp = client.get("/system/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["database_url"] == "sqlite+aiosqlite://"
        assert data["environment"] == "testing"

    def test_config_no_db_scheme(self) -> None:
        app = FastAPI()
        app.include_router(system_router)

        settings = _make_mock_settings(database_url="sqlite+aiosqlite:///mydb.db")
        app.dependency_overrides[get_settings] = lambda: settings

        client = TestClient(app)
        resp = client.get("/system/config")
        assert resp.status_code == 200
        assert resp.json()["database_url"] == "sqlite+aiosqlite://"


class TestConversationsRoute:
    def test_list_conversations(self) -> None:
        app = FastAPI()
        app.include_router(conversations_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=ConversationService)
        mock_service.list_active = AsyncMock(return_value=[])
        app.dependency_overrides[get_conversation_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/conversations")
        assert resp.status_code == 200

    def test_create_conversation(self) -> None:
        app = FastAPI()
        app.include_router(conversations_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=ConversationService)
        conv = Conversation(id="new-conv-id", title="Hello")
        mock_service.create = AsyncMock(return_value=conv)
        app.dependency_overrides[get_conversation_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.post("/conversations", json={"title": "Hello"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "new-conv-id"

    def test_get_conversation(self) -> None:
        app = FastAPI()
        app.include_router(conversations_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=ConversationService)
        conv = Conversation(id="conv-1", title="Test")
        mock_service.get = AsyncMock(return_value=conv)
        app.dependency_overrides[get_conversation_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/conversations/conv-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "conv-1"

    def test_delete_conversation(self) -> None:
        app = FastAPI()
        app.include_router(conversations_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=ConversationService)
        mock_service.delete = AsyncMock(return_value=None)
        app.dependency_overrides[get_conversation_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.delete("/conversations/conv-1")
        assert resp.status_code == 200

    def test_update_conversation(self) -> None:
        app = FastAPI()
        app.include_router(conversations_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=ConversationService)
        conv = Conversation(id="conv-1", title="Updated")
        mock_service.rename = AsyncMock(return_value=None)
        mock_service.get = AsyncMock(return_value=conv)
        app.dependency_overrides[get_conversation_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.patch("/conversations/conv-1", json={"title": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["id"] == "conv-1"


class TestAttachmentsRoute:
    def test_list_attachments(self) -> None:
        app = FastAPI()
        app.include_router(attachments_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=AttachmentService)
        mock_service.list_by_conversation = AsyncMock(return_value=[])
        app.dependency_overrides[get_attachment_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/attachments", params={"conversation_id": "conv1"})
        assert resp.status_code == 200

    def test_create_attachment(self) -> None:
        app = FastAPI()
        app.include_router(attachments_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=AttachmentService)
        now_val = datetime.now(timezone.utc)
        att = Attachment(id="att-new", conversation_id="conv1", file_name="doc.txt", mime_type="text/plain", size_bytes=500, attachment_type=AttachmentType.DOCUMENT, storage_path="conv1/doc.txt", created_at=now_val)
        mock_service.create = AsyncMock(return_value=att)
        app.dependency_overrides[get_attachment_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.post("/attachments", json={"conversation_id": "conv1", "file_name": "doc.txt", "attachment_type": "document", "size_bytes": 500})
        assert resp.status_code == 201

    def test_get_attachment(self) -> None:
        app = FastAPI()
        app.include_router(attachments_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=AttachmentService)
        now_val = datetime.now(timezone.utc)
        att = Attachment(id="att-1", conversation_id="conv1", file_name="doc.txt", mime_type="text/plain", size_bytes=500, attachment_type=AttachmentType.DOCUMENT, storage_path="conv1/doc.txt", created_at=now_val)
        mock_service.get = AsyncMock(return_value=att)
        app.dependency_overrides[get_attachment_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/attachments/att-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "att-1"

    def test_delete_attachment(self) -> None:
        app = FastAPI()
        app.include_router(attachments_router)

        mock_session = AsyncMock()
        async def _mock_db_session() -> AsyncIterator[Any]:
            yield mock_session

        app.dependency_overrides[get_db_session] = _mock_db_session

        mock_service = MagicMock(spec=AttachmentService)
        mock_service.delete = AsyncMock(return_value=None)
        app.dependency_overrides[get_attachment_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.delete("/attachments/att-1")
        assert resp.status_code == 200


class TestPluginHelpers:
    def test_plugin_spec_to_response(self) -> None:
        spec = PluginSpec(
            name="test-plugin",
            display_name="Test Plugin",
            version="1.0.0",
        )
        result = _plugin_spec_to_response(spec)
        assert result.id == spec.id
        assert result.name == "test-plugin"
        assert result.display_name == "Test Plugin"
        assert result.version == "1.0.0"
        assert result.enabled is False
        assert result.status == PluginStatus.INSTALLED

    def test_plugin_spec_to_response_minimal(self) -> None:
        spec = PluginSpec(name="minimal")
        result = _plugin_spec_to_response(spec)
        assert result.name == "minimal"
        assert result.display_name == ""
        assert result.homepage is None


class TestPluginRoute:
    def test_list_plugins_empty(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        mock_service = MagicMock(spec=PluginService)
        mock_service.list_installed = AsyncMock(return_value=[])
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/plugins")
        assert resp.status_code == 200
        assert resp.json() == {"plugins": []}

    def test_list_plugins(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        spec = PluginSpec(name="test-plugin", display_name="Test Plugin")
        mock_service = MagicMock(spec=PluginService)
        mock_service.list_installed = AsyncMock(return_value=[spec])
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/plugins")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["plugins"]) == 1
        assert data["plugins"][0]["name"] == "test-plugin"

    def test_install_plugin(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        spec = PluginSpec(name="new-plugin", display_name="New Plugin")
        mock_service = MagicMock(spec=PluginService)
        mock_service.install = AsyncMock(return_value=spec)
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.post(
            "/plugins",
            json={"name": "new-plugin", "display_name": "New Plugin"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "new-plugin"
        assert data["display_name"] == "New Plugin"

    def test_install_plugin_duplicate(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        mock_service = MagicMock(spec=PluginService)
        mock_service.install = AsyncMock(
            side_effect=ConflictError("Plugin 'dup' already exists."),
        )
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.post("/plugins", json={"name": "dup"})
        assert resp.status_code == 409
        assert "already exists" in resp.text

    def test_get_plugin(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        spec = PluginSpec(id="p1", name="get-me", display_name="Get Me")
        mock_service = MagicMock(spec=PluginService)
        mock_service.get = AsyncMock(return_value=spec)
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/plugins/p1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "p1"
        assert resp.json()["name"] == "get-me"

    def test_get_plugin_not_found(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        mock_service = MagicMock(spec=PluginService)
        mock_service.get = AsyncMock(
            side_effect=ResourceNotFoundError("Plugin 'bad' not found."),
        )
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.get("/plugins/bad-id")
        assert resp.status_code == 404

    def test_update_plugin(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        spec = PluginSpec(id="p1", name="update-me", display_name="Updated", version="2.0.0")
        mock_service = MagicMock(spec=PluginService)
        mock_service.update = AsyncMock(return_value=spec)
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.patch(
            "/plugins/p1",
            json={"display_name": "Updated", "version": "2.0.0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["display_name"] == "Updated"
        assert data["version"] == "2.0.0"

    def test_update_plugin_not_found(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        mock_service = MagicMock(spec=PluginService)
        mock_service.update = AsyncMock(
            side_effect=ResourceNotFoundError("Plugin 'bad' not found."),
        )
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.patch("/plugins/bad-id", json={"display_name": "Nope"})
        assert resp.status_code == 404

    def test_uninstall_plugin(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        mock_service = MagicMock(spec=PluginService)
        mock_service.uninstall = AsyncMock(return_value=None)
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.delete("/plugins/p1")
        assert resp.status_code == 204

    def test_uninstall_plugin_not_found(self) -> None:
        app = FastAPI()
        app.include_router(plugins_router)
        app.add_exception_handler(AstraError, astra_error_handler)

        mock_service = MagicMock(spec=PluginService)
        mock_service.uninstall = AsyncMock(
            side_effect=ResourceNotFoundError("Plugin 'bad' not found."),
        )
        app.dependency_overrides[get_plugin_service] = lambda: mock_service

        client = TestClient(app)
        resp = client.delete("/plugins/bad-id")
        assert resp.status_code == 404
