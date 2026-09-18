"""LLM exception to HTTP response mapping.

Registers FastAPI exception handlers for exceptions that are not part of
the :class:`app.core.exceptions.AstraError` hierarchy. Every handler
returns a standardised ORJSON error response matching the format produced
by the core exception handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from fastapi import Request
from fastapi.responses import ORJSONResponse
from starlette import status

from app.core.exceptions import AstraError, ProviderError, ProviderUnavailableError, _get_request_id
from app.core.logging import get_logger
from app.llm.exceptions import (
    GenerationError,
    LLMError,
    ModelNotSupportedError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderTimeoutError,
    RouterNoProviderError,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class _StartupState:
    """Simple holder for uptime tracking."""

    start_time: float


_startup = _StartupState(start_time=monotonic())


def _map_llm_error(exc: LLMError) -> tuple[int, str, str]:
    """Map an LLM exception to HTTP status, error code, and message.

    Args:
        exc: The LLM-layer exception.

    Returns:
        A ``(http_status, error_code, message)`` tuple.
    """
    if isinstance(exc, RouterNoProviderError):
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "provider_unavailable",
            str(exc),
        )
    if isinstance(exc, ProviderTimeoutError):
        return (
            status.HTTP_504_GATEWAY_TIMEOUT,
            "provider_timeout",
            str(exc),
        )
    if isinstance(exc, ProviderConnectionError):
        return (
            status.HTTP_502_BAD_GATEWAY,
            "provider_connection_error",
            str(exc),
        )
    if isinstance(exc, ProviderAuthenticationError):
        return (
            status.HTTP_502_BAD_GATEWAY,
            "provider_authentication_error",
            str(exc),
        )
    if isinstance(exc, ModelNotSupportedError):
        return (
            status.HTTP_400_BAD_REQUEST,
            "model_not_supported",
            str(exc),
        )
    if isinstance(exc, GenerationError):
        return (
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "generation_error",
            str(exc),
        )
    return (
        status.HTTP_502_BAD_GATEWAY,
        "llm_error",
        str(exc),
    )


async def llm_error_handler(request: Request, exc: LLMError) -> ORJSONResponse:
    """Handle any :class:`LLMError` and return a standardised JSON response.

    Args:
        request: The incoming HTTP request.
        exc: The LLM-layer exception.

    Returns:
        A structured JSON error response.
    """
    request_id = _get_request_id(request)
    http_status, error_code, message = _map_llm_error(exc)

    logger.warning(
        "handled_llm_exception",
        path=request.url.path,
        method=request.method,
        error_code=error_code,
        http_status=http_status,
        request_id=request_id,
        exception_type=type(exc).__name__,
    )

    if http_status in (status.HTTP_502_BAD_GATEWAY, status.HTTP_503_SERVICE_UNAVAILABLE, status.HTTP_504_GATEWAY_TIMEOUT):
        err: AstraError = ProviderUnavailableError(message=message, cause=exc)
    else:
        err = ProviderError(message=message, cause=exc)

    payload = err.to_response(request_id=request_id)
    return ORJSONResponse(
        status_code=http_status,
        content=payload.model_dump(mode="json", exclude_none=True),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> ORJSONResponse:
    """Catch-all handler for exceptions that escaped all other handlers.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        A generic 500 JSON error response.
    """
    request_id = _get_request_id(request)

    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        request_id=request_id,
        exception_type=type(exc).__name__,
    )

    from app.core.exceptions import InternalServerError

    error = InternalServerError(message="An unexpected error occurred.", cause=exc)
    payload = error.to_response(request_id=request_id)
    return ORJSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(mode="json", exclude_none=True),
    )


def get_uptime() -> float:
    """Return the wall-clock seconds since this module was loaded."""
    return monotonic() - _startup.start_time
