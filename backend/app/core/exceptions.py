
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ErrorContext:
    """
    Immutable structured context attached to exceptions.

    Attributes:
        error_code: Stable machine-readable error identifier.
        message: Human-readable error message safe for internal use.
        http_status: HTTP status code associated with the error.
        details: Optional structured metadata for diagnostics and clients.
        cause: Optional original exception for internal tracing.
    """

    error_code: str
    message: str
    http_status: int
    details: dict[str, Any] | None = None
    cause: Exception | None = None


class ErrorResponse(BaseModel):
    """
    Standard API error response payload.

    Attributes:
        error: Top-level error envelope.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    error: ErrorPayload


class ErrorPayload(BaseModel):
    """
    Structured error payload returned to API clients.

    Attributes:
        code: Stable machine-readable error code.
        message: Safe human-readable message.
        details: Optional structured error details.
        request_id: Request correlation identifier when available.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured error details.",
    )
    request_id: str | None = Field(
        default=None,
        description="Request correlation identifier.",
    )


class AstraError(Exception):
    """
    Base application exception for Astra X.

    All typed application exceptions should inherit from this class.
    Subclasses may override class defaults for `default_message`,
    `default_error_code`, and `default_http_status`.

    Args:
        message: Optional human-readable message.
        error_code: Optional stable machine-readable error code.
        http_status: Optional HTTP status code.
        details: Optional structured metadata.
        cause: Optional original exception.
    """

    default_message: str = "An application error occurred."
    default_error_code: str = "application_error"
    default_http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        http_status: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        self._context = ErrorContext(
            error_code=error_code or self.default_error_code,
            message=message or self.default_message,
            http_status=http_status or self.default_http_status,
            details=details,
            cause=cause,
        )
        super().__init__(self._context.message)

    @property
    def message(self) -> str:
        """Return the safe human-readable error message."""
        return self._context.message

    @property
    def error_code(self) -> str:
        """Return the stable machine-readable error code."""
        return self._context.error_code

    @property
    def http_status(self) -> int:
        """Return the HTTP status code associated with the error."""
        return self._context.http_status

    @property
    def details(self) -> dict[str, Any] | None:
        """Return optional structured error details."""
        return self._context.details

    @property
    def cause(self) -> Exception | None:
        """Return the original underlying exception, if any."""
        return self._context.cause

    def to_response(self, request_id: str | None = None) -> ErrorResponse:
        """
        Convert the exception into the standard API error response model.

        Args:
            request_id: Optional request correlation identifier.

        Returns:
            A validated immutable error response model.
        """
        return ErrorResponse(
            error=ErrorPayload(
                code=self.error_code,
                message=self.message,
                details=self.details,
                request_id=request_id,
            ),
        )


class ConfigurationError(AstraError):
    """Raised when application configuration is invalid or unsafe."""

    default_message = "Application configuration is invalid."
    default_error_code = "configuration_error"
    default_http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class ValidationError(AstraError):
    """Raised for application-level validation failures."""

    default_message = "Request validation failed."
    default_error_code = "validation_error"
    default_http_status = status.HTTP_422_UNPROCESSABLE_ENTITY


class AuthenticationError(AstraError):
    """Raised when authentication fails or is missing."""

    default_message = "Authentication is required or invalid."
    default_error_code = "authentication_error"
    default_http_status = status.HTTP_401_UNAUTHORIZED


class AuthorizationError(AstraError):
    """Raised when an authenticated principal lacks permission."""

    default_message = "You do not have permission to perform this action."
    default_error_code = "authorization_error"
    default_http_status = status.HTTP_403_FORBIDDEN


class ResourceNotFoundError(AstraError):
    """Raised when a requested resource does not exist."""

    default_message = "The requested resource was not found."
    default_error_code = "resource_not_found"
    default_http_status = status.HTTP_404_NOT_FOUND


class ConflictError(AstraError):
    """Raised when a request conflicts with current resource state."""

    default_message = "The request conflicts with the current resource state."
    default_error_code = "conflict_error"
    default_http_status = status.HTTP_409_CONFLICT


class ProviderError(AstraError):
    """Raised for provider interaction failures."""

    default_message = "The AI provider request failed."
    default_error_code = "provider_error"
    default_http_status = status.HTTP_502_BAD_GATEWAY


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is unavailable or unreachable."""

    default_message = "The AI provider is currently unavailable."
    default_error_code = "provider_unavailable"
    default_http_status = status.HTTP_503_SERVICE_UNAVAILABLE


class ModelNotFoundError(ProviderError):
    """Raised when a requested model does not exist at the provider."""

    default_message = "The requested model was not found."
    default_error_code = "model_not_found"
    default_http_status = status.HTTP_404_NOT_FOUND


class DatabaseError(AstraError):
    """Raised for database-level failures."""

    default_message = "A database error occurred."
    default_error_code = "database_error"
    default_http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class RepositoryError(AstraError):
    """Raised for repository operation failures."""

    default_message = "A repository operation failed."
    default_error_code = "repository_error"
    default_http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class ToolExecutionError(AstraError):
    """Raised when a tool execution fails."""

    default_message = "Tool execution failed."
    default_error_code = "tool_execution_error"
    default_http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class MemoryError(AstraError):
    """Raised for memory subsystem failures."""

    default_message = "A memory subsystem error occurred."
    default_error_code = "memory_error"
    default_http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class AgentError(AstraError):
    """Raised for agent orchestration or reasoning failures."""

    default_message = "An agent execution error occurred."
    default_error_code = "agent_error"
    default_http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


class RateLimitError(AstraError):
    """Raised when a rate limit has been exceeded."""

    default_message = "Rate limit exceeded."
    default_error_code = "rate_limit_error"
    default_http_status = status.HTTP_429_TOO_MANY_REQUESTS


class InternalServerError(AstraError):
    """Raised for unexpected internal server failures."""

    default_message = "An internal server error occurred."
    default_error_code = "internal_server_error"
    default_http_status = status.HTTP_500_INTERNAL_SERVER_ERROR


def _get_request_id(request: Request) -> str | None:
    """
    Extract a request ID from request state or headers.

    Resolution order:
    1. request.state.request_id
    2. X-Request-ID header

    Args:
        request: Incoming FastAPI request.

    Returns:
        The request identifier if available, otherwise None.
    """
    state_request_id = getattr(request.state, "request_id", None)
    if isinstance(state_request_id, str) and state_request_id:
        return state_request_id

    header_request_id = request.headers.get("X-Request-ID")
    if header_request_id:
        return header_request_id

    return None


def _build_error_response(
    error: AstraError,
    request_id: str | None,
) -> ORJSONResponse:
    """
    Build an ORJSON HTTP response from an application exception.

    Args:
        error: Typed application exception.
        request_id: Optional request correlation identifier.

    Returns:
        Serialized JSON response with the appropriate HTTP status.
    """
    payload = error.to_response(request_id=request_id)
    return ORJSONResponse(
        status_code=error.http_status,
        content=payload.model_dump(mode="json", exclude_none=True),
    )


def _log_astra_error(request: Request, error: AstraError, request_id: str | None) -> None:
    """
    Log a typed application exception with structured context.

    Args:
        request: Incoming FastAPI request.
        error: Typed application exception.
        request_id: Optional request correlation identifier.
    """
    logger.warning(
        "handled_application_exception",
        path=request.url.path,
        method=request.method,
        error_code=error.error_code,
        http_status=error.http_status,
        request_id=request_id,
        details=error.details,
        cause_type=type(error.cause).__name__ if error.cause is not None else None,
    )


def _log_unhandled_exception(
    request: Request,
    error: Exception,
    request_id: str | None,
) -> None:
    """
    Log an unhandled exception with stack trace and structured request context.

    Args:
        request: Incoming FastAPI request.
        error: Unhandled exception instance.
        request_id: Optional request correlation identifier.
    """
    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        request_id=request_id,
        exception_type=type(error).__name__,
    )


async def astra_error_handler(request: Request, exc: AstraError) -> ORJSONResponse:
    """
    Handle all typed Astra application exceptions.

    Args:
        request: Incoming FastAPI request.
        exc: Typed Astra application exception.

    Returns:
        Production-safe JSON error response.
    """
    request_id = _get_request_id(request)
    _log_astra_error(request, exc, request_id)
    return _build_error_response(exc, request_id)


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> ORJSONResponse:
    """
    Handle FastAPI request validation errors.

    Args:
        request: Incoming FastAPI request.
        exc: FastAPI validation exception.

    Returns:
        Standardized validation error response.
    """
    request_id = _get_request_id(request)
    details: dict[str, Any] = {"errors": exc.errors()}

    error = ValidationError(
        message="Request validation failed.",
        details=details,
        cause=exc,
    )
    _log_astra_error(request, error, request_id)
    return _build_error_response(error, request_id)


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> ORJSONResponse:
    """
    Handle framework HTTP exceptions and normalize them into API error responses.

    Args:
        request: Incoming FastAPI request.
        exc: Starlette/FastAPI HTTP exception.

    Returns:
        Standardized JSON error response.
    """
    request_id = _get_request_id(request)

    error: AstraError
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        error = AuthenticationError(
            message=str(exc.detail),
            cause=exc,
        )
    elif exc.status_code == status.HTTP_403_FORBIDDEN:
        error = AuthorizationError(
            message=str(exc.detail),
            cause=exc,
        )
    elif exc.status_code == status.HTTP_404_NOT_FOUND:
        error = ResourceNotFoundError(
            message=str(exc.detail),
            cause=exc,
        )
    elif exc.status_code == status.HTTP_409_CONFLICT:
        error = ConflictError(
            message=str(exc.detail),
            cause=exc,
        )
    elif exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY:
        error = ValidationError(
            message=str(exc.detail),
            cause=exc,
        )
    elif exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        error = RateLimitError(
            message=str(exc.detail),
            cause=exc,
        )
    else:
        error = InternalServerError(
            message=str(exc.detail),
            cause=exc,
        )

    _log_astra_error(request, error, request_id)
    return _build_error_response(error, request_id)
