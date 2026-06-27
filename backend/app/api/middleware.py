"""HTTP middleware for the API layer.

Provides request ID injection, timing measurement, structured request
logging, and security headers. Each concern is implemented as a separate
:class:`starlette.middleware.base.BaseHTTPMiddleware` subclass so they
can be individually composed, tested, and excluded if needed.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import ClassVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request.

    The request ID is read from the ``X-Request-ID`` header if provided
    by the caller; otherwise a new UUID is generated. It is stored on
    ``request.state.request_id`` for use by downstream handlers and
    reflected back in the response header.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Record request duration and expose it via response header.

    The elapsed time in milliseconds is set on the ``X-Request-Time-Ms``
    response header for debugging and client-side latency tracking.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - start) * 1000)
        response.headers["X-Request-Time-Ms"] = str(elapsed_ms)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration.

    This middleware runs outermost (last added, first executed) so that
    it captures the total request lifecycle including all other
    middleware processing.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - start) * 1000)
        logger.info(
            "api.request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            request_id=getattr(request.state, "request_id", None),
        )
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply security-related HTTP headers to every response.

    Headers applied:
    * ``X-Content-Type-Options: nosniff``
    * ``X-Frame-Options: DENY``
    * ``X-XSS-Protection: 1; mode=block``
    """

    _SECURITY_HEADERS: ClassVar[dict[str, str]] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
    }

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        for header, value in self._SECURITY_HEADERS.items():
            response.headers[header] = value
        return response
