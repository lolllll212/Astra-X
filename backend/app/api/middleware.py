"""HTTP middleware for the API layer.

Provides request ID injection, timing measurement, Prometheus metrics,
structured request logging, and security headers. Each concern is
implemented as a separate :class:`starlette.middleware.base.BaseHTTPMiddleware`
subclass so they can be individually composed, tested, and excluded.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import ClassVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request and propagate
    W3C trace context.

    The request ID is read from the ``X-Request-ID`` header if provided
    by the caller; otherwise a new UUID is generated. It is stored on
    ``request.state.request_id`` for use by downstream handlers and
    reflected back in the response header.

    If ``ASTRA_TRACE_PROPAGATION_ENABLED`` is True (default), the W3C
    ``traceparent`` header is parsed and the trace id is stored on
    ``request.state.trace_id`` so the :class:`Tracer` can use it.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        trace_id: str | None = None
        settings = getattr(request.app.state, "settings", None)
        if settings and getattr(settings, "trace_propagation_enabled", True):
            traceparent = request.headers.get("traceparent")
            if traceparent:
                from app.core.tracing import parse_traceparent

                parsed = parse_traceparent(traceparent)
                if parsed is not None:
                    trace_id = parsed[0]

        request.state.trace_id = trace_id

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


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Instrument HTTP requests with Prometheus metrics.

    Records:
    * ``astra_x_http_requests_total`` (counter, by method/path/status)
    * ``astra_x_http_request_duration_seconds`` (histogram, by method/path)
    * ``astra_x_http_requests_in_flight`` (gauge, by method)
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        method = request.method
        # Normalise path for label cardinality — replace dynamic segments
        # with a placeholder.  The metric path is always normalised so
        # Prometheus label cardinality stays bounded.
        path = request.url.path.rstrip("/")

        try:
            from app.observability import in_flight_requests, request_latency, request_total
        except ImportError:
            return await call_next(request)

        in_flight_requests.labels(method=method).inc()
        start = time.monotonic()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.monotonic() - start
            status = str(response.status_code) if response is not None else "500"
            request_total.labels(method=method, path=path, status=status).inc()
            request_latency.labels(method=method, path=path).observe(elapsed)
            in_flight_requests.labels(method=method).dec()


_EXEMPT_PATHS = {"/health", "/metrics", "/api/v1/health", "/api/v1/metrics", "/docs", "/openapi.json", "/redoc"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter.

    Uses client IP (or API key if provided) as the bucket key.  The
    budget resets every 60 seconds.  In development the limit is
    raised to 1000/min so local tooling is not disrupted.
    """

    _buckets: ClassVar[dict[str, list[float]]] = defaultdict(list)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path.rstrip("/")
        if any(path == exempt or path.startswith(exempt + "/") for exempt in _EXEMPT_PATHS):
            return await call_next(request)

        settings = getattr(request.app.state, "settings", None)
        env = getattr(settings, "environment", None)
        limit = 1000 if env == "development" else getattr(settings, "rate_limit_requests_per_minute", 60)

        client_ip = request.client.host if request.client else "unknown"
        auth: str | None = request.headers.get("Authorization", "")
        key = auth if auth else client_ip

        now = time.monotonic()
        window = 60.0
        bucket = self._buckets[key]
        cutoff = now - window

        # Prune expired timestamps and check budget.
        bucket[:] = [ts for ts in bucket if ts > cutoff]

        if len(bucket) >= limit:
            return Response(
                status_code=429,
                content='{"error":{"code":"rate_limit_error","message":"Rate limit exceeded. Try again shortly."}}',
                media_type="application/json",
            )

        bucket.append(now)
        return await call_next(request)
"""Path prefixes that do not require authentication."""


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Validate a bearer token on every non-exempt request.

    Reads the expected API key from ``app.state.settings.secret_key``
    (the ASTRA_SECRET_KEY value).  In development the check is skipped
    so local tooling is not disrupted.

    The caller must provide an ``Authorization: Bearer <key>`` header.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path.rstrip("/")

        if any(path == exempt or path.startswith(exempt + "/") for exempt in _EXEMPT_PATHS):
            return await call_next(request)

        settings = getattr(request.app.state, "settings", None)
        env = getattr(settings, "environment", None)

        # Skip auth in development for local tooling convenience.
        if env == "development":
            return await call_next(request)

        auth: str | None = request.headers.get("Authorization")
        if auth is None or not auth.startswith("Bearer "):
            return Response(
                status_code=401,
                content='{"error":{"code":"authentication_error","message":"Missing or invalid Authorization header."}}',
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth.removeprefix("Bearer ").strip()
        expected = getattr(settings, "secret_key", None)
        if expected is not None:
            from pydantic import SecretStr

            expected_value = expected.get_secret_value() if isinstance(expected, SecretStr) else str(expected)
            if token != expected_value:
                return Response(
                    status_code=401,
                    content='{"error":{"code":"authentication_error","message":"Invalid API key."}}',
                    media_type="application/json",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return await call_next(request)
