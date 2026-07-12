"""Application entry point and composition root for Astra X.

This module assembles the FastAPI application from its low-level building
blocks and is the only module with broad knowledge of the system's
structure. It is responsible for:

1. Creating the :class:`fastapi.FastAPI` instance.
2. Attaching the :func:`app.core.lifecycle.lifespan` context manager.
3. Registering middleware (CORS, trusted hosts, request ID, timing,
   security headers, logging).
4. Registering structured exception handlers (Astra errors, validation
   errors, LLM errors, unhandled exceptions).
5. Mounting API routers.
6. Providing a ``__main__`` entry point for `python -m app.main`.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.errors import llm_error_handler, unhandled_error_handler
from app.api.middleware import (
    AuthenticationMiddleware,
    PrometheusMetricsMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    TimingMiddleware,
)
from app.api.routes import (
    attachments_router,
    chat_router,
    conversations_router,
    dashboard_router,
    health_router,
    metrics_router,
    plugins_router,
    providers_router,
    system_router,
)
from app.api.websocket import router as websocket_router
from app.config.settings import get_settings
from app.core.exceptions import (
    AstraError,
    astra_error_handler,
    http_exception_handler,
    request_validation_exception_handler,
)
from app.core.lifecycle import lifespan
from app.llm.exceptions import LLMError

__all__ = ["app"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

settings = get_settings()

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url=f"{settings.api_v1_prefix}/docs",
    redoc_url=f"{settings.api_v1_prefix}/redoc",
    openapi_url=f"{settings.api_v1_prefix}/openapi.json",
)

# Register API-layer lifecycle hooks directly. We don't import
# app.api.lifespan here because the module-level ``app`` (FastAPI)
# would collide with the ``app`` package that lifespan belongs to.
# Instead, the hooks are imported lazily via __import__.
__import__("app.api.lifespan")

# ---------------------------------------------------------------------------
# Middleware
#
# Starlette middleware is executed in a stack: the LAST middleware added
# is the OUTERMOST (runs first on incoming requests, last on outgoing
# responses). The order below is designed to:
#
# 1. Capture every request for logging (outermost).
# 2. Add security headers.
# 3. Measure timing including all inner middleware.
# 4. Record Prometheus metrics (latency, counters, in-flight gauge).
# 5. Compress response bodies.
# 6. Assign request IDs for correlation.
# 7. Handle CORS and host validation (innermost).
# ---------------------------------------------------------------------------

app.add_middleware(
    RequestLoggingMiddleware,
)
app.add_middleware(
    RateLimitMiddleware,
)
app.add_middleware(
    AuthenticationMiddleware,
)
app.add_middleware(
    SecurityHeadersMiddleware,
)
app.add_middleware(
    PrometheusMetricsMiddleware,
)
app.add_middleware(
    TimingMiddleware,
)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)
app.add_middleware(
    RequestIDMiddleware,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts,
)

# ---------------------------------------------------------------------------
# Exception handlers
#
# Registered in order of specificity. FastAPI dispatches to the first
# matching handler, so more specific handlers must come first.
# ---------------------------------------------------------------------------

app.add_exception_handler(AstraError, astra_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(LLMError, llm_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_error_handler)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(chat_router, prefix=settings.api_v1_prefix)
app.include_router(conversations_router, prefix=settings.api_v1_prefix)
app.include_router(dashboard_router, prefix=settings.api_v1_prefix)
app.include_router(providers_router, prefix=settings.api_v1_prefix)
app.include_router(plugins_router, prefix=settings.api_v1_prefix)
app.include_router(attachments_router, prefix=settings.api_v1_prefix)
app.include_router(metrics_router, prefix=settings.api_v1_prefix)
app.include_router(system_router, prefix=settings.api_v1_prefix)
app.include_router(websocket_router)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.value.lower(),
    )
