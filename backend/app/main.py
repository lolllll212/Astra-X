"""Application entry point and composition root for Astra X.

This module assembles the FastAPI application from its low-level building
blocks and is the only module with broad knowledge of the system's
structure. It is responsible for:

1. Creating the :class:`fastapi.FastAPI` instance.
2. Attaching the :func:`app.core.lifecycle.lifespan` context manager.
3. Registering security middleware (CORS, trusted hosts).
4. Registering structured exception handlers.
5. Mounting API routers (added as routes are implemented).
6. Providing a ``__main__`` entry point for `python -m app.main`.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config.settings import get_settings
from app.core.exceptions import (
    AstraError,
    astra_error_handler,
    http_exception_handler,
    request_validation_exception_handler,
)
from app.core.lifecycle import lifespan

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
    docs_url=f"{settings.api_v1_prefix}/docs" if settings.is_development else None,
    redoc_url=f"{settings.api_v1_prefix}/redoc" if settings.is_development else None,
    openapi_url=f"{settings.api_v1_prefix}/openapi.json" if settings.is_development else None,
)

# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------

app.add_exception_handler(AstraError, astra_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Routers
#
# API routers are imported and included here as they are implemented, e.g.:
#
#     from app.api.routers import health_router, chat_router
#     app.include_router(health_router, prefix=settings.api_v1_prefix)
#     app.include_router(chat_router, prefix=settings.api_v1_prefix)
#
# ---------------------------------------------------------------------------

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
