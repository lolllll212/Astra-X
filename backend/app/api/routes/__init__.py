"""Route module re-exports for the API layer.

Each sub-module defines a ``router`` :class:`fastapi.APIRouter` instance
that is included in the main FastAPI application.
"""

from __future__ import annotations

from app.api.routes.attachments import router as attachments_router
from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.providers import router as providers_router
from app.api.routes.system import router as system_router

__all__ = [
    "attachments_router",
    "chat_router",
    "conversations_router",
    "health_router",
    "metrics_router",
    "providers_router",
    "system_router",
]
