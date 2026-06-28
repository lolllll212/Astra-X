# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Health-check endpoint.

Provides a single ``GET /health`` endpoint that reports application
health including database connectivity and LLM provider availability.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session, get_llm_router, get_settings
from app.api.errors import get_uptime
from app.api.schemas.health import HealthResponse
from app.config.settings import Settings
from app.llm.router import LLMRouter

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
)
async def health_check(
    settings: Settings = Depends(get_settings),
    db_session: AsyncSession = Depends(get_db_session),
    llm_router: LLMRouter = Depends(get_llm_router),
) -> HealthResponse:
    """Return the current health status of the application.

    Checks database connectivity with a lightweight query and LLM
    provider availability via each registered provider's health
    endpoint.
    """
    database_healthy = False
    try:
        await db_session.execute(text("SELECT 1"))
        database_healthy = True
    except Exception:
        pass

    providers_healthy: bool | None = None
    try:
        provider_status = await llm_router.check_health()
        providers_healthy = None if not provider_status else any(provider_status.values())
    except Exception:
        providers_healthy = False

    # Overall is healthy if DB works and either no providers are
    # configured or all configured providers are healthy.
    no_providers = providers_healthy is None
    overall = "healthy" if database_healthy and (no_providers or providers_healthy) else "unhealthy"

    return HealthResponse(
        status=overall,
        database=database_healthy,
        providers=providers_healthy,
        uptime=get_uptime(),
    )
