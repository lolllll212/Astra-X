# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""System information endpoints.

Provides version, configuration, and health-related metadata about
the running application.
"""

from __future__ import annotations

import sys

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.api.schemas.system import ConfigInfo, VersionInfo
from app.config.settings import Settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get(
    "/version",
    response_model=VersionInfo,
    summary="Application version",
)
async def get_version(
    settings: Settings = Depends(get_settings),
) -> VersionInfo:
    """Return the running application's version and runtime information."""
    return VersionInfo(
        app_name=settings.app_name,
        app_version=settings.app_version,
        python_version=sys.version.split()[0],
    )


@router.get(
    "/config",
    response_model=ConfigInfo,
    summary="Sanitised application configuration",
)
async def get_config(
    settings: Settings = Depends(get_settings),
) -> ConfigInfo:
    """Return the application configuration with secrets redacted.

    The database URL is truncated to show only the scheme (e.g.
    ``sqlite+aiosqlite://``) so that connection strings with embedded
    credentials are never exposed.
    """
    db_scheme = settings.database_url.split("://")[0] + "://" if "://" in settings.database_url else settings.database_url
    return ConfigInfo(
        environment=settings.environment.value,
        debug=settings.debug,
        log_level=settings.log_level.value,
        log_format=settings.log_format.value,
        database_url=db_scheme,
        default_llm_provider=settings.default_llm_provider,
        default_llm_model=settings.default_llm_model,
        api_v1_prefix=settings.api_v1_prefix,
    )
