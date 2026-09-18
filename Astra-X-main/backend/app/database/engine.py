"""Async database engine creation and disposal.

Provides a single factory for constructing the :class:`AsyncEngine`
from application settings, and a safe disposal helper.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config.settings import Settings

__all__ = [
    "build_engine",
    "dispose_engine",
    "verify_connectivity",
]


def build_engine(settings: Settings) -> AsyncEngine:
    """Construct a configured :class:`AsyncEngine` from application settings.

    Args:
        settings: Application settings providing database URL, pool size,
            echo flag, and related configuration.

    Returns:
        A ready-to-use async engine.

    Raises:
        Exception: Propagates any SQLAlchemy engine-creation error.
    """
    is_sqlite = settings.database_url.startswith("sqlite")
    if is_sqlite:
        return create_async_engine(
            url=settings.database_url,
            echo=settings.database_echo,
        )
    return create_async_engine(
        url=settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=10,
        pool_pre_ping=True,
    )


async def verify_connectivity(engine: AsyncEngine) -> None:
    """Verify the engine can reach the database.

    Executes a lightweight ``SELECT 1`` query and rolls it back
    immediately. Raises on connection failure.

    Args:
        engine: The engine to test.

    Raises:
        Exception: Propagates any connection or query error.
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def dispose_engine(engine: AsyncEngine | None) -> None:
    """Safely dispose an engine, releasing all connections.

    Safe to call with ``None`` (no-op).

    Args:
        engine: The engine to dispose, or ``None``.
    """
    if engine is None:
        return
    await engine.dispose()


@asynccontextmanager
async def engine_lifespan(settings: Settings) -> AsyncIterator[AsyncEngine]:
    """Context manager that builds, verifies, and disposes an engine.

    Usage::

        async with engine_lifespan(settings) as engine:
            # use engine

    Args:
        settings: Application settings.

    Yields:
        A verified :class:`AsyncEngine`.
    """
    engine = build_engine(settings)
    try:
        await verify_connectivity(engine)
        yield engine
    finally:
        await dispose_engine(engine)
