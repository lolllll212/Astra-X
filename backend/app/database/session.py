"""Async session factory and helpers.

Provides the :class:`AsyncSession` type alias, a factory to create a
sessionmaker bound to an engine, and an async context manager for
request-scoped sessions.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
)

__all__ = [
    "AsyncSession",
    "create_session_factory",
    "session_context",
]


def create_session_factory(engine: Any) -> async_sessionmaker[AsyncSession]:
    """Create a :class:`async_sessionmaker` bound to the given engine.

    Args:
        engine: An :class:`AsyncEngine` instance.

    Returns:
        A sessionmaker configured for the engine.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def session_context(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Async context manager providing a request-scoped session.

    The session is automatically committed on success and rolled back
    on exception.

    Args:
        session_factory: A sessionmaker to produce the session.

    Yields:
        An :class:`AsyncSession` ready for use.
    """
    session = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
