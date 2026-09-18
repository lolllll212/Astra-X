"""SQLAlchemy declarative base for all ORM models.

Every database model in ``app.database.models`` inherits from
:class:`Base`, which binds the shared :class:`MetaData` instance and
enables async relationship loading via :class:`AsyncAttrs`.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase

from app.database.metadata import metadata

__all__ = ["Base"]


class Base(AsyncAttrs, DeclarativeBase):
    """Declarative base for all Astra X ORM models.

    Combines :class:`AsyncAttrs` (enables ``await instance.awaitable_attrs.relationship``)
    with :class:`DeclarativeBase` for the standard SQLAlchemy 2.0
    declarative mapping.
    """

    metadata = metadata
