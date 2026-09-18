"""SQLAlchemy naming convention and shared MetaData.

Centralises the MetaData instance used by every ORM model so that
naming conventions for constraints and indexes are consistent across
all tables. Alembic autogenerate reads conventions from here.
"""

from __future__ import annotations

from sqlalchemy import MetaData

__all__ = [
    "metadata",
    "naming_convention",
]

naming_convention: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
"""Naming convention dictionary passed to :class:`MetaData`.

Every constraint and index created by SQLAlchemy or Alembic uses these
templates so that generated names are deterministic and readable.
"""

metadata: MetaData = MetaData(naming_convention=naming_convention)
"""Shared :class:`MetaData` instance for all ORM models."""
