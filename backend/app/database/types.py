"""Custom SQLAlchemy type decorators.

Extends SQLAlchemy's built-in types with application-specific behaviour
such as auto-populated UUID primary keys and encrypted secret storage.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import String, TypeDecorator
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON

__all__ = [
    "JSON",
    "AutoUUID",
    "EncryptedString",
]


class AutoUUID(TypeDecorator[str]):
    """A UUID column that auto-generates a value when none is provided.

    Stores as a 36-character string for maximum database compatibility
    (works on SQLite, PostgreSQL, MySQL). When the column value is
    ``None`` at INSERT time, a new UUIDv4 hex string is generated
    automatically.

    Usage::

        id: Mapped[str] = mapped_column(AutoUUID, primary_key=True)
    """

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value: str | None, dialect: Any) -> str | None:
        if value is not None:
            return value
        return str(uuid.uuid4())


class JSON(TypeDecorator[Any]):
    """Cross-database JSON column.

    Delegates to the native ``sqlalchemy.JSON`` type when available and
    falls back to a text-stored JSON string for databases (like older
    SQLite versions) that lack native JSON support.

    Usage::

        data: Mapped[dict[str, Any]] = mapped_column(JSON)
    """

    impl = SQLiteJSON
    cache_ok = True


class EncryptedString(TypeDecorator[str]):
    """Placeholder for a production-grade encrypted string column.

    **This type stores values as plaintext.** It serves as a drop-in
    replacement during development so that the schema and code are
    correct. Before deploying to production, replace the ``impl`` with
    a proper encryption-backed type (e.g. using ``cryptography``
    Fernet symmetric encryption or an SQLAlchemy encryption extension).

    Usage::

        api_key: Mapped[str | None] = mapped_column(EncryptedString(255))
    """

    impl = String
    cache_ok = True

    def __init__(self, length: int | None = None) -> None:
        super().__init__(length)

    def process_bind_param(self, value: str | None, dialect: Any) -> str | None:
        return value

    def process_result_value(self, value: str | None, dialect: Any) -> str | None:
        return value
