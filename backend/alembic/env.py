"""Alembic environment configuration.

Uses an async engine via SQLAlchemy's ``run_async()`` for online migrations,
and references the application's declarative ``Base.metadata`` for autogenerate
support. This lets ``alembic revision --autogenerate`` detect model changes.

Database URL resolution (first match wins):
  1. ``-x dburl=postgresql+asyncpg://...`` command-line argument
  2. ``sqlalchemy.url`` from ``alembic.ini``
  3. ``ASTRA_DATABASE_URL`` environment variable
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from app.database.base import Base
# Ensure all ORM models are loaded so autogenerate detects their tables.
from app.database import models  # noqa: F401

# Alembic Config object
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at the application's full metadata for autogenerate
target_metadata = Base.metadata


def _resolve_db_url() -> str:
    """Resolve the database URL from multiple sources."""
    # 1. -x dburl=... command-line argument
    dburl = context.get_x_argument(as_dictionary=True).get("dburl")
    if dburl:
        return dburl

    # 2. sqlalchemy.url from alembic.ini
    url = config.get_main_option("sqlalchemy.url")
    if url and url != "driver://user:pass@localhost/dbname":
        return url

    # 3. ASTRA_DATABASE_URL environment variable
    env_url = os.environ.get("ASTRA_DATABASE_URL")
    if env_url:
        return env_url

    msg = (
        "No database URL configured. Provide one via:\n"
        "  alembic -x dburl=\"postgresql+asyncpg://user:pass@host/dbname\" <command>\n"
        "  or set sqlalchemy.url in alembic.ini\n"
        "  or set the ASTRA_DATABASE_URL environment variable"
    )
    raise RuntimeError(msg)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    url = _resolve_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: object) -> None:
    """Configure and run migrations against the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations within an async connection."""
    url = _resolve_db_url()
    connectable = create_async_engine(url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    asyncio.run(run_async_migrations())


if __name__ == "__main__":
    # Only run migrations when this script is invoked directly by the
    # ``alembic`` command. Importing this module (e.g. as a side effect of
    # importing application code) must never trigger a migration run.
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()
