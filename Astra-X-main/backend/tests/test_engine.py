"""Integration tests for the database engine.

These tests require a running database — they will be skipped if the
database is unreachable (e.g. CI without a DB).  Uses the standard
test environment configuration.
"""

from __future__ import annotations

import os

import pytest

from app.config.settings import get_settings
from app.database.engine import build_engine, verify_connectivity


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("SKIP_DB_TESTS", "0") == "1",
    reason="Database integration test skipped via SKIP_DB_TESTS=1",
)
async def test_database_connectivity() -> None:
    """Verify the database engine can connect and disconnect cleanly."""
    settings = get_settings()
    engine = build_engine(settings)
    try:
        await verify_connectivity(engine)
    finally:
        await engine.dispose()
