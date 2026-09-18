"""Shared test fixtures and configuration."""

from __future__ import annotations

import os

# Ensure we never hit a real database during unit tests.
# These must be set before any app module is imported.
os.environ.setdefault("ASTRA_DATABASE_URL", "sqlite+aiosqlite://")
os.environ.setdefault("ASTRA_ENVIRONMENT", "testing")
os.environ.setdefault("ASTRA_SECRET_KEY", "test-secret-key-for-testing-only")

import pytest

from app.config.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear the cached Settings singleton before each test.

    This ensures each test gets a fresh Settings instance with the
    correct environment variables, even if a previous test modified them.
    """
    get_settings.cache_clear()
