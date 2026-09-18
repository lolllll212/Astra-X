"""Alembic migration helpers.

Provides convenience functions for running database migrations
programmatically — for example, during application startup or in
integration test fixtures.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

__all__ = [
    "make_alembic_config",
    "run_migrations",
]


def _find_root_dir() -> Path:
    """Walk up from this file's location until we find the project root.

    The project root is identified by the presence of a ``pyproject.toml``
    or ``alembic.ini`` file.

    Returns:
        Absolute path to the project root directory.
    """
    candidate = Path(__file__).resolve().parent.parent.parent
    while not (candidate / "pyproject.toml").exists():
        if candidate.parent == candidate:
            msg = "Could not locate project root (no pyproject.toml found)"
            raise RuntimeError(msg)
        candidate = candidate.parent
    return candidate


def make_alembic_config(ini_path: Path | None = None) -> Config:
    """Create an :class:`alembic.config.Config` pointing at the project's
    Alembic configuration.

    Args:
        ini_path: Path to ``alembic.ini``. Defaults to ``<project_root>/alembic.ini``.

    Returns:
        A configured Alembic ``Config`` object.
    """
    root = _find_root_dir()
    resolved = ini_path or root / "alembic.ini"
    return Config(str(resolved))


def run_migrations(alembic_config: Config | None = None) -> None:
    """Run all pending Alembic migrations up to the latest revision.

    Args:
        alembic_config: An Alembic configuration object. Created from
            the default ini location when omitted.
    """
    config = alembic_config or make_alembic_config()
    command.upgrade(config, "head")
