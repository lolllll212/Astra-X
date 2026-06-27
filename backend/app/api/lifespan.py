"""API-layer lifecycle extensions.

Optional startup/shutdown hooks that are specific to the API layer
(e.g. initialising provider adapters, seeding default data). These are
registered with :func:`app.core.lifecycle.register_startup_hook` and
:func:`app.core.lifecycle.register_shutdown_hook` at import time.
"""

from __future__ import annotations

from app.core.container import Container
from app.core.lifecycle import register_startup_hook
from app.core.logging import get_logger

logger = get_logger(__name__)


async def _seed_defaults(container: Container) -> None:
    """Seed default data if the database is empty.

    Runs during startup to ensure essential records (e.g. default
    providers, system tools) exist without requiring manual migration
    steps.

    Args:
        container: The fully-initialised DI container.
    """
    logger.info("api.seeding_defaults")


@register_startup_hook
async def on_startup(container: Container) -> None:
    """API-layer startup initialisation.

    Args:
        container: The fully-initialised DI container.
    """
    await _seed_defaults(container)
