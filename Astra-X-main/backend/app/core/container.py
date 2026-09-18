"""Dependency injection container for Astra X.

This module defines :class:`Container` — a lightweight manual DI
container that replaces the (unavailable on some Windows systems)
``dependency_injector`` library.

All dependency resolution flows through this container; no module in the
application should instantiate its own dependencies directly.

Usage::

    container = Container()
    container.settings = settings

    # Resolve a dependency:
    chat_service = container.chat_service()
"""

from __future__ import annotations

from typing import Any

from app.config.settings import Settings

__all__ = [
    "Container",
]


class Container:
    """Central dependency injection container for Astra X.

    Application-level components are set as attributes on instances of
    this class.  Providers that are not yet implemented are simply absent;
    they are added as their corresponding modules are built.

    Usage::

        container = Container()
        container.settings = get_settings()
        container.init_resources()

    Attributes:
        config: A dict-like configuration store populated from
            pydantic-settings.
        settings: The validated application settings singleton.
        embedder: The configured embedding provider.
        memory_manager: The core semantic memory manager.
    """

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.settings: Settings | None = None
        self.embedder: Any = None
        self.memory_manager: Any = None

    def from_pydantic(self, settings: Settings) -> None:
        """Populate config from a pydantic-settings instance.

        Args:
            settings: The application settings object.
        """
        self.config = dict(settings)
        self.settings = settings

    def wire(self, modules: list[str] | None = None) -> None:
        """Compatibility stub — replaced ``dependency_injector.container.wire``.

        The original ``wire()`` method injected container dependencies
        into module globals.  Because ``dependency_injector`` is not
        available on this platform, wiring is a no-op; all actual
        dependency injection is handled by FastAPI's ``Depends()``
        system.

        Args:
            modules: Ignored.  Retained for API compatibility.
        """
        return

    def shutdown_resources(self) -> None:
        """Compatibility stub — replaced ``dependency_injector.shutdown_resources``.

        Performs no actual work since all short-lived resources (database
        sessions, HTTP clients) are managed per-request via FastAPI
        dependencies and the application lifespan.
        """
        return
