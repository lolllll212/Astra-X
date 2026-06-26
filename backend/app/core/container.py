"""Dependency injection container for Astra X.

This module defines :class:`Container` — a
:class:`dependency_injector.containers.DeclarativeContainer` subclass that
wires together every application component.

All dependency resolution flows through this container; no module in the
application should instantiate its own dependencies directly. This keeps
the system testable (dependencies can be overridden per test case) and
enforces the single-responsibility principle at the composition root.

Usage::

    container = Container()
    container.config.from_pydantic(settings)
    container.wire(modules=["app.api.routes", "app.services.chat"])

    # Resolve a dependency:
    chat_service = container.chat_service()

When new components are added to the application, their providers are
added here rather than coupled to the component's constructor.
"""

from __future__ import annotations

from dependency_injector import containers, providers

from app.config.settings import Settings, get_settings

__all__ = [
    "Container",
]


class Container(containers.DeclarativeContainer):
    """Central dependency injection container for Astra X.

    All application-level providers — configuration, database, repositories,
    services, LLM provider adapters, memory stores, agent registries, tool
    registries, schedulers, and metrics collectors — are declared as class
    attributes on this container.

    Providers that are not yet implemented are simply absent; they are added
    as their corresponding modules are built, keeping this file in sync with
    the actual codebase without accumulating dead or placeholder code.

    Wiring order
    ------------
    The container is constructed and wired during application startup by
    :func:`app.core.lifecycle.lifespan`. The lifecycle is responsible for:

    1. Loading and validating :class:`app.config.settings.Settings`.
    2. Calling ``container.config.from_pydantic(settings)`` to populate
       the configuration provider.
    3. Calling ``container.wire(modules=...)`` with the list of modules
       that use dependency injection.

    Attributes:
        config: Configuration provider populated from pydantic-settings.
            Individual values are accessed as ``container.config.KEY``
            and support dot-separated lookup for nested structures.
        settings: The validated application settings singleton.
    """

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    config: providers.Configuration = providers.Configuration()

    settings: providers.Singleton[Settings] = providers.Singleton(
        get_settings,
    )

    # Future providers for database, repositories, services, LLM provider
    # adapters, memory stores, agent/tool registries, schedulers, and
    # metrics collectors are added as their corresponding modules are
    # implemented.
