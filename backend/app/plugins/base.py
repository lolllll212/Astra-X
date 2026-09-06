"""Base class and context for plugin implementations.

Every third-party plugin subclasses :class:`Plugin` and implements the
``on_load`` / ``on_unload`` lifecycle hooks. The :class:`PluginContext`
dataclass provides access to the core services a plugin can interact with.

Context structure::

    PluginContext
    ├── registries      # shared system registries
    │   ├── tool        # ToolRegistry
    │   └── capability  # CapabilityRegistry
    ├── services        # application service layer
    │   └── provider    # ProviderService
    ├── logger          # stdlib logger (reserved for future use)
    └── config          # arbitrary plugin configuration (reserved)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.provider_service import ProviderService
    from app.tools.capabilities import CapabilityRegistry
    from app.tools.registry import ToolRegistry


@dataclass
class PluginRegistries:
    """System registries exposed to plugins.

    Attributes:
        tool: Registry for registering/unregistering callable tools.
        capability: Registry that maps capability identifiers to tools.
    """

    tool: ToolRegistry | None = None
    capability: CapabilityRegistry | None = None


@dataclass
class PluginServices:
    """Application service layer exposed to plugins.

    Attributes:
        provider: Service for managing LLM provider configurations.
    """

    provider: ProviderService | None = None


@dataclass
class PluginContext:
    """Context provided to a plugin at load time.

    Attributes:
        registries: Shared system registries (tool, capability).
        services: Application service interfaces (provider).
        logger: Standard library logger instance (reserved).
        config: Arbitrary plugin configuration dict (reserved).
    """

    registries: PluginRegistries = field(default_factory=PluginRegistries)
    services: PluginServices = field(default_factory=PluginServices)
    logger: logging.Logger | None = None
    config: dict[str, Any] | None = None


class Plugin(ABC):
    """Abstract base class for all plugins.

    Subclasses must provide ``name`` and implement the two lifecycle
    hooks ``on_load`` and ``on_unload``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin identifier matching ``PluginSpec.name``."""

    @abstractmethod
    async def on_load(self, context: PluginContext) -> None:
        """Called when the plugin is loaded by PluginManager.

        Use this hook to register tools, providers, or any other
        resources.  The *context* provides access to the core services
        managed by the application.

        Args:
            context: Available services and registries.
        """

    @abstractmethod
    async def on_unload(self) -> None:
        """Called when the plugin is unloaded by PluginManager.

        Use this hook to clean up resources, unregister tools, and
        remove providers that were registered during ``on_load``.
        """
