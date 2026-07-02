"""Base class and context for plugin implementations.

Every third-party plugin subclasses :class:`Plugin` and implements the
``on_load`` / ``on_unload`` lifecycle hooks. The :class:`PluginContext`
dataclass provides access to the core services a plugin can interact with.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.provider_service import ProviderService
    from app.tools.capabilities import CapabilityRegistry
    from app.tools.registry import ToolRegistry


@dataclass
class PluginContext:
    """Services and registries provided to a plugin at load time.

    Attributes:
        tool_registry: Registry for registering/unregistering tools.
        provider_service: Service for managing LLM providers.
        capability_registry: Registry for capability-to-tool resolution.
    """

    tool_registry: ToolRegistry | None = None
    provider_service: ProviderService | None = None
    capability_registry: CapabilityRegistry | None = None


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
