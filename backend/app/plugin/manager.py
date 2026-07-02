"""Plugin lifecycle manager.

The :class:`PluginManager` is the runtime owner of all installed plugins.
It is responsible for loading enabled plugins from the database at
startup, providing access to loaded plugin metadata, and co-ordinating
with the :class:`CapabilityRegistry` so that plugin-declared capabilities
are visible to the planner and tool resolution system.

Plugins integrate with the capability system by declaring capability
identifiers in their ``PluginSpec.capabilities`` field. When a plugin is
loaded, those capabilities are registered with the
:class:`CapabilityRegistry` so they can be resolved by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.database.repositories.plugin_repository import PluginRepository
    from app.domain.plugin import PluginSpec
    from app.tools.capabilities import CapabilityRegistry

logger = get_logger(__name__)


class PluginManager:
    """Manages the lifecycle and capability registration of installed plugins.

    Usage::

        manager = PluginManager(plugin_repo, capability_registry)
        await manager.load_enabled()

        for spec in manager.list_loaded():
            print(spec.name, spec.capabilities)
    """

    def __init__(
        self,
        plugin_repository: PluginRepository,
        capability_registry: CapabilityRegistry | None = None,
    ) -> None:
        self._repo = plugin_repository
        self._capability_registry = capability_registry
        self._loaded: dict[str, PluginSpec] = {}

    # -- lifecycle ---------------------------------------------------------

    async def load_enabled(self) -> int:
        """Load all enabled plugins from the database into memory.

        This is called once during application startup.

        Returns:
            The number of plugins loaded.
        """
        specs = await self._repo.list_enabled()
        self._loaded.clear()
        for spec in specs:
            self._loaded[spec.name] = spec
        self._register_capabilities()

        if specs:
            logger.info(
                "plugin_manager.loaded",
                count=len(specs),
                names=[s.name for s in specs],
            )
        else:
            logger.info("plugin_manager.no_plugins_found")
        return len(specs)

    async def reload_plugin(self, name: str) -> PluginSpec | None:
        """Reload a single plugin from the database.

        Args:
            name: The plugin name.

        Returns:
            The updated spec, or ``None`` if the plugin no longer exists.
        """
        spec = await self._repo.find_by_name(name)
        if spec is None:
            self._loaded.pop(name, None)
            self._register_capabilities()
            return None
        self._loaded[name] = spec
        self._register_capabilities()
        return spec

    def unload_all(self) -> None:
        """Clear all loaded plugins from memory."""
        self._loaded.clear()
        self._register_capabilities()
        logger.info("plugin_manager.unloaded_all")

    # -- queries -----------------------------------------------------------

    def list_loaded(self) -> list[PluginSpec]:
        """Return all plugins currently loaded in memory."""
        return list(self._loaded.values())

    def get_loaded(self, name: str) -> PluginSpec | None:
        """Look up a loaded plugin by name.

        Args:
            name: The plugin name.

        Returns:
            The plugin spec if loaded, or ``None``.
        """
        return self._loaded.get(name)

    def is_loaded(self, name: str) -> bool:
        """Return ``True`` if the plugin is currently loaded."""
        return name in self._loaded

    # -- capabilities ------------------------------------------------------

    def list_capabilities(self) -> dict[str, str]:
        """Return a mapping of capability → plugin name for all loaded plugins.

        This aggregates the ``capabilities`` field from every loaded
        :class:`PluginSpec`.  The planner uses this to resolve which plugin
        provides a given capability.

        Returns:
            A dict like ``{"search_web": "web-search-plugin", ...}``.
        """
        mapping: dict[str, str] = {}
        for name, spec in self._loaded.items():
            for cap in getattr(spec, "capabilities", []):
                mapping[cap] = name
        return mapping

    def get_plugins_for_capability(self, capability: str) -> list[PluginSpec]:
        """Return all loaded plugins that declare *capability*.

        Args:
            capability: The capability identifier.

        Returns:
            Matching plugin specs (may be empty).
        """
        return [
            spec
            for spec in self._loaded.values()
            if capability in getattr(spec, "capabilities", [])
        ]

    # -- internal ----------------------------------------------------------

    def _register_capabilities(self) -> None:
        """Re-register all plugin-declared capabilities with the
        :class:`CapabilityRegistry`.

        This is called whenever the set of loaded plugins changes (load,
        reload, unload).
        """
        if self._capability_registry is None:
            return
        # The CapabilityRegistry builds its index from the ToolRegistry.
        # Plugin-declared capability identifiers are resolved at the
        # tool level, so rebuilding the registry picks up any tools
        # that the loaded plugins may have registered.
        self._capability_registry.rebuild()
