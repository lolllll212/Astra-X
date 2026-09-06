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

import importlib
import time
from typing import TYPE_CHECKING

from packaging.version import Version

from app.core.logging import get_logger
from app.observability import plugin_load_duration

if TYPE_CHECKING:
    from app.database.repositories.plugin_repository import PluginRepository
    from app.domain.plugin import PluginSpec
    from app.plugins.base import Plugin
    from app.services.provider_service import ProviderService
    from app.tools.capabilities import CapabilityRegistry
    from app.tools.registry import ToolRegistry

logger = get_logger(__name__)


def _check_version_compatibility(
    app_version: str,
    spec: PluginSpec,
) -> bool:
    """Check whether *spec* is compatible with *app_version*.

    Returns ``True`` if the plugin should be loaded, ``False`` if it
    should be skipped.
    """
    try:
        current = Version(app_version)
    except Exception:
        logger.warning("plugin_manager.invalid_app_version", version=app_version)
        return True  # cannot validate — load optimistically

    if spec.minimum_core_version:
        try:
            if current < Version(spec.minimum_core_version):
                logger.info(
                    "plugin_manager.skipped_below_minimum",
                    plugin=spec.name,
                    minimum=spec.minimum_core_version,
                    app=app_version,
                )
                return False
        except Exception as exc:
            logger.warning(
                "plugin_manager.invalid_minimum_version",
                plugin=spec.name,
                version=spec.minimum_core_version,
                error=str(exc),
            )

    if spec.maximum_core_version:
        try:
            if current > Version(spec.maximum_core_version):
                logger.info(
                    "plugin_manager.skipped_above_maximum",
                    plugin=spec.name,
                    maximum=spec.maximum_core_version,
                    app=app_version,
                )
                return False
        except Exception as exc:
            logger.warning(
                "plugin_manager.invalid_maximum_version",
                plugin=spec.name,
                version=spec.maximum_core_version,
                error=str(exc),
            )

    return True


class PluginManager:
    """Manages the lifecycle and capability registration of installed plugins.

    Usage::

        manager = PluginManager(plugin_repo, capability_registry, app_version="0.1.0")
        await manager.load_enabled()

        for spec in manager.list_loaded():
            print(spec.name, spec.capabilities)
    """

    def __init__(
        self,
        plugin_repository: PluginRepository,
        capability_registry: CapabilityRegistry | None = None,
        app_version: str = "0.0.0",
        tool_registry: ToolRegistry | None = None,
        provider_service: ProviderService | None = None,
    ) -> None:
        self._repo = plugin_repository
        self._capability_registry = capability_registry
        self._app_version = app_version
        self._tool_registry = tool_registry
        self._provider_service = provider_service
        self._loaded: dict[str, PluginSpec] = {}
        self._plugin_instances: dict[str, Plugin] = {}

    # -- lifecycle ---------------------------------------------------------

    async def load_enabled(self) -> int:
        """Load all enabled plugins from the database into memory.

        Each plugin is checked for core version compatibility before being
        loaded.  Incompatible plugins are skipped with a warning log.

        If a plugin declares an ``entry_point``, the corresponding Python
        class is dynamically imported and its ``on_load`` lifecycle hook
        is called.

        This is called once during application startup.

        Returns:
            The number of plugins successfully loaded.
        """
        specs = await self._repo.list_enabled()
        self._loaded.clear()
        self._plugin_instances.clear()
        loaded_count = 0
        for spec in specs:
            if not _check_version_compatibility(self._app_version, spec):
                continue
            plugin_instance = await self._load_plugin_instance(spec)
            if plugin_instance is not None:
                self._plugin_instances[spec.name] = plugin_instance
            self._loaded[spec.name] = spec
            loaded_count += 1
        self._register_capabilities()

        if loaded_count:
            logger.info(
                "plugin_manager.loaded",
                count=loaded_count,
                names=[s.name for s in self._loaded.values()],
            )
        else:
            logger.info("plugin_manager.no_plugins_loaded")
        return loaded_count

    async def reload_plugin(self, name: str) -> PluginSpec | None:
        """Reload a single plugin from the database.

        If the plugin was previously loaded its ``on_unload`` hook is
        called first.  The plugin is then checked for core version
        compatibility.  If it is incompatible it is removed from memory
        and ``None`` is returned.

        Args:
            name: The plugin name.

        Returns:
            The updated spec, or ``None`` if the plugin no longer exists
            or is incompatible with the current app version.
        """
        # Unload existing instance if present.
        old = self._plugin_instances.pop(name, None)
        if old is not None:
            try:
                await old.on_unload()
            except Exception as exc:
                logger.warning(
                    "plugin_manager.reload_unload_failed",
                    plugin=name,
                    error=str(exc),
                )

        spec = await self._repo.find_by_name(name)
        if spec is None:
            self._loaded.pop(name, None)
            self._register_capabilities()
            return None
        if not _check_version_compatibility(self._app_version, spec):
            self._loaded.pop(name, None)
            self._register_capabilities()
            return None

        plugin_instance = await self._load_plugin_instance(spec)
        if plugin_instance is not None:
            self._plugin_instances[name] = plugin_instance

        self._loaded[name] = spec
        self._register_capabilities()
        return spec

    async def unload_all(self) -> None:
        """Unload all plugins, calling each plugin's ``on_unload`` hook."""
        errors = 0
        for name, plugin in list(self._plugin_instances.items()):
            try:
                await plugin.on_unload()
            except Exception as exc:
                logger.warning(
                    "plugin_manager.unload_failed",
                    plugin=name,
                    error=str(exc),
                )
                errors += 1
        self._plugin_instances.clear()
        self._loaded.clear()
        self._register_capabilities()
        if errors:
            logger.warning(
                "plugin_manager.unloaded_all_with_errors",
                errors=errors,
            )
        else:
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
            for cap in spec.capabilities:
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
            if capability in spec.capabilities
        ]

    # -- internal ----------------------------------------------------------

    async def _load_plugin_instance(self, spec: PluginSpec) -> Plugin | None:
        """Dynamically import and activate a plugin from its ``entry_point``.

        If the spec has no ``entry_point`` the plugin is loaded in
        metadata-only mode (no Python class is instantiated).

        Args:
            spec: The plugin specification.

        Returns:
            The activated plugin instance, or ``None`` if loading failed.
        """
        if not spec.entry_point:
            return None

        _t0 = time.monotonic()
        status = "success"

        try:
            module_path, class_name = spec.entry_point.split(":", 1)
            module = importlib.import_module(module_path)
            plugin_cls = getattr(module, class_name)
        except Exception as exc:
            status = "import_failed"
            logger.warning(
                "plugin_manager.import_failed",
                plugin=spec.name,
                entry_point=spec.entry_point,
                error=str(exc),
            )
            plugin_load_duration.labels(
                plugin_name=spec.name,
                status=status,
            ).observe(time.monotonic() - _t0)
            return None

        try:
            plugin = plugin_cls()
        except Exception as exc:
            status = "instantiate_failed"
            logger.warning(
                "plugin_manager.instantiate_failed",
                plugin=spec.name,
                error=str(exc),
            )
            plugin_load_duration.labels(
                plugin_name=spec.name,
                status=status,
            ).observe(time.monotonic() - _t0)
            return None

        from app.plugins.base import (
            PluginContext as _PluginContext,
        )
        from app.plugins.base import (
            PluginRegistries as _PluginRegistries,
        )
        from app.plugins.base import (
            PluginServices as _PluginServices,
        )

        context = _PluginContext(
            registries=_PluginRegistries(
                tool=self._tool_registry,
                capability=self._capability_registry,
            ),
            services=_PluginServices(
                provider=self._provider_service,
            ),
        )
        try:
            await plugin.on_load(context)
        except Exception as exc:
            logger.warning(
                "plugin_manager.on_load_failed",
                plugin=spec.name,
                error=str(exc),
            )
            return None

        plugin_load_duration.labels(
            plugin_name=spec.name,
            status="success",
        ).observe(time.monotonic() - _t0)

        logger.info(
            "plugin_manager.activated",
            plugin=spec.name,
            class_name=class_name,
        )
        return plugin

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
