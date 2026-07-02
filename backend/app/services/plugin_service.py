"""Plugin management service.

Handles CRUD operations for installed plugins and keeps the in-memory
plugin registry in sync.
"""

from __future__ import annotations

from app.core.exceptions import ConflictError, ResourceNotFoundError
from app.core.logging import get_logger
from app.database.repositories.plugin_repository import PluginRepository
from app.domain.plugin import PluginSpec, PluginStatus

logger = get_logger(__name__)


class PluginService:
    """Manages plugin lifecycle and metadata."""

    def __init__(self, repository: PluginRepository) -> None:
        self._repo = repository

    async def install(
        self,
        name: str,
        display_name: str = "",
        version: str = "0.1.0",
        sdk_version: str = ">=0.1.0",
        description: str = "",
        author: str = "",
        homepage: str | None = None,
        entry_point: str | None = None,
        manifest_path: str | None = None,
        install_path: str | None = None,
        enabled: bool = False,
        trust_tier: str = "community",
    ) -> PluginSpec:
        """Register a new plugin in the database.

        Args:
            name: Unique plugin identifier.
            display_name: Human-readable label.
            version: Semver version string.
            sdk_version: SDK ABI compatibility specifier.
            description: Free-text summary.
            author: Plugin author.
            homepage: URL to documentation or source.
            entry_point: Python entry point.
            manifest_path: Path to plugin metadata file.
            install_path: Filesystem path to plugin root.
            enabled: Whether the plugin is active at runtime.
            trust_tier: Operator-assigned trust level.

        Returns:
            The newly created plugin spec.

        Raises:
            ConflictError: If a plugin with the same name already exists.
            ValueError: If any field fails validation.
        """
        existing = await self._repo.find_by_name(name)
        if existing is not None:
            raise ConflictError(
                message=f"Plugin '{name}' already exists.",
                details={"plugin_name": name},
            )

        spec = PluginSpec(
            name=name,
            display_name=display_name,
            version=version,
            sdk_version=sdk_version,
            description=description,
            author=author,
            homepage=homepage,
            entry_point=entry_point,
            manifest_path=manifest_path,
            install_path=install_path,
            enabled=enabled,
            trust_tier=trust_tier,
        )
        result = await self._repo.add(spec)

        logger.info(
            "plugin.installed",
            plugin_name=name,
            version=version,
        )
        return result

    async def uninstall(self, plugin_id: str) -> None:
        """Remove a plugin and all of its associated data.

        Args:
            plugin_id: The plugin UUID.

        Raises:
            ResourceNotFoundError: If the plugin does not exist.
        """
        deleted = await self._repo.delete(plugin_id)
        if not deleted:
            raise ResourceNotFoundError(
                message=f"Plugin '{plugin_id}' not found.",
            )
        logger.info("plugin.uninstalled", plugin_id=plugin_id)

    async def get(self, plugin_id: str) -> PluginSpec:
        """Retrieve a plugin by its UUID.

        Args:
            plugin_id: The plugin UUID.

        Returns:
            The plugin spec.

        Raises:
            ResourceNotFoundError: If the plugin does not exist.
        """
        spec = await self._repo.get(plugin_id)
        if spec is None:
            raise ResourceNotFoundError(
                message=f"Plugin '{plugin_id}' not found.",
            )
        return spec

    async def get_by_name(self, name: str) -> PluginSpec:
        """Retrieve a plugin by its unique name.

        Args:
            name: The plugin name.

        Returns:
            The plugin spec.

        Raises:
            ResourceNotFoundError: If the plugin does not exist.
        """
        spec = await self._repo.find_by_name(name)
        if spec is None:
            raise ResourceNotFoundError(
                message=f"Plugin '{name}' not found.",
            )
        return spec

    async def list_installed(self) -> list[PluginSpec]:
        """List all installed plugins.

        Returns:
            Every plugin in the database.
        """
        return await self._repo.list_all()

    async def list_enabled(self) -> list[PluginSpec]:
        """List all enabled plugins.

        Returns:
            Enabled plugin specs.
        """
        return await self._repo.list_enabled()

    async def update(
        self,
        plugin_id: str,
        display_name: str | None = None,
        version: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        status: str | None = None,
        status_message: str | None = None,
        trust_tier: str | None = None,
    ) -> PluginSpec:
        """Update selected fields of an existing plugin.

        Only the provided fields are modified.

        Args:
            plugin_id: The plugin UUID.
            display_name: New display name.
            version: New version string.
            description: New description.
            enabled: Whether the plugin is active.
            status: New lifecycle status.
            status_message: New status message.
            trust_tier: New trust tier.

        Returns:
            The updated plugin spec.

        Raises:
            ResourceNotFoundError: If the plugin does not exist.
        """
        spec = await self._repo.get(plugin_id)
        if spec is None:
            raise ResourceNotFoundError(
                message=f"Plugin '{plugin_id}' not found.",
            )

        update_kwargs: dict[str, object] = {}
        if display_name is not None:
            update_kwargs["display_name"] = display_name
        if version is not None:
            update_kwargs["version"] = version
        if description is not None:
            update_kwargs["description"] = description
        if enabled is not None:
            update_kwargs["enabled"] = enabled
        if status is not None:
            update_kwargs["status"] = status
        if status_message is not None:
            update_kwargs["status_message"] = status_message
        if trust_tier is not None:
            update_kwargs["trust_tier"] = trust_tier

        updated = spec.model_copy(update=update_kwargs)
        result = await self._repo.update(updated)

        logger.info(
            "plugin.updated",
            plugin_id=plugin_id,
            updated_fields=list(update_kwargs.keys()),
        )
        return result

    async def activate(self, plugin_id: str) -> PluginSpec:
        """Set a plugin's status to *active* and enable it.

        This is a convenience wrapper around :meth:`update`.
        """
        return await self.update(
            plugin_id,
            enabled=True,
            status=PluginStatus.ACTIVE,
        )

    async def deactivate(self, plugin_id: str) -> PluginSpec:
        """Set a plugin's status to *disabled*.

        This is a convenience wrapper around :meth:`update`.
        """
        return await self.update(
            plugin_id,
            enabled=False,
            status=PluginStatus.DISABLED,
        )
