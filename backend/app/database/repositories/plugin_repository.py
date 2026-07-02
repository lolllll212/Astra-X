"""Plugin repository.

Provides CRUD operations for :class:`PluginSpec` entities, plus
specialised lookups needed by the plugin lifecycle system.
"""

from __future__ import annotations

from sqlalchemy import select, update

from app.database.converters.plugin_mapper import (
    plugin_from_model,
    plugin_to_model,
)
from app.database.models.plugin import PluginModel
from app.database.repositories.base import BaseRepository
from app.domain.plugin import PluginSpec


class PluginRepository(BaseRepository[PluginSpec, PluginModel]):
    """Repository for :class:`PluginSpec` entities."""

    @property
    def _model_cls(self) -> type[PluginModel]:
        return PluginModel

    def _to_domain(self, model: PluginModel) -> PluginSpec:
        return plugin_from_model(model)

    def _to_model(self, domain: PluginSpec) -> PluginModel:
        return plugin_to_model(domain)

    async def find_by_name(self, name: str) -> PluginSpec | None:
        """Look up a plugin by its unique name.

        Args:
            name: The plugin name (e.g. ``"weather-tools"``).

        Returns:
            The matching plugin spec, or ``None``.
        """
        stmt = select(self._model_cls).where(self._model_cls.name == name)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_domain(model)

    async def list_enabled(self) -> list[PluginSpec]:
        """Return all plugins that are marked as enabled.

        Returns:
            The list of enabled plugin specs.
        """
        stmt = (
            select(self._model_cls)
            .where(self._model_cls.enabled == True)
            .order_by(self._model_cls.name)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def update_status(self, plugin_id: str, status: str) -> PluginSpec | None:
        """Set the lifecycle status of a plugin.

        Args:
            plugin_id: The plugin UUID.
            status: One of the :class:`PluginStatus` values.

        Returns:
            The updated plugin spec, or ``None`` if the plugin was not found.
        """
        stmt = (
            update(self._model_cls)
            .where(self._model_cls.id == plugin_id)
            .values(status=status)
        )
        result = await self._session.execute(stmt)
        if result.rowcount == 0:
            return None
        return await self.get(plugin_id)

    async def upsert(self, domain: PluginSpec) -> PluginSpec:
        """Insert or update a plugin by name.

        If a plugin with the same ``name`` already exists, its fields are
        updated (excluding ``id`` and ``created_at``). Otherwise a new
        plugin is inserted.

        Args:
            domain: The plugin spec to persist.

        Returns:
            The inserted or updated plugin spec.
        """
        existing = await self.find_by_name(domain.name)
        if existing is not None:
            updated = existing.model_copy(
                update=domain.model_dump(exclude={"id", "created_at"}),
            )
            return await self.update(updated)
        return await self.add(domain)
