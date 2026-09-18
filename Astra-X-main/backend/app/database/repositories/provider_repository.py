from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.database.converters.provider_mapper import (
    provider_from_model,
    provider_to_model,
)
from app.database.models.provider import ProviderModel
from app.database.repositories.base import BaseRepository
from app.domain.enums import ProviderType
from app.domain.provider import ProviderSpec

# These placeholder IDs may exist in the database from earlier testing
# or schema defaults. They are silently excluded from query results.
_PLACEHOLDER_IDS: frozenset[str] = frozenset({"string", "placeholder", "changeme"})


class ProviderRepository(BaseRepository[ProviderSpec, ProviderModel]):
    """Repository for :class:`ProviderSpec` entities."""

    @property
    def _model_cls(self) -> type[ProviderModel]:
        return ProviderModel

    def _to_domain(self, model: ProviderModel) -> ProviderSpec:
        return provider_from_model(model)

    def _to_model(self, domain: ProviderSpec) -> ProviderModel:
        return provider_to_model(domain)

    async def list_all(self, **filters: Any) -> list[ProviderSpec]:
        """List all providers, excluding placeholder entries.

        Args:
            **filters: Column-value pairs to filter by.

        Returns:
            A list of valid provider specs.
        """
        stmt = select(self._model_cls).where(
            self._model_cls.id.notin_(list(_PLACEHOLDER_IDS)),
        )
        for column, value in filters.items():
            col_attr = getattr(self._model_cls, column, None)
            if col_attr is not None:
                stmt = stmt.where(col_attr == value)
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def get(self, id_: str) -> ProviderSpec | None:
        """Retrieve a provider by ID.

        Returns ``None`` for placeholder IDs so callers see a 404
        rather than attempting to convert an invalid row.

        Args:
            id_: The provider identifier.

        Returns:
            The provider spec, or ``None``.
        """
        if id_.strip().lower() in _PLACEHOLDER_IDS:
            return None
        return await super().get(id_)

    async def exists(self, provider_id: str) -> bool:
        """Check whether a provider with the given ID exists.

        Placeholder IDs always return ``False``.

        Args:
            provider_id: The provider identifier to check.

        Returns:
            ``True`` if a valid row with that primary key exists.
        """
        if provider_id.strip().lower() in _PLACEHOLDER_IDS:
            return False
        model = await self._session.get(self._model_cls, provider_id)
        return model is not None

    async def find_by_type(self, provider_type: ProviderType) -> list[ProviderSpec]:
        """Find all providers of a given backend type.

        Args:
            provider_type: The provider type to filter by.

        Returns:
            Matching provider specs.
        """
        stmt = (
            select(self._model_cls)
            .where(self._model_cls.provider_type == provider_type.value)
            .where(self._model_cls.id.notin_(list(_PLACEHOLDER_IDS)))
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def list_enabled(self) -> list[ProviderSpec]:
        """List all enabled providers.

        Returns:
            Enabled provider specs.
        """
        stmt = (
            select(self._model_cls)
            .where(self._model_cls.is_enabled == True)  # noqa: E712
            .where(self._model_cls.id.notin_(list(_PLACEHOLDER_IDS)))
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]


