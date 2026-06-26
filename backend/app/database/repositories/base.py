"""Abstract base repository with common CRUD operations.

Provides a generic :class:`BaseRepository` that concrete repositories
extend. Each concrete repository supplies its own model class and
converter functions, inheriting the standard ``add``, ``get``, ``list``,
``update``, and ``delete`` operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.base import Base


class BaseRepository[DomainT, ModelT: Base](ABC):
    """Generic repository providing standard CRUD against a single table.

    Type Parameters:
        DomainT: The domain entity type.
        ModelT: The SQLAlchemy ORM model type (must subclass ``Base``).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    @abstractmethod
    def _model_cls(self) -> type[ModelT]:
        """Return the ORM model class this repository manages."""
        ...

    @abstractmethod
    def _to_domain(self, model: ModelT) -> DomainT:
        """Convert an ORM model instance to a domain entity."""
        ...

    @abstractmethod
    def _to_model(self, domain: DomainT) -> ModelT:
        """Convert a domain entity to an ORM model instance."""
        ...

    async def add(self, domain: DomainT) -> DomainT:
        """Persist a new domain entity.

        Args:
            domain: The domain entity to persist.

        Returns:
            The persisted domain entity (with any auto-generated fields).
        """
        model = self._to_model(domain)
        self._session.add(model)
        await self._session.flush()
        return self._to_domain(model)

    async def get(self, id_: str) -> DomainT | None:
        """Retrieve a domain entity by its primary key.

        Args:
            id_: The primary key value.

        Returns:
            The domain entity if found, otherwise ``None``.
        """
        model = await self._session.get(self._model_cls, id_)
        if model is None:
            return None
        return self._to_domain(model)

    async def list(self, **filters: Any) -> list[DomainT]:
        """Retrieve all entities matching the given filters.

        Keyword arguments are passed as equality filters on the ORM
        model columns.

        Args:
            **filters: Column-value pairs to filter by.

        Returns:
            A list of matching domain entities (empty if none found).
        """
        stmt = select(self._model_cls)
        for column, value in filters.items():
            col_attr = getattr(self._model_cls, column, None)
            if col_attr is not None:
                stmt = stmt.where(col_attr == value)
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def update(self, domain: DomainT) -> DomainT:
        """Update an existing entity by merging the domain state.

        Args:
            domain: The domain entity with updated values.

        Returns:
            The updated domain entity.
        """
        model = self._to_model(domain)
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_domain(merged)

    async def delete(self, id_: str) -> bool:
        """Delete an entity by its primary key.

        Args:
            id_: The primary key value.

        Returns:
            ``True`` if a row was deleted, ``False`` if the ID did not exist.
        """
        model = await self._session.get(self._model_cls, id_)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def count(self, **filters: Any) -> int:
        """Count entities matching the given filters.

        Args:
            **filters: Column-value pairs to filter by.

        Returns:
            The number of matching rows.
        """
        stmt = select(func.count()).select_from(self._model_cls)
        for column, value in filters.items():
            col_attr = getattr(self._model_cls, column, None)
            if col_attr is not None:
                stmt = stmt.where(col_attr == value)
        result = await self._session.execute(stmt)
        return result.scalar_one()
