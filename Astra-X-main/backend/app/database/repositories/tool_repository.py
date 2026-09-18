"""Tool repository."""

from __future__ import annotations

from app.database.converters.tool_mapper import (
    tool_from_model,
    tool_to_model,
)
from app.database.models.tool import ToolModel
from app.database.repositories.base import BaseRepository
from app.domain.tool import ToolSpec


class ToolRepository(BaseRepository[ToolSpec, ToolModel]):
    """Repository for :class:`ToolSpec` entities."""

    @property
    def _model_cls(self) -> type[ToolModel]:
        return ToolModel

    def _to_domain(self, model: ToolModel) -> ToolSpec:
        return tool_from_model(model)

    def _to_model(self, domain: ToolSpec) -> ToolModel:
        return tool_to_model(domain)
