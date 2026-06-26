"""Provider repository."""

from __future__ import annotations

from app.database.converters.provider_mapper import (
    provider_from_model,
    provider_to_model,
)
from app.database.models.provider import ProviderModel
from app.database.repositories.base import BaseRepository
from app.domain.provider import ProviderSpec


class ProviderRepository(BaseRepository[ProviderSpec, ProviderModel]):
    """Repository for :class:`ProviderSpec` entities."""

    @property
    def _model_cls(self) -> type[ProviderModel]:
        return ProviderModel

    def _to_domain(self, model: ProviderModel) -> ProviderSpec:
        return provider_from_model(model)

    def _to_model(self, domain: ProviderSpec) -> ProviderModel:
        return provider_to_model(domain)
