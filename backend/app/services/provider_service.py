"""Provider management service.

Handles CRUD operations for LLM provider configurations stored in the
database.
"""

from __future__ import annotations

from app.core.exceptions import ResourceNotFoundError
from app.core.logging import get_logger
from app.database.repositories.provider_repository import ProviderRepository
from app.domain.enums import ModelCapability, ProviderType
from app.domain.provider import ProviderID, ProviderSpec

logger = get_logger(__name__)


class ProviderService:
    """Manages provider configurations."""

    def __init__(self, repository: ProviderRepository) -> None:
        self._repo = repository

    async def register(
        self,
        provider_id: str,
        provider_type: ProviderType,
        display_name: str,
        supported_capabilities: frozenset[ModelCapability] | None = None,
    ) -> ProviderSpec:
        """Register a new provider.

        Args:
            provider_id: Unique provider identifier.
            provider_type: The provider backend type.
            display_name: Human-readable name.
            supported_capabilities: Capabilities this provider supports.

        Returns:
            The registered provider spec.
        """
        spec = ProviderSpec(
            id=ProviderID(provider_id),
            provider_type=provider_type,
            display_name=display_name,
            supported_capabilities=supported_capabilities or frozenset(),
        )
        result = await self._repo.add(spec)
        logger.info("provider.registered", provider_id=provider_id)
        return result

    async def get(self, provider_id: str) -> ProviderSpec:
        """Retrieve a provider by ID.

        Args:
            provider_id: The provider identifier.

        Returns:
            The provider spec.
        """
        spec = await self._repo.get(provider_id)
        if spec is None:
            raise ResourceNotFoundError(
                message=f"Provider '{provider_id}' not found.",
            )
        return spec

    async def list_all(self) -> list[ProviderSpec]:
        """List all registered providers.

        Returns:
            All provider specs.
        """
        return await self._repo.list()

    async def remove(self, provider_id: str) -> None:
        """Remove a provider registration.

        Args:
            provider_id: The provider identifier.
        """
        deleted = await self._repo.delete(provider_id)
        if not deleted:
            raise ResourceNotFoundError(
                message=f"Provider '{provider_id}' not found.",
            )
        logger.info("provider.removed", provider_id=provider_id)
