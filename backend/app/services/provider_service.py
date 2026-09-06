"""Provider management service.

Handles CRUD operations for LLM provider configurations stored in the
database and keeps the in-memory router registry in sync.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from app.config.settings import Settings
from app.core.exceptions import ConflictError, ResourceNotFoundError
from app.core.logging import get_logger
from app.database.repositories.provider_repository import ProviderRepository
from app.domain.enums import ModelCapability, ProviderType
from app.domain.provider import ProviderID, ProviderSpec
from app.llm.factory import provider_type_to_default_base_url
from app.llm.router import LLMRouter

logger = get_logger(__name__)

# These placeholder values should never be accepted as real provider IDs.
_PLACEHOLDER_IDS: frozenset[str] = frozenset({"string", "placeholder", "changeme"})


def _validate_not_placeholder(value: str, field_name: str) -> None:
    """Raise ``ValueError`` if *value* is a known placeholder string.

    Args:
        value: The value to check.
        field_name: The field name, used in the error message.

    Raises:
        ValueError: If the value is a placeholder.
    """
    stripped = value.strip().lower()
    if stripped in _PLACEHOLDER_IDS:
        raise ValueError(
            f"{field_name}: '{value}' is a placeholder value and is not allowed. "
            f"Please provide a meaningful identifier.",
        )


def _validate_url(value: str | None, field_name: str) -> None:
    """Validate that a URL, if provided, uses an http(s) scheme.

    Args:
        value: The URL to validate.
        field_name: The field name, used in the error message.

    Raises:
        ValueError: If the value is a non-empty string without http(s) scheme.
    """
    if value is None:
        return
    if not value.startswith(("http://", "https://")):
        raise ValueError(
            f"{field_name} must be an absolute URL starting with 'http://' "
            f"or 'https://', got: {value!r}",
        )


class ProviderService:
    """Manages provider configurations and syncs with the router."""

    def __init__(
        self,
        repository: ProviderRepository,
        llm_router: LLMRouter,
        settings: Settings,
    ) -> None:
        self._repo = repository
        self._router = llm_router
        self._settings = settings

    async def register(
        self,
        provider_id: str,
        provider_type: ProviderType,
        display_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        models: list[str] | None = None,
        is_enabled: bool = True,
    ) -> ProviderSpec:
        """Register a new provider.

        Args:
            provider_id: Unique provider identifier.
            provider_type: The provider backend type.
            display_name: Human-readable name.
            base_url: Provider API base URL.
            api_key: API key (stored encrypted).
            models: Model identifiers the provider serves.
            is_enabled: Whether the provider is active.

        Returns:
            The registered provider spec.

        Raises:
            ConflictError: If a provider with the same ID already exists.
            ValueError: If validation fails.
        """
        normalized_id = provider_id.strip().lower()
        _validate_not_placeholder(normalized_id, "provider_id")
        _validate_url(base_url, "base_url")

        exists = await self._repo.exists(normalized_id)
        if exists:
            raise ConflictError(
                message=f"Provider '{normalized_id}' already exists.",
                details={"provider_id": normalized_id},
            )

        resolved_base_url: str | None = base_url
        if resolved_base_url is None:
            resolved_base_url = provider_type_to_default_base_url(provider_type, self._settings)

        spec = ProviderSpec(
            id=ProviderID(normalized_id),
            provider_type=provider_type,
            display_name=display_name,
            is_enabled=is_enabled,
            base_url=resolved_base_url,
            api_key=SecretStr(api_key) if api_key else None,
            supported_capabilities=frozenset({ModelCapability.CHAT, ModelCapability.STREAMING}),
            models=models or [],
        )
        result = await self._repo.add(spec)

        # Register the adapter in the router's in-memory registry.
        if is_enabled:
            self._router.register_adapter(result)

        logger.info(
            "provider.registered",
            provider_id=normalized_id,
            provider_type=provider_type.value,
        )
        return result

    async def update(
        self,
        provider_id: str,
        display_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        is_enabled: bool | None = None,
        models: list[str] | None = None,
    ) -> ProviderSpec:
        """Update an existing provider.

        Only the provided fields are updated.

        Args:
            provider_id: The provider identifier.
            display_name: New display name.
            base_url: New base URL.
            api_key: New API key.
            is_enabled: Whether the provider is active.
            models: New model list.

        Returns:
            The updated provider spec.

        Raises:
            ResourceNotFoundError: If the provider does not exist.
            ValueError: If validation fails.
        """
        spec = await self._repo.get(provider_id)
        if spec is None:
            raise ResourceNotFoundError(
                message=f"Provider '{provider_id}' not found.",
            )

        _validate_url(base_url, "base_url")

        update_kwargs: dict[str, Any] = {}
        if display_name is not None:
            update_kwargs["display_name"] = display_name
        if base_url is not None:
            update_kwargs["base_url"] = base_url
        if api_key is not None:
            update_kwargs["api_key"] = SecretStr(api_key)
        if is_enabled is not None:
            update_kwargs["is_enabled"] = is_enabled
        if models is not None:
            update_kwargs["models"] = models

        updated = spec.model_copy(update=update_kwargs)

        # The repository merge requires a mutable model; flush the update.
        result = await self._repo.update(updated)

        # Sync the router registry.
        self._router.refresh_adapter(result)

        logger.info(
            "provider.updated",
            provider_id=provider_id,
        )
        return result

    async def get(self, provider_id: str) -> ProviderSpec:
        """Retrieve a provider by ID.

        Args:
            provider_id: The provider identifier.

        Returns:
            The provider spec.

        Raises:
            ResourceNotFoundError: If the provider does not exist.
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
        return await self._repo.list_all()

    async def list_enabled(self) -> list[ProviderSpec]:
        """List all enabled providers.

        Returns:
            Enabled provider specs.
        """
        return await self._repo.list_enabled()

    async def remove(self, provider_id: str) -> None:
        """Remove a provider registration.

        Also removes the adapter from the router's in-memory registry.

        Args:
            provider_id: The provider identifier.

        Raises:
            ResourceNotFoundError: If the provider does not exist.
        """
        deleted = await self._repo.delete(provider_id)
        if not deleted:
            raise ResourceNotFoundError(
                message=f"Provider '{provider_id}' not found.",
            )
        self._router.remove_adapter(provider_id)
        logger.info("provider.removed", provider_id=provider_id)

    async def check_health(self, provider_id: str | None = None) -> dict[str, bool]:
        """Check health of one or all providers.

        Args:
            provider_id: If set, checks only this provider.

        Returns:
            A mapping of provider ID to health status.
        """
        return await self._router.check_health(provider_id=provider_id)
