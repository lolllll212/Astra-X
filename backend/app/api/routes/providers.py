# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Provider management endpoints.

Provides CRUD operations and health checks for LLM provider
configurations stored in the database.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_llm_router, get_provider_service
from app.api.schemas.provider import (
    ModelInfoResponse,
    ModelListResponse,
    ProviderCheckRequest,
    ProviderCheckResponse,
    ProviderListResponse,
    ProviderRegisterRequest,
    ProviderResponse,
    ProviderUpdateRequest,
)
from app.domain.provider import ProviderSpec
from app.llm.router import LLMRouter
from app.services.provider_service import ProviderService

router = APIRouter(prefix="/providers", tags=["providers"])


def _spec_to_response(
    spec: ProviderSpec,
    healthy: bool | None = None,
) -> ProviderResponse:
    """Convert a domain ``ProviderSpec`` to an API ``ProviderResponse``."""
    return ProviderResponse(
        id=str(spec.id),
        provider_type=spec.provider_type,
        display_name=spec.display_name,
        base_url=spec.base_url,
        is_enabled=spec.is_enabled,
        supported_capabilities=sorted(c.value for c in spec.supported_capabilities),
        models=spec.models,
        healthy=healthy,
        created_at=spec.created_at,
        updated_at=spec.updated_at,
    )


@router.get(
    "",
    response_model=ProviderListResponse,
    summary="List registered providers",
)
async def list_providers(
    provider_service: ProviderService = Depends(get_provider_service),
) -> ProviderListResponse:
    """Return all registered LLM provider configurations."""
    specs = await provider_service.list_all()
    providers = [_spec_to_response(spec) for spec in specs]
    return ProviderListResponse(providers=providers)


@router.post(
    "",
    response_model=ProviderResponse,
    status_code=201,
    summary="Register a new provider",
)
async def register_provider(
    body: ProviderRegisterRequest,
    provider_service: ProviderService = Depends(get_provider_service),
) -> ProviderResponse:
    """Register a new LLM provider configuration.

    Rejects duplicate IDs (``409 Conflict``), placeholder values, and
    invalid URLs. The ``api_key`` is stored encrypted and never returned
    in the response.
    """
    spec = await provider_service.register(
        provider_id=body.provider_id,
        provider_type=body.provider_type,
        display_name=body.display_name,
        base_url=body.base_url,
        api_key=body.api_key,
        models=body.models,
    )
    return _spec_to_response(spec)


@router.get(
    "/{provider_id}",
    response_model=ProviderResponse,
    summary="Get provider details",
)
async def get_provider(
    provider_id: str,
    provider_service: ProviderService = Depends(get_provider_service),
) -> ProviderResponse:
    """Return details for a single provider by ID.

    Raises ``404 Not Found`` if the provider does not exist.
    """
    spec = await provider_service.get(provider_id)
    return _spec_to_response(spec)


@router.patch(
    "/{provider_id}",
    response_model=ProviderResponse,
    summary="Update provider",
)
async def update_provider(
    provider_id: str,
    body: ProviderUpdateRequest,
    provider_service: ProviderService = Depends(get_provider_service),
) -> ProviderResponse:
    """Update an existing provider. Only provided fields are changed.

    Raises ``404 Not Found`` if the provider does not exist.
    """
    spec = await provider_service.update(
        provider_id=provider_id,
        display_name=body.display_name,
        base_url=body.base_url,
        api_key=body.api_key,
        is_enabled=body.is_enabled,
        models=body.models,
    )
    return _spec_to_response(spec)


@router.delete(
    "/{provider_id}",
    status_code=204,
    summary="Delete provider",
)
async def delete_provider(
    provider_id: str,
    provider_service: ProviderService = Depends(get_provider_service),
) -> None:
    """Remove a provider registration.

    Also removes the associated adapter from the in-memory router registry.
    Raises ``404 Not Found`` if the provider does not exist.
    """
    await provider_service.remove(provider_id)


@router.post(
    "/check",
    response_model=ProviderCheckResponse,
    summary="Check provider health",
)
async def check_provider(
    body: ProviderCheckRequest,
    provider_service: ProviderService = Depends(get_provider_service),
) -> ProviderCheckResponse:
    """Check whether a specific provider is reachable and responding.

    Performs a lightweight health probe against the provider's API.
    """
    status_map = await provider_service.check_health(provider_id=body.provider_id)
    healthy = status_map.get(body.provider_id, False)
    return ProviderCheckResponse(
        provider_id=body.provider_id,
        healthy=healthy,
    )


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List all available models",
)
async def list_models(
    llm_router: LLMRouter = Depends(get_llm_router),
) -> ModelListResponse:
    """Return all models available from registered providers."""
    models: list[ModelInfoResponse] = []
    seen: set[str] = set()
    for provider in llm_router._registry.list():
        try:
            provider_models = await provider.list_models()
        except Exception:
            continue
        for model_id in provider_models:
            if model_id not in seen:
                seen.add(model_id)
                models.append(
                    ModelInfoResponse(
                        id=model_id,
                        name=model_id,
                        capabilities=[],
                    ),
                )
    return ModelListResponse(models=models)
