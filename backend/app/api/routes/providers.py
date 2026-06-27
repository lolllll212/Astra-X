# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Provider management endpoints.

Provides CRUD operations and health checks for LLM provider
configurations.
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
)
from app.llm.router import LLMRouter
from app.services.provider_service import ProviderService

router = APIRouter(prefix="/providers", tags=["providers"])


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
    providers = [
        ProviderResponse(
            id=spec.id,
            provider_type=spec.provider_type,
            display_name=spec.display_name,
            supported_capabilities=list(spec.supported_capabilities),
        )
        for spec in specs
    ]
    return ProviderListResponse(providers=providers)


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
    for provider in llm_router._registry.list():
        try:
            provider_models = await provider.list_models()
        except Exception:
            continue
        for model_id in provider_models:
            models.append(
                ModelInfoResponse(
                    id=model_id,
                    name=model_id,
                    capabilities=[],
                )
            )
    return ModelListResponse(models=models)


@router.post(
    "/check",
    response_model=ProviderCheckResponse,
    summary="Check provider health",
)
async def check_provider(
    body: ProviderCheckRequest,
    llm_router: LLMRouter = Depends(get_llm_router),
) -> ProviderCheckResponse:
    """Check whether a specific provider is reachable and responding."""
    status_map = await llm_router.check_health(provider_id=body.provider_id)
    healthy = status_map.get(body.provider_id, False)
    return ProviderCheckResponse(
        provider_id=body.provider_id,
        healthy=healthy,
    )


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
    """Register a new LLM provider configuration."""
    spec = await provider_service.register(
        provider_id=body.provider_id,
        provider_type=body.provider_type,
        display_name=body.display_name,
    )
    return ProviderResponse(
        id=spec.id,
        provider_type=spec.provider_type,
        display_name=spec.display_name,
        supported_capabilities=list(spec.supported_capabilities),
    )
