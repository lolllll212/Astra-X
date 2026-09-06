# ruff: noqa: B008 — FastAPI Depends() in default args is intentional

"""Plugin management endpoints.

Provides CRUD operations for installed plugin configurations stored in
the database.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_plugin_service
from app.api.schemas.plugin import (
    PluginInstallRequest,
    PluginListResponse,
    PluginResponse,
    PluginUpdateRequest,
)
from app.domain.plugin import PluginSpec
from app.services.plugin_service import PluginService

router = APIRouter(prefix="/plugins", tags=["plugins"])


def _spec_to_response(spec: PluginSpec) -> PluginResponse:
    """Convert a domain ``PluginSpec`` to an API ``PluginResponse``."""
    return PluginResponse(
        id=spec.id,
        name=spec.name,
        display_name=spec.display_name,
        version=spec.version,
        sdk_version=spec.sdk_version,
        description=spec.description,
        author=spec.author,
        homepage=spec.homepage,
        license_=spec.license_,
        trust_tier=spec.trust_tier,
        enabled=spec.enabled,
        install_type=spec.install_type,
        entry_point=spec.entry_point,
        manifest_path=spec.manifest_path,
        install_path=spec.install_path,
        capabilities=spec.capabilities,
        minimum_core_version=spec.minimum_core_version,
        maximum_core_version=spec.maximum_core_version,
        status=spec.status,
        status_message=spec.status_message,
        checksum=spec.checksum,
        created_at=spec.created_at,
        updated_at=spec.updated_at,
    )


@router.get(
    "",
    response_model=PluginListResponse,
    summary="List installed plugins",
)
async def list_plugins(
    plugin_service: PluginService = Depends(get_plugin_service),
) -> PluginListResponse:
    """Return all installed plugins."""
    specs = await plugin_service.list_installed()
    plugins = [_spec_to_response(spec) for spec in specs]
    return PluginListResponse(plugins=plugins)


@router.post(
    "",
    response_model=PluginResponse,
    status_code=201,
    summary="Install a new plugin",
)
async def install_plugin(
    body: PluginInstallRequest,
    plugin_service: PluginService = Depends(get_plugin_service),
) -> PluginResponse:
    """Register a new plugin in the database.

    Rejects duplicate names (``409 Conflict``).
    """
    spec = await plugin_service.install(
        name=body.name,
        display_name=body.display_name,
        version=body.version,
        sdk_version=body.sdk_version,
        description=body.description,
        author=body.author,
        homepage=body.homepage,
        entry_point=body.entry_point,
        manifest_path=body.manifest_path,
        install_path=body.install_path,
        enabled=body.enabled,
        trust_tier=body.trust_tier,
        capabilities=body.capabilities,
        minimum_core_version=body.minimum_core_version,
        maximum_core_version=body.maximum_core_version,
    )
    return _spec_to_response(spec)


@router.get(
    "/{plugin_id}",
    response_model=PluginResponse,
    summary="Get plugin details",
)
async def get_plugin(
    plugin_id: str,
    plugin_service: PluginService = Depends(get_plugin_service),
) -> PluginResponse:
    """Return details for a single plugin by UUID.

    Raises ``404 Not Found`` if the plugin does not exist.
    """
    spec = await plugin_service.get(plugin_id)
    return _spec_to_response(spec)


@router.patch(
    "/{plugin_id}",
    response_model=PluginResponse,
    summary="Update plugin",
)
async def update_plugin(
    plugin_id: str,
    body: PluginUpdateRequest,
    plugin_service: PluginService = Depends(get_plugin_service),
) -> PluginResponse:
    """Update an existing plugin. Only provided fields are changed.

    Raises ``404 Not Found`` if the plugin does not exist.
    """
    spec = await plugin_service.update(
        plugin_id=plugin_id,
        display_name=body.display_name,
        version=body.version,
        description=body.description,
        enabled=body.enabled,
        status=body.status,
        status_message=body.status_message,
        trust_tier=body.trust_tier,
        capabilities=body.capabilities,
        minimum_core_version=body.minimum_core_version,
        maximum_core_version=body.maximum_core_version,
    )
    return _spec_to_response(spec)


@router.delete(
    "/{plugin_id}",
    status_code=204,
    summary="Uninstall plugin",
)
async def uninstall_plugin(
    plugin_id: str,
    plugin_service: PluginService = Depends(get_plugin_service),
) -> None:
    """Remove a plugin and all of its associated data.

    Raises ``404 Not Found`` if the plugin does not exist.
    """
    await plugin_service.uninstall(plugin_id)
