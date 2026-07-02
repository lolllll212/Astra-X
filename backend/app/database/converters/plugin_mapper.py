"""Map between ORM PluginModel and domain PluginSpec.

Follows the ``provider_mapper.py`` pattern exactly —
standalone ``*_to_model`` and ``*_from_model`` functions that the
``PluginRepository`` delegates to.
"""

from __future__ import annotations

from app.database.models.plugin import (
    PluginConfigModel,
    PluginDependencyModel,
    PluginLimitModel,
    PluginModel,
    PluginPermissionModel,
)
from app.domain.plugin import (
    PluginConfigSpec,
    PluginDependencySpec,
    PluginLimitSpec,
    PluginPermissionSpec,
    PluginSpec,
)

__all__ = [
    "plugin_to_model",
    "plugin_from_model",
    "plugin_config_to_model",
    "plugin_config_from_model",
    "plugin_dependency_to_model",
    "plugin_dependency_from_model",
    "plugin_permission_to_model",
    "plugin_permission_from_model",
    "plugin_limit_to_model",
    "plugin_limit_from_model",
]


# -- Plugin -------------------------------------------------------------------


def plugin_to_model(domain: PluginSpec) -> PluginModel:
    return PluginModel(
        id=domain.id,
        name=domain.name,
        display_name=domain.display_name,
        version=domain.version,
        sdk_version=domain.sdk_version,
        description=domain.description,
        author=domain.author,
        homepage=domain.homepage,
        license_=domain.license_,
        trust_tier=domain.trust_tier,
        enabled=domain.enabled,
        install_type=domain.install_type,
        entry_point=domain.entry_point,
        manifest_path=domain.manifest_path,
        install_path=domain.install_path,
        capabilities=domain.capabilities,
        minimum_core_version=domain.minimum_core_version,
        maximum_core_version=domain.maximum_core_version,
        status=domain.status,
        status_message=domain.status_message,
        checksum=domain.checksum,
    )


def plugin_from_model(model: PluginModel) -> PluginSpec:
    return PluginSpec(
        id=model.id,
        name=model.name,
        display_name=model.display_name,
        version=model.version,
        sdk_version=model.sdk_version,
        description=model.description,
        author=model.author,
        homepage=model.homepage,
        license_=model.license_,
        trust_tier=model.trust_tier,
        enabled=model.enabled,
        install_type=model.install_type,
        entry_point=model.entry_point,
        manifest_path=model.manifest_path,
        install_path=model.install_path,
        capabilities=model.capabilities or [],
        minimum_core_version=model.minimum_core_version,
        maximum_core_version=model.maximum_core_version,
        status=model.status,
        status_message=model.status_message,
        checksum=model.checksum,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


# -- Plugin Config ------------------------------------------------------------


def plugin_config_to_model(domain: PluginConfigSpec) -> PluginConfigModel:
    return PluginConfigModel(
        id=domain.id,
        plugin_id=domain.plugin_id,
        key=domain.key,
        value=domain.value,
        is_secret=domain.is_secret,
    )


def plugin_config_from_model(model: PluginConfigModel) -> PluginConfigSpec:
    return PluginConfigSpec(
        id=model.id,
        plugin_id=model.plugin_id,
        key=model.key,
        value=model.value,
        is_secret=model.is_secret,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


# -- Plugin Dependency --------------------------------------------------------


def plugin_dependency_to_model(domain: PluginDependencySpec) -> PluginDependencyModel:
    return PluginDependencyModel(
        id=domain.id,
        plugin_id=domain.plugin_id,
        dependency_name=domain.dependency_name,
        version_spec=domain.version_spec,
        is_optional=domain.is_optional,
    )


def plugin_dependency_from_model(model: PluginDependencyModel) -> PluginDependencySpec:
    return PluginDependencySpec(
        id=model.id,
        plugin_id=model.plugin_id,
        dependency_name=model.dependency_name,
        version_spec=model.version_spec,
        is_optional=model.is_optional,
        created_at=model.created_at,
    )


# -- Plugin Permission --------------------------------------------------------


def plugin_permission_to_model(domain: PluginPermissionSpec) -> PluginPermissionModel:
    return PluginPermissionModel(
        id=domain.id,
        plugin_id=domain.plugin_id,
        permission=domain.permission,
        granted=domain.granted,
    )


def plugin_permission_from_model(model: PluginPermissionModel) -> PluginPermissionSpec:
    return PluginPermissionSpec(
        id=model.id,
        plugin_id=model.plugin_id,
        permission=model.permission,
        granted=model.granted,
        created_at=model.created_at,
    )


# -- Plugin Limit -------------------------------------------------------------


def plugin_limit_to_model(domain: PluginLimitSpec) -> PluginLimitModel:
    return PluginLimitModel(
        id=domain.id,
        plugin_id=domain.plugin_id,
        resource_type=domain.resource_type,
        soft_limit=domain.soft_limit,
        hard_limit=domain.hard_limit,
        duration_seconds=domain.duration_seconds,
    )


def plugin_limit_from_model(model: PluginLimitModel) -> PluginLimitSpec:
    return PluginLimitSpec(
        id=model.id,
        plugin_id=model.plugin_id,
        resource_type=model.resource_type,
        soft_limit=model.soft_limit,
        hard_limit=model.hard_limit,
        duration_seconds=model.duration_seconds,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
