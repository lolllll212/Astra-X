"""Tests for plugin ABC, example plugins, and PluginManager activation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.domain.plugin import PluginSpec
from app.plugin.manager import PluginManager
from app.plugins.base import Plugin, PluginContext, PluginRegistries, PluginServices
from app.plugins.example_provider import ExampleProviderPlugin
from app.plugins.weather import WeatherPlugin, WeatherTool
from app.tools.capabilities import CapabilityRegistry
from app.tools.registry import ToolRegistry

# =========================================================================
# Plugin ABC
# =========================================================================


class TestPluginABC:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            Plugin()  # type: ignore[abstract]


# =========================================================================
# WeatherTool
# =========================================================================


class TestWeatherTool:
    @pytest.fixture
    def tool(self) -> WeatherTool:
        return WeatherTool()

    def test_name(self, tool: WeatherTool) -> None:
        assert tool.name == "weather"

    def test_description(self, tool: WeatherTool) -> None:
        assert "weather" in tool.description.lower()

    def test_capabilities(self, tool: WeatherTool) -> None:
        assert tool.capabilities == ["get_weather"]

    def test_schema_has_location_param(self, tool: WeatherTool) -> None:
        schema = tool.schema
        assert schema.name == "weather"
        param_names = [p.name for p in schema.parameters]
        assert "location" in param_names
        assert schema.capabilities == ["get_weather"]

    @pytest.mark.asyncio
    async def test_execute_returns_mock_weather(self, tool: WeatherTool) -> None:
        from app.tools.context import ToolContext

        ctx = ToolContext(conversation_id="test")
        result = await tool.execute(ctx, location="London")
        assert result.success
        assert "London" in result.output
        assert "22°C" in result.output

    @pytest.mark.asyncio
    async def test_execute_requires_location(self, tool: WeatherTool) -> None:
        from app.tools.context import ToolContext

        ctx = ToolContext(conversation_id="test")
        result = await tool.execute(ctx)
        assert not result.success
        assert "required" in result.error


# =========================================================================
# WeatherPlugin
# =========================================================================


class TestWeatherPlugin:
    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        return ToolRegistry()

    @pytest.fixture
    def capability_registry(self, tool_registry: ToolRegistry) -> CapabilityRegistry:
        return CapabilityRegistry(tool_registry)

    @pytest.fixture
    def context(self, tool_registry: ToolRegistry, capability_registry: CapabilityRegistry) -> PluginContext:
        return PluginContext(
            registries=PluginRegistries(
                tool=tool_registry,
                capability=capability_registry,
            ),
        )

    @pytest.fixture
    def plugin(self) -> WeatherPlugin:
        return WeatherPlugin()

    def test_name(self, plugin: WeatherPlugin) -> None:
        assert plugin.name == "weather"

    @pytest.mark.asyncio
    async def test_on_load_registers_tool(
        self,
        plugin: WeatherPlugin,
        tool_registry: ToolRegistry,
        context: PluginContext,
    ) -> None:
        assert not tool_registry.exists("weather")
        await plugin.on_load(context)
        assert tool_registry.exists("weather")
        tool = tool_registry.get("weather")
        assert isinstance(tool, WeatherTool)

    @pytest.mark.asyncio
    async def test_on_load_rebuilds_capability_registry(
        self,
        plugin: WeatherPlugin,
        capability_registry: CapabilityRegistry,
        context: PluginContext,
    ) -> None:
        await plugin.on_load(context)
        assert capability_registry.has_capability("get_weather")
        tool = capability_registry.resolve("get_weather")
        assert isinstance(tool, WeatherTool)

    @pytest.mark.asyncio
    async def test_unload_unregisters_tool(
        self,
        plugin: WeatherPlugin,
        tool_registry: ToolRegistry,
        context: PluginContext,
    ) -> None:
        await plugin.on_load(context)
        assert tool_registry.exists("weather")
        await plugin.on_unload()
        assert not tool_registry.exists("weather")

    @pytest.mark.asyncio
    async def test_unload_rebuilds_capability_registry(
        self,
        plugin: WeatherPlugin,
        capability_registry: CapabilityRegistry,
        context: PluginContext,
    ) -> None:
        await plugin.on_load(context)
        assert capability_registry.has_capability("get_weather")
        await plugin.on_unload()
        assert not capability_registry.has_capability("get_weather")


# =========================================================================
# ExampleProviderPlugin
# =========================================================================


class TestExampleProviderPlugin:
    @pytest.fixture
    def mock_provider_service(self) -> AsyncMock:
        svc = AsyncMock()
        svc.register = AsyncMock()
        svc.remove = AsyncMock()
        return svc

    @pytest.fixture
    def context(self, mock_provider_service: AsyncMock) -> PluginContext:
        return PluginContext(
            services=PluginServices(provider=mock_provider_service),
        )

    @pytest.fixture
    def plugin(self) -> ExampleProviderPlugin:
        return ExampleProviderPlugin()

    def test_name(self, plugin: ExampleProviderPlugin) -> None:
        assert plugin.name == "example_provider"

    @pytest.mark.asyncio
    async def test_on_load_registers_provider(
        self,
        plugin: ExampleProviderPlugin,
        mock_provider_service: AsyncMock,
        context: PluginContext,
    ) -> None:
        await plugin.on_load(context)
        mock_provider_service.register.assert_awaited_once()
        call_kwargs = mock_provider_service.register.call_args[1]
        assert call_kwargs["provider_id"] == "example"
        assert call_kwargs["display_name"] == "Example Plugin Provider"

    @pytest.mark.asyncio
    async def test_on_unload_removes_provider(
        self,
        plugin: ExampleProviderPlugin,
        mock_provider_service: AsyncMock,
        context: PluginContext,
    ) -> None:
        await plugin.on_load(context)
        registered_id = plugin._provider_id
        assert registered_id is not None
        await plugin.on_unload()
        mock_provider_service.remove.assert_awaited_once_with(registered_id)
        assert plugin._provider_id is None

    @pytest.mark.asyncio
    async def test_on_unload_noop_if_not_loaded(
        self,
        plugin: ExampleProviderPlugin,
        mock_provider_service: AsyncMock,
        context: PluginContext,
    ) -> None:
        await plugin.on_unload()
        mock_provider_service.remove.assert_not_awaited()


# =========================================================================
# PluginManager — _load_plugin_instance
# =========================================================================


def _make_spec(
    name: str = "weather",
    entry_point: str | None = "app.plugins.weather:WeatherPlugin",
    enabled: bool = True,
    capabilities: list[str] | None = None,
    minimum_core_version: str | None = None,
    maximum_core_version: str | None = None,
) -> PluginSpec:
    return PluginSpec(
        id=str(uuid4()),
        name=name,
        display_name=name.replace("_", " ").title(),
        version="1.0.0",
        enabled=enabled,
        entry_point=entry_point,
        status="active",
        capabilities=capabilities or [],
        minimum_core_version=minimum_core_version,
        maximum_core_version=maximum_core_version,
    )


class TestPluginManagerLoadInstance:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        r = AsyncMock()
        r.list_enabled = AsyncMock(return_value=[])
        r.find_by_name = AsyncMock(return_value=None)
        return r

    @pytest.fixture
    def manager(self, repo: AsyncMock) -> PluginManager:
        return PluginManager(
            plugin_repository=repo,
            app_version="0.1.0",
        )

    @pytest.mark.asyncio
    async def test_loads_weather_plugin(self, manager: PluginManager) -> None:
        spec = _make_spec()
        instance = await manager._load_plugin_instance(spec)
        assert instance is not None
        assert instance.name == "weather"
        assert isinstance(instance, WeatherPlugin)

    @pytest.mark.asyncio
    async def test_loads_example_provider_plugin(self, manager: PluginManager) -> None:
        spec = _make_spec(
            name="example_provider",
            entry_point="app.plugins.example_provider:ExampleProviderPlugin",
        )
        instance = await manager._load_plugin_instance(spec)
        assert instance is not None
        assert instance.name == "example_provider"

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_entry_point(self, manager: PluginManager) -> None:
        spec = _make_spec(entry_point=None)
        instance = await manager._load_plugin_instance(spec)
        assert instance is None

    @pytest.mark.asyncio
    async def test_returns_none_for_invalid_entry_point(self, manager: PluginManager) -> None:
        spec = _make_spec(entry_point="nonexistent.module:Class")
        instance = await manager._load_plugin_instance(spec)
        assert instance is None

    @pytest.mark.asyncio
    async def test_calls_on_load_during_loading(self, manager: PluginManager) -> None:
        """Verifies that _load_plugin_instance calls on_load on the plugin."""
        spec = _make_spec()
        with patch.object(WeatherPlugin, "on_load", new=AsyncMock()) as mock_on_load:
            instance = await manager._load_plugin_instance(spec)
            assert instance is not None
            mock_on_load.assert_awaited_once()


# =========================================================================
# PluginManager — load_enabled integration
# =========================================================================


class TestPluginManagerLoadEnabled:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        r = AsyncMock()
        spec = _make_spec()
        r.list_enabled = AsyncMock(return_value=[spec])
        return r

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        return ToolRegistry()

    @pytest.fixture
    def capability_registry(self, tool_registry: ToolRegistry) -> CapabilityRegistry:
        return CapabilityRegistry(tool_registry)

    @pytest.fixture
    def manager(
        self,
        repo: AsyncMock,
        tool_registry: ToolRegistry,
        capability_registry: CapabilityRegistry,
    ) -> PluginManager:
        return PluginManager(
            plugin_repository=repo,
            capability_registry=capability_registry,
            app_version="0.1.0",
            tool_registry=tool_registry,
        )

    @pytest.mark.asyncio
    async def test_load_enabled_activates_plugin(
        self,
        manager: PluginManager,
        tool_registry: ToolRegistry,
    ) -> None:
        assert not tool_registry.exists("weather")
        count = await manager.load_enabled()
        assert count == 1
        assert manager.is_loaded("weather")
        assert tool_registry.exists("weather")

    @pytest.mark.asyncio
    async def test_load_enabled_calls_on_load(self, manager: PluginManager) -> None:
        with patch.object(WeatherPlugin, "on_load", new=AsyncMock()) as mock_on_load:
            count = await manager.load_enabled()
            assert count == 1
            mock_on_load.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_enabled_skips_incompatible_version(self, repo: AsyncMock) -> None:
        spec = _make_spec(minimum_core_version="999.0.0")
        repo.list_enabled = AsyncMock(return_value=[spec])
        manager = PluginManager(
            plugin_repository=repo,
            app_version="0.1.0",
        )
        count = await manager.load_enabled()
        assert count == 0
        assert not manager.is_loaded("weather")


# =========================================================================
# PluginManager — reload_plugin
# =========================================================================


class TestPluginManagerReload:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        r = AsyncMock()
        spec = _make_spec()
        r.list_enabled = AsyncMock(return_value=[spec])
        r.find_by_name = AsyncMock(return_value=spec)
        return r

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        return ToolRegistry()

    @pytest.fixture
    def manager(
        self,
        repo: AsyncMock,
        tool_registry: ToolRegistry,
    ) -> PluginManager:
        return PluginManager(
            plugin_repository=repo,
            app_version="0.1.0",
            tool_registry=tool_registry,
        )

    @pytest.mark.asyncio
    async def test_reload_calls_on_unload_then_on_load(
        self,
        manager: PluginManager,
    ) -> None:
        await manager.load_enabled()
        assert manager.is_loaded("weather")

        with (
            patch.object(WeatherPlugin, "on_unload", new=AsyncMock()) as mock_unload,
            patch.object(WeatherPlugin, "on_load", new=AsyncMock()) as mock_load,
        ):
            result = await manager.reload_plugin("weather")
            assert result is not None
            mock_unload.assert_awaited_once()
            mock_load.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reload_removes_plugin_if_not_found(
        self,
        manager: PluginManager,
        repo: AsyncMock,
    ) -> None:
        await manager.load_enabled()
        assert manager.is_loaded("weather")
        repo.find_by_name = AsyncMock(return_value=None)

        result = await manager.reload_plugin("weather")
        assert result is None
        assert not manager.is_loaded("weather")

    @pytest.mark.asyncio
    async def test_reload_skips_incompatible_plugin(
        self,
        manager: PluginManager,
        repo: AsyncMock,
    ) -> None:
        await manager.load_enabled()
        assert manager.is_loaded("weather")

        new_spec = _make_spec(minimum_core_version="999.0.0")
        repo.find_by_name = AsyncMock(return_value=new_spec)

        result = await manager.reload_plugin("weather")
        assert result is None
        assert not manager.is_loaded("weather")


# =========================================================================
# PluginManager — unload_all
# =========================================================================


class TestPluginManagerUnloadAll:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        r = AsyncMock()
        r.list_enabled = AsyncMock(return_value=[_make_spec()])
        return r

    @pytest.fixture
    def manager(self, repo: AsyncMock) -> PluginManager:
        return PluginManager(
            plugin_repository=repo,
            app_version="0.1.0",
        )

    @pytest.mark.asyncio
    async def test_unload_all_calls_on_unload(self, manager: PluginManager) -> None:
        await manager.load_enabled()
        assert manager.is_loaded("weather")

        with patch.object(WeatherPlugin, "on_unload", new=AsyncMock()) as mock_unload:
            await manager.unload_all()
            mock_unload.assert_awaited_once()
            assert not manager.is_loaded("weather")

    @pytest.mark.asyncio
    async def test_unload_all_clears_plugin_instances(self, manager: PluginManager) -> None:
        await manager.load_enabled()
        assert len(manager._plugin_instances) == 1
        await manager.unload_all()
        assert len(manager._plugin_instances) == 0
        assert len(manager._loaded) == 0


# =========================================================================
# PluginManager — list_capabilities integration
# =========================================================================


class TestPluginManagerCapabilities:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        r = AsyncMock()
        spec = _make_spec(capabilities=["get_weather"])
        r.list_enabled = AsyncMock(return_value=[spec])
        return r

    @pytest.fixture
    def tool_registry(self) -> ToolRegistry:
        return ToolRegistry()

    @pytest.fixture
    def capability_registry(self, tool_registry: ToolRegistry) -> CapabilityRegistry:
        return CapabilityRegistry(tool_registry)

    @pytest.fixture
    def manager(
        self,
        repo: AsyncMock,
        tool_registry: ToolRegistry,
        capability_registry: CapabilityRegistry,
    ) -> PluginManager:
        return PluginManager(
            plugin_repository=repo,
            capability_registry=capability_registry,
            app_version="0.1.0",
            tool_registry=tool_registry,
        )

    @pytest.mark.asyncio
    async def test_plugin_capabilities_visible_via_registry(
        self,
        manager: PluginManager,
        capability_registry: CapabilityRegistry,
    ) -> None:
        await manager.load_enabled()
        # The weather plugin registers a tool with "get_weather" capability.
        assert capability_registry.has_capability("get_weather")
        tool = capability_registry.resolve("get_weather")
        assert tool.name == "weather"

    @pytest.mark.asyncio
    async def test_list_capabilities_aggregates_all(self, manager: PluginManager) -> None:
        await manager.load_enabled()
        caps = manager.list_capabilities()
        assert "get_weather" in caps
        assert caps["get_weather"] == "weather"

    @pytest.mark.asyncio
    async def test_get_plugins_for_capability(
        self,
        manager: PluginManager,
    ) -> None:
        await manager.load_enabled()
        plugins = manager.get_plugins_for_capability("get_weather")
        assert len(plugins) == 1
        assert plugins[0].name == "weather"
