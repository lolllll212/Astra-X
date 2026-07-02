"""Weather Tool Plugin — demonstrates tool registration lifecycle.

Provides a ``WeatherTool`` that returns mock weather data for a given
location, and a ``WeatherPlugin`` that registers/unregisters the tool
with the ``ToolRegistry`` on load/unload.
"""

from __future__ import annotations

from typing import Any

from app.plugins.base import Plugin, PluginContext
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class WeatherTool(Tool):
    """Mock weather tool — returns sample data for a location."""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get the current weather for a given location."

    @property
    def capabilities(self) -> list[str]:
        return ["get_weather"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="location",
                    type_="string",
                    description="City or region name, e.g. 'London'",
                    required=True,
                ),
            ],
            capabilities=self.capabilities,
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        location: str = kwargs.get("location", "")
        if not location:
            return ToolResult(success=False, error="location is required")

        mock_data = (
            f"Weather in {location}: 22°C, partly cloudy, "
            f"humidity 45%, wind 12 km/h."
        )
        return ToolResult(success=True, output=mock_data)


class WeatherPlugin(Plugin):
    """Plugin that registers WeatherTool on load and unregisters on unload."""

    def __init__(self) -> None:
        self._tool: WeatherTool | None = None
        self._context: PluginContext | None = None

    @property
    def name(self) -> str:
        return "weather"

    async def on_load(self, context: PluginContext) -> None:
        self._context = context
        self._tool = WeatherTool()
        if context.tool_registry:
            context.tool_registry.register(self._tool)
        if context.capability_registry:
            context.capability_registry.rebuild()

    async def on_unload(self) -> None:
        if self._tool is not None and self._context is not None:
            tr = self._context.tool_registry
            if tr and tr.exists(self._tool.name):
                tr.unregister(self._tool.name)
            cr = self._context.capability_registry
            if cr:
                cr.rebuild()
        self._tool = None
        self._context = None
