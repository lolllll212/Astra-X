"""Calculator Plugin — provides safe mathematical expression evaluation."""

from __future__ import annotations

from app.plugins.base import Plugin, PluginContext
from app.tools.builtin.calculator import CalculatorTool


class CalculatorPlugin(Plugin):
    """Plugin that registers CalculatorTool on load and unregisters on unload."""

    def __init__(self) -> None:
        self._tool: CalculatorTool | None = None
        self._context: PluginContext | None = None

    @property
    def name(self) -> str:
        return "calculator"

    async def on_load(self, context: PluginContext) -> None:
        self._context = context
        self._tool = CalculatorTool()
        tr = context.registries.tool
        if tr:
            tr.register(self._tool)
        cr = context.registries.capability
        if cr:
            cr.rebuild()

    async def on_unload(self) -> None:
        if self._tool is not None and self._context is not None:
            tr = self._context.registries.tool
            if tr and tr.exists(self._tool.name):
                tr.unregister(self._tool.name)
            cr = self._context.registries.capability
            if cr:
                cr.rebuild()
        self._tool = None
        self._context = None