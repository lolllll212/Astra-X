"""Web Tools Plugin — provides web search, fetch, scrape, and download capabilities."""

from __future__ import annotations

from app.plugins.base import Plugin, PluginContext
from app.tools.web.search import SearchTool
from app.tools.web.fetch import FetchTool
from app.tools.web.scrape import ScrapeTool
from app.tools.web.download import DownloadTool


class WebToolsPlugin(Plugin):
    """Plugin that registers all web tools on load and unregisters on unload."""

    def __init__(self) -> None:
        self._tools: list = []
        self._context: PluginContext | None = None

    @property
    def name(self) -> str:
        return "web_tools"

    async def on_load(self, context: PluginContext) -> None:
        self._context = context
        tr = context.registries.tool
        if tr is None:
            return

        self._tools = [
            SearchTool(),
            FetchTool(),
            ScrapeTool(),
            DownloadTool(),
        ]
        for tool in self._tools:
            tr.register(tool)

        cr = context.registries.capability
        if cr:
            cr.rebuild()

    async def on_unload(self) -> None:
        if self._context is not None:
            tr = self._context.registries.tool
            if tr:
                for tool in self._tools:
                    if tr.exists(tool.name):
                        tr.unregister(tool.name)
            cr = self._context.registries.capability
            if cr:
                cr.rebuild()
        self._tools.clear()
        self._context = None