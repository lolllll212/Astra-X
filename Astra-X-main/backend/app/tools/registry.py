from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from app.tools.errors import ToolNotFoundError

if TYPE_CHECKING:
    from app.tools.base import Tool
    from app.tools.models import ToolSchema


class ToolRegistry:
    """Maps tool names to :class:`Tool` instances.

    Usage::

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tool = registry.get("calculator")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance.

        Raises :class:`ValueError` if a tool with the same name is already
        registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry.

        Raises :class:`ToolNotFoundError` if the tool is not registered.
        """
        if tool_name not in self._tools:
            raise ToolNotFoundError(tool_name)
        del self._tools[tool_name]

    def get(self, tool_name: str) -> Tool:
        """Look up a tool by name.

        Raises :class:`ToolNotFoundError` if the tool is not registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name)
        return tool

    def list_tools(self) -> Sequence[Tool]:
        """Return all registered tools (order not guaranteed)."""
        return list(self._tools.values())

    def exists(self, tool_name: str) -> bool:
        """Return ``True`` if a tool is registered under *tool_name*."""
        return tool_name in self._tools

    def schemas(self) -> list[ToolSchema]:
        """Return the schema for every registered tool."""
        return [tool.schema for tool in self._tools.values()]

    @property
    def count(self) -> int:
        return len(self._tools)
