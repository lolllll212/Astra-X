"""Capability-based tool resolution.

The :class:`CapabilityRegistry` maps abstract capability identifiers
(e.g. ``"search_web"``, ``"calculate"``, ``"read_file"``) to concrete
:class:`Tool` instances.  The planner reasons about capabilities instead
of concrete tool names, making the system extensible — adding a new tool
that provides an existing capability requires zero planner changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.tools.errors import CapabilityNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.tools.base import Tool
    from app.tools.registry import ToolRegistry


class CapabilityRegistry:
    """Maps capability identifiers to concrete tools.

    Usage::

        registry = CapabilityRegistry(tool_registry)
        tool = registry.resolve("search_web")
        tool_name = registry.resolve_name("search_web")
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._capabilities: dict[str, list[Tool]] = {}
        self._rebuild()

    # -- public API ---------------------------------------------------------

    def resolve(self, capability: str) -> Tool:
        """Resolve a capability to the best-matching concrete tool.

        Args:
            capability: The capability identifier (e.g. ``"search_web"``).

        Returns:
            A registered tool instance.

        Raises:
            CapabilityNotFoundError: If no tool provides this capability.
        """
        tools = self._capabilities.get(capability)
        if not tools:
            raise CapabilityNotFoundError(capability)
        return tools[0]

    def resolve_name(self, capability: str) -> str:
        """Resolve a capability to a concrete tool name.

        Args:
            capability: The capability identifier.

        Returns:
            The tool name that provides this capability.

        Raises:
            CapabilityNotFoundError: If no tool provides this capability.
        """
        return self.resolve(capability).name

    def has_capability(self, capability: str) -> bool:
        """Return ``True`` if at least one tool provides *capability*."""
        return capability in self._capabilities and bool(self._capabilities[capability])

    def list_capabilities(self) -> list[str]:
        """Return all registered capability identifiers."""
        return list(self._capabilities.keys())

    def get_tools_for_capability(self, capability: str) -> Sequence[Tool]:
        """Return all tools that provide *capability* (may be empty)."""
        return list(self._capabilities.get(capability, []))

    def rebuild(self) -> None:
        """Rebuild the capability index from the current tool registry.

        Call this after registering or unregistering tools.
        """
        self._rebuild()

    # -- internal -----------------------------------------------------------

    def _rebuild(self) -> None:
        self._capabilities.clear()
        for tool in self._tool_registry.list_tools():
            for cap in tool.capabilities:
                self._capabilities.setdefault(cap, []).append(tool)
