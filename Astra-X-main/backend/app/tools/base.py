from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.tools.context import ToolContext
from app.tools.models import ToolSchema
from app.tools.result import ToolResult


class Tool(ABC):
    """Base class for every tool in the system.

    Subclasses must implement ``name``, ``description``, ``schema``, and
    ``execute()``.  Tools are stateless: all mutable context flows through
    the :class:`ToolContext` parameter.

    Each tool also declares one or more **capabilities** — abstract
    identifiers (e.g. ``"search_web"``, ``"calculate"``) that the planner
    uses instead of concrete tool names.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used to route tool calls from the executor."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable summary shown to the planner and the LLM."""

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Structured schema describing parameters for the LLM."""

    @property
    def capabilities(self) -> list[str]:
        """Capability identifiers this tool provides.

        By default the tool's own name is its sole capability.
        Subclasses should override this to declare more specific or
        additional capabilities (e.g. ``["search_web"]`` instead of
        ``["web_search"]``).
        """
        return [self.name]

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given context and keyword arguments.

        Args:
            context: Ambient execution context (conversation, settings, logger, …).
            **kwargs: Tool-specific arguments validated against ``self.schema``.

        Returns:
            A :class:`ToolResult` — never ``None`` and never a raw value.
        """
        return await self._execute(context, **kwargs)

    @abstractmethod
    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        """Internal execution hook.  Override this instead of ``execute``."""
