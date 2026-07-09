"""Tool execution framework.

Every tool implements :class:`Tool` and registers itself with the
:class:`ToolRegistry`. The :class:`ToolExecutor` resolves :class:`ToolCall`
objects by name, validates arguments, and returns :class:`ToolResult`.

The :class:`CapabilityRegistry` maps abstract capability identifiers
to concrete tools, enabling the planner to reason about capabilities
instead of concrete tool names.
"""

from __future__ import annotations

from app.tools.base import Tool
from app.tools.capabilities import CapabilityRegistry
from app.tools.context import ToolContext
from app.tools.errors import (
    CapabilityNotFoundError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall, ToolSchema
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

__all__ = [
    "CapabilityNotFoundError",
    "CapabilityRegistry",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
]
