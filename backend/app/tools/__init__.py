"""Tool execution framework.

Every tool implements :class:`Tool` and registers itself with the
:class:`ToolRegistry`. The :class:`ToolExecutor` resolves :class:`ToolCall`
objects by name, validates arguments, and returns :class:`ToolResult`.

Built-in tools live under ``builtin/``, filesystem under ``filesystem/``,
web tools under ``web/``, Python execution under ``python/``, and GitHub
tools under ``github/``.
"""

from __future__ import annotations

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.errors import ToolError, ToolExecutionError, ToolNotFoundError
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall, ToolSchema
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

__all__ = [
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
