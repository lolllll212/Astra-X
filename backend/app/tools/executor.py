"""Executes :class:`ToolCall` objects through the :class:`ToolRegistry`.

The executor is the only component that calls ``Tool.execute()``.  It
never imports specific tool implementations — all dispatch goes through
the registry.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from app.tools.context import ToolContext
from app.tools.errors import (
    InvalidArgumentsError,
    PermissionDeniedError,
    ToolExecutionError,
    ToolNotFoundError,
)
from app.tools.models import ToolCall, ToolSchema
from app.tools.permissions import PermissionSet
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


class ToolExecutor:
    """Validates and executes tool calls against a registry.

    Usage::

        result = await executor.execute(tool_call, context)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionSet | None = None,
    ) -> None:
        self._registry = registry
        self._permissions = permissions or PermissionSet()

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Resolve, validate, and execute a single tool call."""
        if not self._permissions.is_allowed(call.tool_name):
            return ToolResult(
                success=False,
                error=f"Permission denied for tool '{call.tool_name}'",
            )

        tool = self._registry.get(call.tool_name)

        validation_error = self._validate_args(tool.schema, call.arguments)
        if validation_error is not None:
            return ToolResult(success=False, error=validation_error)

        start = time.monotonic()
        try:
            result: ToolResult = await tool.execute(context, **call.arguments)
        except PermissionDeniedError:
            raise
        except ToolNotFoundError:
            raise
        except InvalidArgumentsError:
            raise
        except ToolExecutionError:
            raise
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return ToolResult(
                success=False,
                error=f"Unexpected error in '{call.tool_name}': {exc}",
                execution_time_ms=elapsed,
            )
        else:
            elapsed = int((time.monotonic() - start) * 1000)
            return ToolResult(
                success=result.success,
                output=result.output,
                error=result.error,
                metadata=result.metadata,
                execution_time_ms=elapsed,
            )

    async def execute_many(
        self,
        calls: Sequence[ToolCall],
        context: ToolContext,
    ) -> list[ToolResult]:
        """Execute multiple calls sequentially (override for parallel)."""
        return [await self.execute(call, context) for call in calls]

    @staticmethod
    def _validate_args(schema: ToolSchema, args: dict[str, Any]) -> str | None:
        required = {p.name for p in schema.parameters if p.required}
        missing = required - set(args)
        if missing:
            return f"Missing required arguments: {', '.join(sorted(missing))}"
        for param in schema.parameters:
            if param.enum is not None and param.name in args and args[param.name] not in param.enum:
                return (
                    f"Argument '{param.name}' must be one of "
                    f"{param.enum}, got '{args[param.name]}'"
                )
        return None
