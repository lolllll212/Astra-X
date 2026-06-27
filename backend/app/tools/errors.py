"""Tool execution exception hierarchy."""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """Base exception for all tool-related errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.details = details or {}
        super().__init__(message)


class ToolNotFoundError(ToolError):
    """Raised when a tool name does not match any registered tool."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' not found in registry")
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Raised when a tool's ``execute()`` method fails."""

    def __init__(self, tool_name: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)
        self.tool_name = tool_name


class InvalidArgumentsError(ToolError):
    """Raised when tool arguments fail validation."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"Invalid arguments for '{tool_name}': {reason}")
        self.tool_name = tool_name


class PermissionDeniedError(ToolError):
    """Raised when a tool is not permitted in the current context."""

    def __init__(self, tool_name: str, reason: str = "Not permitted in current context") -> None:
        super().__init__(f"Permission denied for '{tool_name}': {reason}")
        self.tool_name = tool_name


class TimeoutError_(ToolError):
    """Raised when a tool execution exceeds the configured timeout."""

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        super().__init__(f"Tool '{tool_name}' timed out after {timeout_seconds}s")
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds


class SandboxViolation(ToolError):
    """Raised when a tool attempts an operation forbidden by the sandbox."""

    def __init__(self, tool_name: str, operation: str) -> None:
        super().__init__(f"Sandbox violation in '{tool_name}': {operation}")
        self.tool_name = tool_name
        self.operation = operation
