from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Result of a single tool execution.

    Tools always return this structure — never raw ``dict`` or ``str``.
    The :class:`~app.tools.executor.ToolExecutor` converts this into a
    domain-level :class:`~app.domain.tool.ToolResult` for persistence.

    Attributes:
        success: Whether execution succeeded.
        output: Text output produced by the tool.
        error: Error message if execution failed.
        metadata: Arbitrary metadata from execution.
        execution_time_ms: Time taken in milliseconds.
        artifacts: Structured artifacts for rich frontend rendering.
            Each artifact is a dict with keys ``type``, ``data``, and
            optionally ``label`` and ``metadata``.
    """

    success: bool = True
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
