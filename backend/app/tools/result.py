from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Result of a single tool execution.

    Tools always return this structure — never raw ``dict`` or ``str``.
    The :class:`~app.tools.executor.ToolExecutor` converts this into a
    domain-level :class:`~app.domain.tool.ToolResult` for persistence.
    """

    success: bool = True
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: int = 0
