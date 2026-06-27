"""Provider-independent data models for the tool execution layer.

These types describe tools for the planner and carry invocation data
to the executor.  They are distinct from :mod:`app.domain.tool` which
serves persistence and spec generation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

JsonType = str | int | float | bool | None | list[Any] | dict[str, Any]


@dataclass(frozen=True)
class ToolParameter:
    """Describes a single parameter accepted by a tool."""

    name: str
    description: str = ""
    type_: str = field(default="string")
    required: bool = False
    enum: list[str] | None = None
    default: Any = None


@dataclass(frozen=True)
class ToolSchema:
    """Complete schema that describes a tool to the planner and LLM.

    Returned by the ``Tool.schema`` property.  Provider adapters convert
    this into the provider-specific tool definition format.
    """

    name: str
    description: str = ""
    parameters: Sequence[ToolParameter] = field(default_factory=list)


@dataclass(frozen=True)
class ToolArgument:
    """A single named argument supplied to a tool call."""

    name: str
    value: Any


@dataclass(frozen=True)
class ToolCall:
    """A concrete invocation of a tool from the agent framework.

    Carried through the planner → executor pipeline.  The executor
    converts this into a domain :class:`~app.domain.tool.ToolCall` for
    persistence.
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
