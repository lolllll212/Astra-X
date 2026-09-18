"""Per-execution context injected into every tool call.

Tools receive a :class:`ToolContext` instead of dozens of individual
parameters.  The context carries ambient information about the
conversation, agent state, settings, logger, and workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import Logger
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.agents.state import AgentState


@dataclass
class ToolContext:
    """Contextual information passed to every tool execution."""

    conversation_id: str = ""
    agent_state: AgentState | None = None
    settings: Any | None = None
    logger: Logger | None = None
    request_id: str | None = None
    workspace: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    """Additional environment variables the tool may need (e.g. API keys)."""
