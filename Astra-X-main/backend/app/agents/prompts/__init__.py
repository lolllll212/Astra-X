"""Agent prompt templates."""

from __future__ import annotations

from app.agents.prompts.planner import PLANNER_SYSTEM_PROMPT
from app.agents.prompts.reflection import REFLECTION_SYSTEM_PROMPT
from app.agents.prompts.system import AGENT_SYSTEM_PROMPT

__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "REFLECTION_SYSTEM_PROMPT",
]
