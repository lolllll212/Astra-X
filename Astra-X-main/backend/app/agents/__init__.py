"""Agent framework for structured, multi-step AI reasoning.

The agent framework provides a plan → execute → reflect loop that is
coordinated by :class:`Coordinator` and composed of specialised agents:

* :class:`Planner` — decomposes goals into task sequences.
* :class:`Executor` — runs individual tasks.
* :class:`Reflection` — evaluates results and decides next steps.
* :class:`MemoryManager` — bridges agent execution to memory storage.
* :class:`TaskGraph` — tracks task dependencies and readiness.

This module follows the same architecture as the rest of Astra X:
clean separation of concerns, dependency injection, and no circular
imports.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentConfig
from app.agents.coordinator import Coordinator, CoordinatorResult
from app.agents.executor import Executor
from app.agents.memory_manager import MemoryManager
from app.agents.planner import Planner
from app.agents.reflection import Reflection
from app.agents.state import AgentState
from app.agents.task_graph import TaskGraph

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentState",
    "Coordinator",
    "CoordinatorResult",
    "Executor",
    "MemoryManager",
    "Planner",
    "Reflection",
    "TaskGraph",
]
