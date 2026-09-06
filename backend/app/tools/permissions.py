"""Tool permission and sandbox infrastructure.

This module provides the building blocks for restricting which tools
an agent may invoke and what resources (files, network, system calls)
they can access.

Current state — **scaffold only**.  Concrete sandbox implementations
(policy-based allow/deny lists, subprocess jail, capability tokens)
are added on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PermissionRule:
    """A single allow / deny rule targeting a tool or resource pattern."""

    effect: str  # "allow" | "deny"
    tool_pattern: str  # glob, e.g. "filesystem.*", "python.*"
    reason: str = ""


@dataclass
class PermissionSet:
    """Collection of rules evaluated in order (first match wins)."""

    rules: list[PermissionRule] = field(default_factory=list)

    def is_allowed(self, tool_name: str) -> bool:
        for rule in self.rules:
            if self._matches(rule.tool_pattern, tool_name):
                return rule.effect == "allow"
        return True

    @staticmethod
    def _matches(pattern: str, name: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(".*") and name.startswith(pattern[:-1]):
            return True
        return pattern == name
