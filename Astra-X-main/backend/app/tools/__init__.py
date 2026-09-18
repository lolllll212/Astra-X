"""Tool execution framework.

Every tool implements :class:`Tool` and registers itself with the
:class:`ToolRegistry`. The :class:`ToolExecutor` resolves :class:`ToolCall`
objects by name, validates arguments, and returns :class:`ToolResult`.

The :class:`CapabilityRegistry` maps abstract capability identifiers
to concrete tools, enabling the planner to reason about capabilities
instead of concrete tool names.
"""

from __future__ import annotations

from app.tools.audit import (
    CandidateValidationTool,
    CoverageCriticResult,
    CoverageUnit,
    CoverageUnitType,
    Finding,
    Hunter,
    HunterStatus,
    IndependentVerificationTool,
    ReconnaissanceTool,
    StructuredOutputTool,
    TargetNeutralReportingTool,
)
from app.tools.autofix import AutoFixTool
from app.tools.base import Tool
from app.tools.browserskill import BrowserSkillTool
from app.tools.capabilities import CapabilityRegistry
from app.tools.colibri import ColibriTool
from app.tools.context import ToolContext
from app.tools.ecc import ECCTool
from app.tools.errors import (
    CapabilityNotFoundError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from app.tools.executor import ToolExecutor
from app.tools.ghidra import GhidraTool
from app.tools.lmstudio import LmStudioTool
from app.tools.models import ToolCall, ToolSchema
from app.tools.n8n import N8NTool
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult
from app.tools.voicebox import VoiceboxTool

__all__ = [
    "AutoFixTool",
    "BrowserSkillTool",
    "CandidateValidationTool",
    "CapabilityNotFoundError",
    "CapabilityRegistry",
    "ColibriTool",
    "CoverageCriticResult",
    "CoverageUnit",
    "CoverageUnitType",
    "ECCTool",
    "Finding",
    "GhidraTool",
    "Hunter",
    "HunterStatus",
    "IndependentVerificationTool",
    "LmStudioTool",
    "N8NTool",
    "ReconnaissanceTool",
    "StructuredOutputTool",
    "TargetNeutralReportingTool",
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
    "VoiceboxTool",
]
