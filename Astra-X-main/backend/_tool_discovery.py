"""Dump the schema of every trainable tool for the training harness.

Run: python _tool_discovery.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools import Tool, ToolExecutor, ToolRegistry, ToolSchema
from app.tools.audit import (
    CandidateValidationTool,
    HuntingTool,
    IndependentVerificationTool,
    ReconnaissanceTool,
    StructuredOutputTool,
    TargetNeutralReportingTool,
)
from app.tools.autofix import AutoFixTool
from app.tools.browserskill import BrowserSkillTool
from app.tools.builtin import CalculatorTool, DateTimeTool, JsonTool, TextTool, UuidTool
from app.tools.colibri import ColibriTool
from app.tools.ecc import ECCTool
from app.tools.filesystem import ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool
from app.tools.ghidra import GhidraTool
from app.tools.github import GitHubCommitsTool, GitHubIssuesTool, GitHubPullRequestsTool, GitHubRepositoryTool
from app.tools.lmstudio import LmStudioTool
from app.tools.voicebox import VoiceboxTool
from app.tools.web import DownloadTool, FetchTool, ScrapeTool, SearchTool
from app.tools.capabilities import CapabilityRegistry

TOOL_CLASSES = [
    CalculatorTool, DateTimeTool, JsonTool, TextTool, UuidTool,
    ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool,
    GitHubCommitsTool, GitHubIssuesTool, GitHubPullRequestsTool, GitHubRepositoryTool,
    DownloadTool, FetchTool, ScrapeTool, SearchTool,
    ReconnaissanceTool, HuntingTool, CandidateValidationTool,
    StructuredOutputTool, IndependentVerificationTool, TargetNeutralReportingTool,
    BrowserSkillTool, VoiceboxTool, ECCTool, AutoFixTool, ColibriTool, GhidraTool, LmStudioTool,
]


def main() -> None:
    registry = ToolRegistry()
    for cls in TOOL_CLASSES:
        try:
            tool = cls()
            registry.register(tool)
        except Exception as exc:
            print(f"REGISTER-FAIL {getattr(cls, '__name__', cls)}: {exc}")

    caps = CapabilityRegistry(registry)
    caps.rebuild()

    report = {}
    for tool in registry.list_tools():
        s = tool.schema
        report[tool.name] = {
            "name": s.name,
            "description": s.description[:160],
            "capabilities": tool.capabilities,
            "params": [{"name": p.name, "type": p.type_, "required": getattr(p, "required", False),
                        "enum": getattr(p, "enum", None), "default": getattr(p, "default", None)} for p in s.parameters],
        }
    with open("tool_schemas.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"Registered {registry.count} tools, schema dump -> tool_schemas.json")
    for name in report:
        print(f"  - {name:<35} params={[p['name'] for p in report[name]['params']]}")
    print(f"\nCapabilities registered: {len(caps.list_capabilities())}")


if __name__ == "__main__":
    main()