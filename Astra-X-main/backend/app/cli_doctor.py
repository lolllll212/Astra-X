"""Doctor checks for the ``astra`` CLI.

Runs health/connectivity checks against every registered tool plus the
default LLM provider, printing a readable summary table.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from app.tools.models import ToolCall

if TYPE_CHECKING:
    from .cli_runtime import AstraRuntime

__all__ = ["run_doctor"]


async def run_doctor(runtime: "AstraRuntime") -> None:
    """Run health checks for all tools and the default LLM provider."""
    print()
    print("  Astra X doctor — tool and provider connectivity")
    print("  " + "-" * 52)

    import asyncio

    tool_tasks = [
        runtime.executor.execute(ToolCall(tool_name=tool.name, arguments={"action": "get_status"}), runtime.tool_context())
        for tool in runtime.registry.list_tools()
    ]
    results = await asyncio.gather(*tool_tasks, return_exceptions=True)

    rows: list[tuple[str, bool, str]] = []
    for tool, result in zip(runtime.registry.list_tools(), results, strict=False):
        if isinstance(result, BaseException):
            rows.append((tool.name, False, str(result)))
            continue
        ok = result.success
        summary = _summarize_status(result.output)
        rows.append((tool.name, ok, summary))

    # LLM provider check
    try:
        health = await runtime.router.check_health(runtime.provider_id)
        llm_ok = bool(health.get(runtime.provider_id))
        llm_summary = "reachable" if llm_ok else "unreachable"
    except Exception as exc:
        llm_ok = False
        llm_summary = str(exc)
    rows.append(("lm_studio (provider)", llm_ok, llm_summary))

    print()
    header = f"  {'Tool':<26} {'Status':<9} {'Details'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, ok, summary in rows:
        status = _paint_green("OK") if ok else _paint_red("FAIL")
        print(f"  {name:<26} {status:<12} {summary}")
    print()


def _summarize_status(output: str | None) -> str:
    """Extract a one-line summary from a tool status JSON output."""
    if not output:
        return "-"
    try:
        payload = json.loads(output)
    except Exception:
        first_line = next((ln.strip() for ln in output.splitlines() if ln.strip()), "")
        return first_line[:60] or "-"
    parts: list[str] = []
    for key in ("reachable", "installed", "built", "cli_found", "server_running", "runs_count"):
        if key in payload:
            parts.append(f"{key}={payload[key]}")
    if "model_count" in payload:
        parts.append(f"models={payload['model_count']}")
    if "supported_families" in payload:
        families = payload["supported_families"]
        parts.append(f"families={len(families)}")
    return ", ".join(parts) if parts else json.dumps(payload)[:60]


def _paint_green(text: str) -> str:
    return f"\x1b[32m{text}\x1b[0m"


def _paint_red(text: str) -> str:
    return f"\x1b[31m{text}\x1b[0m"