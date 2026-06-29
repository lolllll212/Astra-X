"""Search for files by name pattern or text content."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class SearchFilesTool(Tool):
    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return "Search for files by glob pattern or grep text content within files."

    @property
    def capabilities(self) -> list[str]:
        return ["search_files"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="root", type_="string", description="Root directory to search in", required=True),
                ToolParameter(name="pattern", type_="string", description="Glob pattern (e.g. '*.py') or text to search for (when mode=grep)", required=True),
                ToolParameter(name="mode", type_="string", description="'glob' for filename matching, 'grep' for content search", required=False, default="glob",
                             enum=["glob", "grep"]),
                ToolParameter(name="max_results", type_="integer", description="Maximum number of results to return", required=False, default=50),
                ToolParameter(name="include_hidden", type_="boolean", description="Include hidden directories in search", required=False, default=False),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        root_str: str = kwargs.get("root", "")
        pattern: str = kwargs.get("pattern", "")
        mode: str = kwargs.get("mode", "glob")
        max_results: int = int(kwargs.get("max_results", 50))
        include_hidden: bool = bool(kwargs.get("include_hidden", False))

        resolved_root = self._resolve(root_str, context)
        if isinstance(resolved_root, ToolResult):
            return resolved_root

        if not pattern:
            return ToolResult(success=False, error="pattern is required")

        results: list[dict[str, Any]] = []

        try:
            if mode == "glob":
                matches = list(resolved_root.rglob(pattern))
                if not include_hidden:
                    matches = [m for m in matches if not any(part.startswith(".") for part in m.relative_to(resolved_root).parts)]
                matches = matches[:max_results]
                results = [{"path": str(m.relative_to(resolved_root)), "type": "directory" if m.is_dir() else "file"} for m in matches]
            elif mode == "grep":
                text_files = list(resolved_root.rglob("*"))
                if not include_hidden:
                    text_files = [f for f in text_files if not any(part.startswith(".") for part in f.relative_to(resolved_root).parts)]
                for f in text_files:
                    if len(results) >= max_results:
                        break
                    if not f.is_file():
                        continue
                    try:
                        text = f.read_text(encoding="utf-8", errors="ignore")
                        if pattern in text:
                            results.append({"path": str(f.relative_to(resolved_root)), "size_bytes": f.stat().st_size})
                    except (OSError, PermissionError):
                        continue
            else:
                return ToolResult(success=False, error=f"Unknown mode: {mode}")

            return ToolResult(
                success=True,
                output=json.dumps({"results": results, "total": len(results), "mode": mode, "pattern": pattern}, indent=2),
                metadata={"total": len(results), "mode": mode},
            )
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {resolved_root}")
        except OSError as exc:
            return ToolResult(success=False, error=str(exc))

    @staticmethod
    def _resolve(path_str: str, context: ToolContext) -> Path | ToolResult:
        p = Path(path_str)
        if not p.is_absolute():
            base = Path(context.workspace) if context.workspace else Path.cwd()
            p = base / p
        p = p.resolve()

        workspace_root = Path(context.workspace).resolve() if context.workspace else Path.cwd().resolve()
        try:
            p.relative_to(workspace_root)
        except ValueError:
            return ToolResult(success=False, error=f"Path '{p}' is outside allowed workspace")

        if not p.exists():
            return ToolResult(success=False, error=f"Directory not found: {p}")
        if not p.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {p}")

        return p
