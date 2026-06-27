"""List the contents of a directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class ListDirectoryTool(Tool):
    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories at the given path. Optionally filter by glob pattern."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="path", type_="string", description="Directory path", required=True),
                ToolParameter(name="pattern", type_="string", description="Glob pattern filter (e.g. '*.py', '**/*.txt')", required=False),
                ToolParameter(name="recursive", type_="boolean", description="List recursively", required=False, default=False),
                ToolParameter(name="max_depth", type_="integer", description="Maximum directory depth for recursive listing", required=False),
                ToolParameter(name="include_hidden", type_="boolean", description="Include hidden files (dotfiles)", required=False, default=False),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", "")
        pattern: str | None = kwargs.get("pattern")
        recursive: bool = bool(kwargs.get("recursive", False))
        max_depth: int | None = kwargs.get("max_depth")
        include_hidden: bool = bool(kwargs.get("include_hidden", False))

        resolved = self._resolve(path_str, context)
        if isinstance(resolved, ToolResult):
            return resolved

        try:
            if pattern:
                if recursive:
                    paths = sorted(resolved.rglob(pattern))
                else:
                    paths = sorted(resolved.glob(pattern))
            elif recursive:
                paths = sorted(resolved.rglob("*"))
            else:
                paths = sorted(resolved.iterdir())

            entries: list[dict[str, Any]] = []
            for entry in paths:
                if not include_hidden and entry.name.startswith("."):
                    continue
                depth = len(entry.relative_to(resolved).parts)
                if max_depth is not None and depth > max_depth:
                    continue
                try:
                    stat = entry.stat()
                    entries.append({
                        "name": entry.name,
                        "path": str(entry.relative_to(resolved)),
                        "type": "directory" if entry.is_dir() else "file",
                        "size_bytes": stat.st_size if entry.is_file() else 0,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    continue

            meta = {
                "path": str(resolved),
                "total_entries": len(entries),
                "directories": sum(1 for e in entries if e["type"] == "directory"),
                "files": sum(1 for e in entries if e["type"] == "file"),
            }
            return ToolResult(
                success=True,
                output=json.dumps({"entries": entries, **meta}, indent=2),
                metadata=meta,
            )
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {resolved}")
        except NotADirectoryError:
            return ToolResult(success=False, error=f"Not a directory: {resolved}")
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
            return ToolResult(success=False, error=f"Path '{p}' is outside allowed workspace '{workspace_root}'")

        if not p.exists():
            return ToolResult(success=False, error=f"Path not found: {p}")
        if not p.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {p}")

        return p
