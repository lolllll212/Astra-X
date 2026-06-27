"""Read the contents of a file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="path", type_="string", description="Absolute or workspace-relative file path", required=True),
                ToolParameter(name="offset", type_="integer", description="Line number to start from (1-indexed)", required=False),
                ToolParameter(name="limit", type_="integer", description="Maximum number of lines to read", required=False),
                ToolParameter(name="encoding", type_="string", description="File encoding (default: utf-8)", required=False, default="utf-8"),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", "")
        encoding: str = kwargs.get("encoding", "utf-8")
        offset: int | None = kwargs.get("offset")
        limit: int | None = kwargs.get("limit")

        resolved = self._resolve(path_str, context)
        if isinstance(resolved, ToolResult):
            return resolved

        try:
            text = resolved.read_text(encoding=encoding)
            lines = text.splitlines(keepends=True)
            total_lines = len(lines)

            start = max(0, int(offset) - 1) if offset is not None else 0

            if limit is not None:
                end = start + int(limit)
                selected = lines[start:end]
            else:
                selected = lines[start:]

            output = "".join(selected)
            meta = {
                "path": str(resolved),
                "total_lines": total_lines,
                "returned_lines": len(selected),
                "size_bytes": resolved.stat().st_size,
            }
            return ToolResult(success=True, output=output, metadata=meta)
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {resolved}")
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {resolved}")
        except UnicodeDecodeError as exc:
            return ToolResult(success=False, error=f"Encoding error: {exc}")
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
            return ToolResult(success=False, error=f"Path '{p}' is outside the allowed workspace '{workspace_root}'")

        if not p.exists():
            return ToolResult(success=False, error=f"File not found: {p}")
        if not p.is_file():
            return ToolResult(success=False, error=f"Not a file: {p}")

        return p
