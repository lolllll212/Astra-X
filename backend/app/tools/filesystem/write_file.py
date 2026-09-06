"""Write content to a file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class WriteFileTool(Tool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if they do not exist."

    @property
    def capabilities(self) -> list[str]:
        return ["write_file", "modify_file_content"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="path", type_="string", description="Absolute or workspace-relative file path", required=True),
                ToolParameter(name="content", type_="string", description="Content to write to the file", required=True),
                ToolParameter(name="encoding", type_="string", description="File encoding (default: utf-8)", required=False, default="utf-8"),
                ToolParameter(name="append", type_="boolean", description="If true, append instead of overwrite", required=False, default=False),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", "")
        content: str = kwargs.get("content", "")
        encoding: str = kwargs.get("encoding", "utf-8")
        append: bool = bool(kwargs.get("append", False))

        resolved = self._resolve(path_str, context)
        if isinstance(resolved, ToolResult):
            return resolved

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding=encoding)
            return ToolResult(
                success=True,
                output=f"Written {len(content.encode(encoding))} bytes to {resolved}",
                metadata={"path": str(resolved), "bytes": len(content.encode(encoding)), "append": append},
            )
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {resolved}")
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

        return p
