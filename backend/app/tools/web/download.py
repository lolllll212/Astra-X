"""Download a file from a URL to the local workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class DownloadTool(Tool):
    @property
    def name(self) -> str:
        return "web_download"

    @property
    def description(self) -> str:
        return "Download a file from a URL and save it to the workspace."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="url", type_="string", description="The URL to download from", required=True),
                ToolParameter(name="output_path", type_="string", description="Relative path within workspace to save the file", required=True),
                ToolParameter(name="timeout", type_="integer", description="Download timeout in seconds", required=False, default=60),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        url: str = kwargs.get("url", "")
        output_path: str = kwargs.get("output_path", "")
        timeout: int = int(kwargs.get("timeout", 60))

        if not url:
            return ToolResult(success=False, error="url is required")
        if not output_path:
            return ToolResult(success=False, error="output_path is required")

        import httpx

        ws = Path(context.workspace).resolve() if context.workspace else Path.cwd().resolve()  # noqa: ASYNC240
        dest = (ws / output_path).resolve()

        try:
            dest.relative_to(ws)
        except ValueError:
            return ToolResult(success=False, error=f"output_path '{output_path}' is outside the workspace")

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "AstraX/1.0"})
                resp.raise_for_status()
                data = resp.content
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Download timed out after {timeout}s")
        except httpx.HTTPStatusError as exc:
            return ToolResult(success=False, error=f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=str(exc))

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(
            success=True,
            output=f"Downloaded {len(data)} bytes to {dest}",
            metadata={"url": url, "path": str(dest), "size_bytes": len(data)},
        )
