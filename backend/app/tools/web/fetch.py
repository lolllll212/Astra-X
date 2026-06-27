"""Fetch the contents of a URL."""

from __future__ import annotations

from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class FetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch the content of a URL and return it as text (HTML stripped to markdown-like plain text)."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="url", type_="string", description="The URL to fetch", required=True),
                ToolParameter(name="format", type_="string", description="Output format: 'text' or 'html'", required=False, default="text"),
                ToolParameter(name="timeout", type_="integer", description="Request timeout in seconds", required=False, default=15),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        url: str = kwargs.get("url", "")
        fmt: str = kwargs.get("format", "text")
        timeout: int = int(kwargs.get("timeout", 15))

        if not url:
            return ToolResult(success=False, error="url is required")
        if timeout < 1 or timeout > 60:
            return ToolResult(success=False, error="timeout must be between 1 and 60")

        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "AstraX/1.0"})
                resp.raise_for_status()
                content = resp.text
                content_type = resp.headers.get("content-type", "")
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Request timed out after {timeout}s")
        except httpx.HTTPStatusError as exc:
            return ToolResult(success=False, error=f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=str(exc))

        if fmt == "text":
            import re
            text = re.sub(r"<[^>]+>", " ", content)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 50000:
                text = text[:50000] + "\n\n[truncated at 50000 characters]"
            content = text

        meta = {
            "url": url,
            "content_type": content_type,
            "size_bytes": len(content),
            "format": fmt,
        }
        return ToolResult(success=True, output=content, metadata=meta)
