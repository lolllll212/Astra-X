"""Scrape structured data from a web page."""

from __future__ import annotations

import json
import re
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class ScrapeTool(Tool):
    @property
    def name(self) -> str:
        return "web_scrape"

    @property
    def description(self) -> str:
        return "Scrape and extract structured data from a web page: title, meta tags, headings, links, and text content."

    @property
    def capabilities(self) -> list[str]:
        return ["scrape_web"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="url", type_="string", description="The URL to scrape", required=True),
                ToolParameter(name="extract", type_="string", description="What to extract: 'all', 'text', 'links', 'images', 'headings'", required=False, default="all",
                             enum=["all", "text", "links", "images", "headings"]),
                ToolParameter(name="timeout", type_="integer", description="Request timeout in seconds", required=False, default=15),
            ],
            capabilities=self.capabilities,
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        url: str = kwargs.get("url", "")
        extract: str = kwargs.get("extract", "all")
        timeout: int = int(kwargs.get("timeout", 15))

        if not url:
            return ToolResult(success=False, error="url is required")

        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "AstraX/1.0"})
                resp.raise_for_status()
                html = resp.text
        except httpx.TimeoutException:
            return ToolResult(success=False, error=f"Request timed out after {timeout}s")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=str(exc))

        result: dict[str, Any] = {"url": url, "extract": extract}

        if extract in ("all", "text"):
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 100000:
                text = text[:100000] + "\n\n[truncated at 100000 characters]"
            result["text"] = text

        if extract in ("all", "links"):
            links = re.findall(r'href="([^"]+)"', html)
            result["links"] = links[:200]

        if extract in ("all", "images"):
            images = re.findall(r'<img[^>]+src="([^"]+)"', html)
            result["images"] = images[:100]

        if extract in ("all", "headings"):
            h1 = re.findall(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
            h2 = re.findall(r'<h2[^>]*>(.*?)</h2>', html, re.IGNORECASE | re.DOTALL)
            h3 = re.findall(r'<h3[^>]*>(.*?)</h3>', html, re.IGNORECASE | re.DOTALL)
            result["headings"] = {
                "h1": [re.sub(r"<[^>]+>", "", h).strip() for h in h1[:20]],
                "h2": [re.sub(r"<[^>]+>", "", h).strip() for h in h2[:40]],
                "h3": [re.sub(r"<[^>]+>", "", h).strip() for h in h3[:60]],
            }

        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            result["title"] = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

        return ToolResult(
            success=True,
            output=json.dumps(result, indent=2, ensure_ascii=False),
            metadata={"url": url, "extract": extract},
        )
