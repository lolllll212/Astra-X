"""Web search tool using the configured search provider."""

from __future__ import annotations

import json
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class SearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for a given query. Returns a list of results with titles, URLs, and snippets."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="query", type_="string", description="Search query", required=True),
                ToolParameter(name="count", type_="integer", description="Number of results to return (max 20)", required=False, default=8),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        query: str = kwargs.get("query", "")
        count: int = int(kwargs.get("count", 8))

        if not query:
            return ToolResult(success=False, error="query is required")
        if count < 1 or count > 20:
            return ToolResult(success=False, error="count must be between 1 and 20")

        import httpx

        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get("https://api.duckduckgo.com/", params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=f"Search failed: {exc}")

        results: list[dict[str, str]] = []
        for topic in data.get("RelatedTopics", []):
            if "Text" in topic and "FirstURL" in topic:
                results.append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })
                if len(results) >= count:
                    break

        if data.get("AbstractText"):
            results.insert(0, {
                "title": data.get("AbstractSource", "Wikipedia"),
                "url": data.get("AbstractURL", ""),
                "snippet": data.get("AbstractText"),
            })

        output = json.dumps({"query": query, "results": results, "total": len(results)}, indent=2)
        return ToolResult(
            success=True,
            output=output,
            metadata={"query": query, "result_count": len(results)},
        )
