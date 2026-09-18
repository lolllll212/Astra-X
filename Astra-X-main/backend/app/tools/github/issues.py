"""GitHub issues retrieval tool."""

from __future__ import annotations

import json
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class GitHubIssuesTool(Tool):
    @property
    def name(self) -> str:
        return "github_issues"

    @property
    def description(self) -> str:
        return "List open issues from a GitHub repository."

    @property
    def capabilities(self) -> list[str]:
        return ["get_issues"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="owner", type_="string", description="Repository owner", required=True),
                ToolParameter(name="repo", type_="string", description="Repository name", required=True),
                ToolParameter(name="state", type_="string", description="Issue state: open, closed, all", required=False, default="open",
                             enum=["open", "closed", "all"]),
                ToolParameter(name="label", type_="string", description="Filter by label", required=False),
                ToolParameter(name="count", type_="integer", description="Number of issues to return (max 50)", required=False, default=10),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        owner: str = kwargs.get("owner", "")
        repo: str = kwargs.get("repo", "")
        state: str = kwargs.get("state", "open")
        label: str | None = kwargs.get("label")
        count: int = int(kwargs.get("count", 10))

        if not owner or not repo:
            return ToolResult(success=False, error="owner and repo are required")
        if count < 1 or count > 50:
            return ToolResult(success=False, error="count must be between 1 and 50")

        import httpx

        token = context.env.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        params: dict[str, Any] = {"state": state, "per_page": count}
        if label:
            params["labels"] = label

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}/issues", headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(success=False, error=f"Repository '{owner}/{repo}' not found")
            return ToolResult(success=False, error=f"GitHub API error: {exc.response.status_code}")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=str(exc))

        issues: list[dict[str, Any]] = []
        for item in data[:count]:
            if "pull_request" in item:
                continue
            issues.append({
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state"),
                "labels": [lb.get("name") for lb in item.get("labels", [])],
                "author": item.get("user", {}).get("login"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "comments": item.get("comments"),
                "url": item.get("html_url"),
            })

        return ToolResult(
            success=True,
            output=json.dumps({"owner": owner, "repo": repo, "state": state, "issues": issues, "count": len(issues)}, indent=2),
            metadata={"owner": owner, "repo": repo, "state": state, "count": len(issues)},
        )
