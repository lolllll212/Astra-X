"""GitHub pull requests retrieval tool."""

from __future__ import annotations

import json
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class GitHubPullRequestsTool(Tool):
    @property
    def name(self) -> str:
        return "github_pull_requests"

    @property
    def description(self) -> str:
        return "List pull requests from a GitHub repository."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="owner", type_="string", description="Repository owner", required=True),
                ToolParameter(name="repo", type_="string", description="Repository name", required=True),
                ToolParameter(name="state", type_="string", description="PR state: open, closed, all", required=False, default="open",
                             enum=["open", "closed", "all"]),
                ToolParameter(name="count", type_="integer", description="Number of PRs to return (max 50)", required=False, default=10),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        owner: str = kwargs.get("owner", "")
        repo: str = kwargs.get("repo", "")
        state: str = kwargs.get("state", "open")
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

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}/pulls", headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(success=False, error=f"Repository '{owner}/{repo}' not found")
            return ToolResult(success=False, error=f"GitHub API error: {exc.response.status_code}")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=str(exc))

        prs: list[dict[str, Any]] = []
        for item in data[:count]:
            prs.append({
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state"),
                "author": item.get("user", {}).get("login"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "draft": item.get("draft", False),
                "head_branch": item.get("head", {}).get("ref"),
                "base_branch": item.get("base", {}).get("ref"),
                "url": item.get("html_url"),
            })

        return ToolResult(
            success=True,
            output=json.dumps({"owner": owner, "repo": repo, "state": state, "pull_requests": prs, "count": len(prs)}, indent=2),
            metadata={"owner": owner, "repo": repo, "state": state, "count": len(prs)},
        )
