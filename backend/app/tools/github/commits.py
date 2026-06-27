"""GitHub commits retrieval tool."""

from __future__ import annotations

import json
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class GitHubCommitsTool(Tool):
    @property
    def name(self) -> str:
        return "github_commits"

    @property
    def description(self) -> str:
        return "List recent commits from a GitHub repository."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="owner", type_="string", description="Repository owner", required=True),
                ToolParameter(name="repo", type_="string", description="Repository name", required=True),
                ToolParameter(name="branch", type_="string", description="Branch name (default: default branch)", required=False),
                ToolParameter(name="count", type_="integer", description="Number of commits to return (max 50)", required=False, default=10),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        owner: str = kwargs.get("owner", "")
        repo: str = kwargs.get("repo", "")
        branch: str | None = kwargs.get("branch")
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

        params: dict[str, Any] = {"per_page": count}
        if branch:
            params["sha"] = branch

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}/commits", headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(success=False, error=f"Repository '{owner}/{repo}' not found")
            return ToolResult(success=False, error=f"GitHub API error: {exc.response.status_code}")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=str(exc))

        commits: list[dict[str, Any]] = []
        for item in data[:count]:
            commit = item.get("commit", {})
            author = commit.get("author", {})
            commits.append({
                "sha": item.get("sha", "")[:7],
                "message": commit.get("message", "").split("\n")[0],
                "author": author.get("name"),
                "email": author.get("email"),
                "date": author.get("date"),
                "url": item.get("html_url"),
            })

        return ToolResult(
            success=True,
            output=json.dumps({"owner": owner, "repo": repo, "commits": commits, "count": len(commits)}, indent=2),
            metadata={"owner": owner, "repo": repo, "count": len(commits)},
        )
