"""GitHub repository information tool."""

from __future__ import annotations

import json
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class GitHubRepositoryTool(Tool):
    @property
    def name(self) -> str:
        return "github_repository"

    @property
    def description(self) -> str:
        return "Get information about a GitHub repository: description, stars, forks, language, topics, license."

    @property
    def capabilities(self) -> list[str]:
        return ["get_repository_info", "search_repository"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="owner", type_="string", description="Repository owner (user or organization)", required=True),
                ToolParameter(name="repo", type_="string", description="Repository name", required=True),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        owner: str = kwargs.get("owner", "")
        repo: str = kwargs.get("repo", "")

        if not owner or not repo:
            return ToolResult(success=False, error="owner and repo are required")

        import httpx

        token = context.env.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(success=False, error=f"Repository '{owner}/{repo}' not found")
            return ToolResult(success=False, error=f"GitHub API error: {exc.response.status_code}")
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=str(exc))

        result = {
            "owner": owner,
            "repo": repo,
            "full_name": data.get("full_name"),
            "description": data.get("description"),
            "url": data.get("html_url"),
            "language": data.get("language"),
            "topics": data.get("topics", []),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
            "license": data.get("license", {}).get("spdx_id") if data.get("license") else None,
            "default_branch": data.get("default_branch"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "is_fork": data.get("fork"),
            "is_archived": data.get("archived"),
        }

        return ToolResult(
            success=True,
            output=json.dumps(result, indent=2),
            metadata={"owner": owner, "repo": repo, "stars": result["stars"]},
        )
