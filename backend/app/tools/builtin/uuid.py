"""UUID generation tool."""

from __future__ import annotations

import uuid as _uuid
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class UuidTool(Tool):
    @property
    def name(self) -> str:
        return "uuid"

    @property
    def description(self) -> str:
        return "Generate one or more UUIDs (version 4)."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="count", type_="integer", description="Number of UUIDs to generate", required=False, default=1),
                ToolParameter(name="version", type_="integer", description="UUID version (4 only)", required=False, default=4),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        count: int = int(kwargs.get("count", 1))
        version: int = int(kwargs.get("version", 4))

        if count < 1 or count > 100:
            return ToolResult(success=False, error="count must be between 1 and 100")
        if version != 4:
            return ToolResult(success=False, error="version must be 4")

        uuids: list[str] = []
        for _ in range(count):
            uuids.append(str(_uuid.uuid4()))

        output = "\n".join(uuids) if count > 1 else uuids[0]
        return ToolResult(
            success=True,
            output=output,
            metadata={"count": count, "version": version},
        )
