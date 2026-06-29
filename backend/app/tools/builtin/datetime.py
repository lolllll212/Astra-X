"""Date and time information tool."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class DateTimeTool(Tool):
    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return "Get the current date, time, and timezone information."

    @property
    def capabilities(self) -> list[str]:
        return ["get_datetime"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="format", type_="string", description="Output format: 'iso' (default), 'unix', 'human'", required=False, default="iso"),
                ToolParameter(name="timezone", type_="string", description="Timezone name e.g. 'UTC', 'America/New_York'", required=False, default="UTC"),
            ],
            capabilities=self.capabilities,
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        fmt: str = kwargs.get("format", "iso")
        tz_name: str = kwargs.get("timezone", "UTC")

        now = datetime.now(UTC)

        if fmt == "unix":
            output = str(int(now.timestamp()))
        elif fmt == "human":
            output = now.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC")
        else:
            output = now.isoformat()

        result_data = {
            "utc_iso": now.isoformat(),
            "unix_timestamp": int(now.timestamp()),
            "formatted": output,
            "timezone": tz_name,
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "weekday": now.strftime("%A"),
        }

        return ToolResult(
            success=True,
            output=json.dumps(result_data, indent=2),
            metadata={"datetime": now.isoformat()},
        )
