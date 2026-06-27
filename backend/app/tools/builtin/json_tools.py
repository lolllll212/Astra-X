"""JSON manipulation tool."""

from __future__ import annotations

import json
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class JsonTool(Tool):
    @property
    def name(self) -> str:
        return "json"

    @property
    def description(self) -> str:
        return "Parse, validate, stringify, or query JSON data. Operations: parse (string → object), stringify (object → pretty string), validate, query (dot-path extraction)."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="operation", type_="string", description="Operation: parse, stringify, validate, query", required=True,
                             enum=["parse", "stringify", "validate", "query"]),
                ToolParameter(name="input", type_="string", description="JSON string input", required=True),
                ToolParameter(name="path", type_="string", description="Dot-separated path for query operation (e.g. 'data.items.0.name')", required=False),
                ToolParameter(name="indent", type_="integer", description="Indentation level for stringify operation", required=False, default=2),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        operation: str = kwargs.get("operation", "")
        input_: str = kwargs.get("input", "")

        if not operation:
            return ToolResult(success=False, error="operation is required")

        try:
            if operation == "parse":
                obj = json.loads(input_)
                return ToolResult(
                    success=True,
                    output=json.dumps(obj, indent=2),
                    metadata={"type": type(obj).__name__},
                )

            if operation == "validate":
                try:
                    json.loads(input_)
                    return ToolResult(success=True, output="Valid JSON")
                except json.JSONDecodeError as exc:
                    return ToolResult(success=False, error=f"Invalid JSON: {exc}")

            if operation == "stringify":
                obj = json.loads(input_)
                indent = int(kwargs.get("indent", 2))
                return ToolResult(
                    success=True,
                    output=json.dumps(obj, indent=indent, ensure_ascii=False),
                )

            if operation == "query":
                obj = json.loads(input_)
                path: str = kwargs.get("path", "")
                if not path:
                    return ToolResult(success=False, error="path is required for query operation")

                value: Any = obj
                for part in path.split("."):
                    if isinstance(value, dict):
                        value = value[part]
                    elif isinstance(value, list) and part.isdigit():
                        value = value[int(part)]
                    else:
                        return ToolResult(success=False, error=f"Path '{path}' not found at '{part}'")

                return ToolResult(
                    success=True,
                    output=json.dumps(value, indent=2, ensure_ascii=False, default=str),
                    metadata={"path": path, "type": type(value).__name__},
                )

            return ToolResult(success=False, error=f"Unknown operation: {operation}")

        except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError) as exc:
            return ToolResult(success=False, error=str(exc))
