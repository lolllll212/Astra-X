"""Text manipulation tool."""

from __future__ import annotations

import json
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class TextTool(Tool):
    @property
    def name(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        return "Manipulate text. Supports operations: count (words/chars), split, join, upper, lower, trim, reverse, contains, replace, substring."

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="operation", type_="string", description="Operation: count_words, count_chars, split, join, upper, lower, trim, reverse, contains, replace, substring", required=True,
                             enum=["count_words", "count_chars", "split", "join", "upper", "lower", "trim", "reverse", "contains", "replace", "substring"]),
                ToolParameter(name="text", type_="string", description="The input text", required=True),
                ToolParameter(name="separator", type_="string", description="Separator for split/join operations", required=False),
                ToolParameter(name="old", type_="string", description="Text to replace (for replace operation)", required=False),
                ToolParameter(name="new", type_="string", description="Replacement text (for replace operation)", required=False),
                ToolParameter(name="start", type_="integer", description="Start index (for substring operation)", required=False),
                ToolParameter(name="end", type_="integer", description="End index (for substring operation)", required=False),
                ToolParameter(name="substring", type_="string", description="Substring to search for (for contains operation)", required=False),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        operation: str = kwargs.get("operation", "")
        text: str = kwargs.get("text", "")

        if not operation:
            return ToolResult(success=False, error="operation is required")
        if not text:
            return ToolResult(success=False, error="text is required")

        result: Any = None
        try:
            if operation == "count_words":
                result = {"count": len(text.split())}
            elif operation == "count_chars":
                result = {"count": len(text)}
            elif operation == "split":
                sep = kwargs.get("separator")
                result = {"parts": text.split(sep) if sep else text.split()}
            elif operation == "join":
                sep = kwargs.get("separator", "")
                items = json.loads(text) if text.startswith("[") else text.split()
                if not isinstance(items, list):
                    return ToolResult(success=False, error="text must be a JSON array or space-separated list for join")
                result = {"output": sep.join(str(i) for i in items)}
            elif operation == "upper":
                result = {"output": text.upper()}
            elif operation == "lower":
                result = {"output": text.lower()}
            elif operation == "trim":
                result = {"output": text.strip()}
            elif operation == "reverse":
                result = {"output": text[::-1]}
            elif operation == "contains":
                substring = kwargs.get("substring", "")
                if not substring:
                    return ToolResult(success=False, error="substring is required for contains operation")
                result = {"found": substring in text, "position": text.find(substring)}
            elif operation == "replace":
                old = kwargs.get("old", "")
                new = kwargs.get("new", "")
                if not old:
                    return ToolResult(success=False, error="old is required for replace operation")
                result = {"output": text.replace(old, new), "count": text.count(old)}
            elif operation == "substring":
                start = kwargs.get("start")
                end = kwargs.get("end")
                if start is not None or end is not None:
                    s = int(start) if start is not None else 0
                    e = int(end) if end is not None else len(text)
                    result = {"output": text[s:e]}
                else:
                    result = {"output": text}
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return ToolResult(success=False, error=str(exc))

        return ToolResult(
            success=True,
            output=json.dumps(result, indent=2),
            metadata={"operation": operation},
        )
