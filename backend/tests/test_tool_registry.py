"""Unit tests for ToolRegistry and ToolExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.errors import ToolNotFoundError
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall, ToolParameter, ToolSchema
from app.tools.permissions import PermissionRule, PermissionSet
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


class _FakeTool(Tool):
    """Minimal tool stub for testing."""

    def __init__(
        self,
        name: str,
        output: str = "done",
        execute_side_effect: object = None,
    ) -> None:
        self._name = name
        self._output = output
        self._execute_side_effect = execute_side_effect

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=self.description,
            parameters=[
                ToolParameter(name="input", type_="string", required=False),
            ],
        )

    @property
    def capabilities(self) -> list[str]:
        return [self._name]

    async def _execute(self, context: object, **kwargs: object) -> ToolResult:
        if self._execute_side_effect:
            raise self._execute_side_effect
        return ToolResult(success=True, output=self._output)


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = _FakeTool("test_tool")
        registry.register(tool)
        assert registry.get("test_tool") is tool

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_FakeTool("dup"))

    def test_get_unknown_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="missing"):
            registry.get("missing")

    def test_unregister_removes_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("tool"))
        registry.unregister("tool")
        assert registry.exists("tool") is False

    def test_unregister_unknown_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="missing"):
            registry.unregister("missing")

    def test_exists_returns_true_for_registered(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("tool"))
        assert registry.exists("tool") is True

    def test_exists_returns_false_for_unregistered(self) -> None:
        registry = ToolRegistry()
        assert registry.exists("missing") is False

    def test_list_tools_returns_all(self) -> None:
        registry = ToolRegistry()
        t1 = _FakeTool("t1")
        t2 = _FakeTool("t2")
        registry.register(t1)
        registry.register(t2)
        tools = registry.list_tools()
        assert t1 in tools
        assert t2 in tools
        assert len(tools) == 2

    def test_list_tools_empty_initially(self) -> None:
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_schemas_returns_all_schemas(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("t1"))
        registry.register(_FakeTool("t2"))
        schemas = registry.schemas()
        names = {s.name for s in schemas}
        assert names == {"t1", "t2"}

    def test_count(self) -> None:
        registry = ToolRegistry()
        assert registry.count == 0
        registry.register(_FakeTool("t1"))
        assert registry.count == 1


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_calls_tool_and_returns_result(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("greet", output="hello"))
        executor = ToolExecutor(registry)
        result = await executor.execute(
            ToolCall(tool_name="greet", arguments={"input": "world"}),
            ToolContext(),
        )
        assert result.success is True
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_execute_permission_denied(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("secret"))
        perms = PermissionSet(rules=[
            PermissionRule(effect="deny", tool_pattern="secret"),
        ])
        executor = ToolExecutor(registry, permissions=perms)
        result = await executor.execute(
            ToolCall(tool_name="secret"),
            ToolContext(),
        )
        assert result.success is False
        assert "Permission denied" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        with pytest.raises(ToolNotFoundError):
            await executor.execute(
                ToolCall(tool_name="nonexistent"),
                ToolContext(),
            )

    @pytest.mark.asyncio
    async def test_execute_validates_required_args(self) -> None:
        class _StrictTool(_FakeTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="strict",
                    description="Needs name",
                    parameters=[
                        ToolParameter(name="name", type_="string", required=True),
                    ],
                )

        registry = ToolRegistry()
        registry.register(_StrictTool("strict"))
        executor = ToolExecutor(registry)
        result = await executor.execute(
            ToolCall(tool_name="strict", arguments={}),
            ToolContext(),
        )
        assert result.success is False
        assert "Missing required" in result.error
        assert "name" in result.error

    @pytest.mark.asyncio
    async def test_execute_validates_enum_args(self) -> None:
        class _EnumTool(_FakeTool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(
                    name="picker",
                    description="Pick one",
                    parameters=[
                        ToolParameter(
                            name="color", type_="string", required=True,
                            enum=["red", "blue"],
                        ),
                    ],
                )

        registry = ToolRegistry()
        registry.register(_EnumTool("picker"))
        executor = ToolExecutor(registry)
        result = await executor.execute(
            ToolCall(tool_name="picker", arguments={"color": "green"}),
            ToolContext(),
        )
        assert result.success is False
        assert "must be one of" in result.error

    @pytest.mark.asyncio
    async def test_execute_unexpected_exception_caught(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("crash", execute_side_effect=RuntimeError("boom")))
        executor = ToolExecutor(registry)
        result = await executor.execute(
            ToolCall(tool_name="crash"),
            ToolContext(),
        )
        assert result.success is False
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_execute_many_runs_all(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("a", output="1"))
        registry.register(_FakeTool("b", output="2"))
        executor = ToolExecutor(registry)
        results = await executor.execute_many(
            [
                ToolCall(tool_name="a"),
                ToolCall(tool_name="b"),
            ],
            ToolContext(),
        )
        assert len(results) == 2
        assert results[0].output == "1"
        assert results[1].output == "2"

    @pytest.mark.asyncio
    async def test_execute_records_execution_time(self) -> None:
        registry = ToolRegistry()
        registry.register(_FakeTool("slow", output="done"))
        executor = ToolExecutor(registry)
        result = await executor.execute(
            ToolCall(tool_name="slow", arguments={}),
            ToolContext(),
        )
        assert result.execution_time_ms >= 0


class TestPermissionSet:
    def test_default_allows_all(self) -> None:
        ps = PermissionSet()
        assert ps.is_allowed("anything") is True

    def test_deny_rule_blocks(self) -> None:
        ps = PermissionSet(rules=[
            PermissionRule(effect="deny", tool_pattern="blocked_tool"),
        ])
        assert ps.is_allowed("blocked_tool") is False
        assert ps.is_allowed("other_tool") is True

    def test_allow_rule_overrides_deny(self) -> None:
        """First match wins: allow before deny overrides deny-all."""
        ps = PermissionSet(rules=[
            PermissionRule(effect="allow", tool_pattern="safe_tool"),
            PermissionRule(effect="deny", tool_pattern="*"),
        ])
        assert ps.is_allowed("safe_tool") is True
        assert ps.is_allowed("other") is False

    def test_wildcard_pattern(self) -> None:
        ps = PermissionSet(rules=[
            PermissionRule(effect="deny", tool_pattern="*"),
        ])
        assert ps.is_allowed("everything") is False

    def test_prefix_pattern(self) -> None:
        ps = PermissionSet(rules=[
            PermissionRule(effect="deny", tool_pattern="filesystem.*"),
        ])
        assert ps.is_allowed("filesystem.read") is False
        assert ps.is_allowed("web.search") is True

    def test_exact_pattern(self) -> None:
        ps = PermissionSet(rules=[
            PermissionRule(effect="deny", tool_pattern="exact_tool"),
        ])
        assert ps.is_allowed("exact_tool") is False
        assert ps.is_allowed("exact_tool_extra") is True
