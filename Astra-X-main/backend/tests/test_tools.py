"""Unit tests for the tool implementations.

Covers CalculatorTool, TextTool, JsonTool, ReadFileTool, FetchTool,
and PythonRunnerTool.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.settings import Environment
from app.tools.builtin.calculator import CalculatorTool
from app.tools.builtin.json_tools import JsonTool
from app.tools.builtin.text import TextTool
from app.tools.context import ToolContext
from app.tools.filesystem.read_file import ReadFileTool
from app.tools.python.runner import PythonRunnerTool
from app.tools.web.fetch import FetchTool

# =========================================================================
# Helpers
# =========================================================================


def _context(**kw: object) -> ToolContext:
    defaults: dict[str, object] = dict(conversation_id="c1")
    defaults.update(kw)
    return ToolContext(**defaults)  # type: ignore[arg-type]


def _dev_settings() -> MagicMock:
    s = MagicMock()
    s.environment = Environment.DEVELOPMENT
    return s


# =========================================================================
# CalculatorTool
# =========================================================================


class TestCalculatorTool:
    @pytest.fixture
    def tool(self) -> CalculatorTool:
        return CalculatorTool()

    @pytest.mark.asyncio
    async def test_name_and_capabilities(self, tool: CalculatorTool) -> None:
        assert tool.name == "calculator"
        assert "calculate" in tool.capabilities

    @pytest.mark.asyncio
    async def test_basic_addition(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="2 + 2")
        assert result.success
        assert result.output == "4"

    @pytest.mark.asyncio
    async def test_complex_expression(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="(3 + 5) * 2")
        assert result.success
        assert result.output == "16"

    @pytest.mark.asyncio
    async def test_expression_with_math_functions(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="sqrt(16)")
        assert result.success
        assert result.output == "4.0"

    @pytest.mark.asyncio
    async def test_expression_with_constants(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="pi * 2")
        assert result.success
        assert "6.283" in result.output

    @pytest.mark.asyncio
    async def test_empty_expression(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="")
        assert not result.success
        assert result.error is not None and "required" in result.error

    @pytest.mark.asyncio
    async def test_syntax_error(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="2 ~ 2")
        assert not result.success

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="'string'")
        assert not result.success

    @pytest.mark.asyncio
    async def test_division(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="10 / 3")
        assert result.success
        assert float(result.output) == pytest.approx(3.333, rel=0.01)

    @pytest.mark.asyncio
    async def test_floordiv(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="10 // 3")
        assert result.success
        assert result.output == "3"

    @pytest.mark.asyncio
    async def test_modulo(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="10 % 3")
        assert result.success
        assert result.output == "1"

    @pytest.mark.asyncio
    async def test_power(self, tool: CalculatorTool) -> None:
        result = await tool.execute(_context(), expression="2 ** 10")
        assert result.success
        assert result.output == "1024"

    @pytest.mark.asyncio
    async def test_schema(self, tool: CalculatorTool) -> None:
        schema = tool.schema
        assert schema.name == "calculator"
        assert len(schema.parameters) == 1
        assert schema.parameters[0].name == "expression"


# =========================================================================
# TextTool
# =========================================================================


class TestTextTool:
    @pytest.fixture
    def tool(self) -> TextTool:
        return TextTool()

    @pytest.mark.asyncio
    async def test_name_and_capabilities(self, tool: TextTool) -> None:
        assert tool.name == "text"
        assert "manipulate_text" in tool.capabilities

    @pytest.mark.asyncio
    async def test_count_words(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="count_words", text="hello world foo")
        data = json.loads(result.output)
        assert data["count"] == 3

    @pytest.mark.asyncio
    async def test_count_chars(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="count_chars", text="hello")
        data = json.loads(result.output)
        assert data["count"] == 5

    @pytest.mark.asyncio
    async def test_upper(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="upper", text="hello")
        data = json.loads(result.output)
        assert data["output"] == "HELLO"

    @pytest.mark.asyncio
    async def test_lower(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="lower", text="HELLO")
        data = json.loads(result.output)
        assert data["output"] == "hello"

    @pytest.mark.asyncio
    async def test_trim(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="trim", text="  hello  ")
        data = json.loads(result.output)
        assert data["output"] == "hello"

    @pytest.mark.asyncio
    async def test_reverse(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="reverse", text="abc")
        data = json.loads(result.output)
        assert data["output"] == "cba"

    @pytest.mark.asyncio
    async def test_contains_found(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="contains", text="hello world", substring="world")
        data = json.loads(result.output)
        assert data["found"] is True
        assert data["position"] == 6

    @pytest.mark.asyncio
    async def test_contains_not_found(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="contains", text="hello world", substring="xyz")
        data = json.loads(result.output)
        assert data["found"] is False
        assert data["position"] == -1

    @pytest.mark.asyncio
    async def test_replace(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="replace", text="hello world", old="world", new="there")
        data = json.loads(result.output)
        assert data["output"] == "hello there"

    @pytest.mark.asyncio
    async def test_substring(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="substring", text="hello world", start=0, end=5)
        data = json.loads(result.output)
        assert data["output"] == "hello"

    @pytest.mark.asyncio
    async def test_split(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="split", text="a,b,c", separator=",")
        data = json.loads(result.output)
        assert data["parts"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_join(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="join", text='["a","b","c"]', separator=",")
        data = json.loads(result.output)
        assert data["output"] == "a,b,c"

    @pytest.mark.asyncio
    async def test_missing_operation(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), text="hello")
        assert not result.success
        assert result.error is not None and "operation" in result.error

    @pytest.mark.asyncio
    async def test_unknown_operation(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="invalid", text="hello")
        assert not result.success
        assert result.error is not None and "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_contains_missing_substring(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="contains", text="hello")
        assert not result.success
        assert result.error is not None and "substring" in result.error

    @pytest.mark.asyncio
    async def test_replace_missing_old(self, tool: TextTool) -> None:
        result = await tool.execute(_context(), operation="replace", text="hello", new="hi")
        assert not result.success
        assert result.error is not None and "old" in result.error

    @pytest.mark.asyncio
    async def test_schema(self, tool: TextTool) -> None:
        schema = tool.schema
        assert schema.name == "text"
        assert any(p.name == "operation" for p in schema.parameters)


# =========================================================================
# JsonTool
# =========================================================================


class TestJsonTool:
    @pytest.fixture
    def tool(self) -> JsonTool:
        return JsonTool()

    @pytest.mark.asyncio
    async def test_name_and_capabilities(self, tool: JsonTool) -> None:
        assert tool.name == "json"
        assert "manipulate_json" in tool.capabilities

    @pytest.mark.asyncio
    async def test_parse(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="parse", input='{"a": 1}')
        assert result.success
        data = json.loads(result.output)
        assert data["a"] == 1

    @pytest.mark.asyncio
    async def test_validate_valid(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="validate", input='{"a": 1}')
        assert result.success
        assert result.output == "Valid JSON"

    @pytest.mark.asyncio
    async def test_validate_invalid(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="validate", input="not json")
        assert not result.success
        assert result.error is not None and "Invalid JSON" in result.error

    @pytest.mark.asyncio
    async def test_stringify(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="stringify", input='{"a": 1, "b": 2}')
        assert result.success
        assert '"a": 1' in result.output

    @pytest.mark.asyncio
    async def test_query(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="query", input='{"data": {"items": [{"name": "foo"}]}}', path="data.items.0.name")
        assert result.success
        data = json.loads(result.output)
        assert data == "foo"

    @pytest.mark.asyncio
    async def test_query_array_index(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="query", input='[10, 20, 30]', path="1")
        assert result.success
        data = json.loads(result.output)
        assert data == 20

    @pytest.mark.asyncio
    async def test_query_not_found(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="query", input='{"a": 1}', path="b")
        assert not result.success

    @pytest.mark.asyncio
    async def test_missing_operation(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), input='{}')
        assert not result.success
        assert result.error is not None and "operation" in result.error

    @pytest.mark.asyncio
    async def test_unknown_operation(self, tool: JsonTool) -> None:
        result = await tool.execute(_context(), operation="unknown", input='{}')
        assert not result.success
        assert result.error is not None and "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_schema(self, tool: JsonTool) -> None:
        schema = tool.schema
        assert schema.name == "json"
        assert any(p.name == "operation" for p in schema.parameters)


# =========================================================================
# ReadFileTool
# =========================================================================


class TestReadFileTool:
    @pytest.fixture
    def tool(self) -> ReadFileTool:
        return ReadFileTool()

    @pytest.mark.asyncio
    async def test_name_and_capabilities(self, tool: ReadFileTool) -> None:
        assert tool.name == "read_file"
        assert "read_file" in tool.capabilities

    @pytest.mark.asyncio
    async def test_read_whole_file(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        ctx = _context(workspace=str(tmp_path))
        result = await tool.execute(ctx, path="test.txt")
        assert result.success
        assert "line1" in result.output
        assert "line3" in result.output

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        ctx = _context(workspace=str(tmp_path))
        result = await tool.execute(ctx, path="test.txt", offset=2)
        assert result.success
        assert "line1" not in result.output
        assert "line2" in result.output

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tool: ReadFileTool, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        ctx = _context(workspace=str(tmp_path))
        result = await tool.execute(ctx, path="test.txt", limit=2)
        assert result.success
        assert "line1" in result.output
        assert "line2" in result.output
        assert "line3" not in result.output

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool: ReadFileTool) -> None:
        result = await tool.execute(_context(), path="nonexistent.txt")
        assert not result.success
        assert result.error is not None and "not found" in result.error

    @pytest.mark.asyncio
    async def test_path_escape_blocked(self, tool: ReadFileTool, tmp_path: Path) -> None:
        ctx = _context(workspace=str(tmp_path))
        result = await tool.execute(ctx, path="..\\..\\windows\\system32\\drivers\\etc\\hosts")
        assert not result.success
        assert result.error is not None and ("outside" in result.error or "not found" in result.error)

    @pytest.mark.asyncio
    async def test_schema(self, tool: ReadFileTool) -> None:
        schema = tool.schema
        assert schema.name == "read_file"
        assert any(p.name == "path" for p in schema.parameters)


# =========================================================================
# FetchTool
# =========================================================================


class TestFetchTool:
    @pytest.fixture
    def tool(self) -> FetchTool:
        return FetchTool()

    @pytest.mark.asyncio
    async def test_name_and_capabilities(self, tool: FetchTool) -> None:
        assert tool.name == "web_fetch"
        assert "fetch_url" in tool.capabilities

    @pytest.mark.asyncio
    async def test_successful_fetch(self, tool: FetchTool) -> None:
        mock_response = AsyncMock()
        mock_response.text = "<html><body>Hello</body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(_context(), url="http://example.com")

        assert result.success
        assert "Hello" in result.output

    @pytest.mark.asyncio
    async def test_fetch_empty_url(self, tool: FetchTool) -> None:
        result = await tool.execute(_context(), url="")
        assert not result.success
        assert result.error is not None and "url" in result.error

    @pytest.mark.asyncio
    async def test_fetch_invalid_timeout(self, tool: FetchTool) -> None:
        result = await tool.execute(_context(), url="http://example.com", timeout=999)
        assert not result.success
        assert result.error is not None and "timeout" in result.error

    @pytest.mark.asyncio
    async def test_schema(self, tool: FetchTool) -> None:
        schema = tool.schema
        assert schema.name == "web_fetch"
        assert any(p.name == "url" for p in schema.parameters)


# =========================================================================
# PythonRunnerTool
# =========================================================================


class TestPythonRunnerTool:
    @pytest.fixture
    def tool(self) -> PythonRunnerTool:
        return PythonRunnerTool()

    @pytest.mark.asyncio
    async def test_name_and_capabilities(self, tool: PythonRunnerTool) -> None:
        assert tool.name == "python_repl"
        assert "execute_python" in tool.capabilities

    @pytest.mark.asyncio
    async def test_simple_code(self, tool: PythonRunnerTool) -> None:
        result = await tool.execute(
            _context(settings=_dev_settings()),
            code="result = 2 + 2\n_return = result",
        )
        assert result.success
        assert "4" in result.output

    @pytest.mark.asyncio
    async def test_stdout_captured(self, tool: PythonRunnerTool) -> None:
        result = await tool.execute(
            _context(settings=_dev_settings()),
            code="print('hello world')",
        )
        assert result.success
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_syntax_error(self, tool: PythonRunnerTool) -> None:
        result = await tool.execute(
            _context(settings=_dev_settings()),
            code="invalid syntax{{{",
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_code(self, tool: PythonRunnerTool) -> None:
        result = await tool.execute(
            _context(settings=_dev_settings()),
            code="",
        )
        assert not result.success
        assert result.error is not None and "required" in result.error

    @pytest.mark.asyncio
    async def test_timeout_too_high(self, tool: PythonRunnerTool) -> None:
        result = await tool.execute(
            _context(settings=_dev_settings()),
            code="print('hi')",
            timeout=999,
        )
        assert not result.success
        assert result.error is not None and "timeout" in result.error

    @pytest.mark.asyncio
    async def test_sandbox_gate_non_development(self, tool: PythonRunnerTool) -> None:
        s = MagicMock()
        s.environment = "production"
        result = await tool.execute(
            _context(settings=s),
            code="print('hi')",
        )
        assert not result.success
        assert result.error is not None and "disabled" in result.error

    @pytest.mark.asyncio
    async def test_schema(self, tool: PythonRunnerTool) -> None:
        schema = tool.schema
        assert schema.name == "python_repl"
        assert any(p.name == "code" for p in schema.parameters)

    @pytest.mark.asyncio
    async def test_with_vars(self, tool: PythonRunnerTool) -> None:
        result = await tool.execute(
            _context(settings=_dev_settings()),
            code="_return = x + y",
            vars='{"x": 10, "y": 20}',
        )
        assert result.success
        assert "30" in result.output
