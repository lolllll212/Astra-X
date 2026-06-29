"""Unit tests for the Python REPL tool runner."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.settings import Environment, Settings
from app.tools.context import ToolContext
from app.tools.python.runner import PythonRunnerTool


@pytest.fixture
def tool() -> PythonRunnerTool:
    return PythonRunnerTool()


@pytest.fixture
def context() -> ToolContext:
    settings = MagicMock(spec=Settings)
    settings.environment = Environment.DEVELOPMENT
    return ToolContext(
        conversation_id="test-conv",
        settings=settings,
    )


@pytest.mark.asyncio
async def test_runner_missing_code(tool: PythonRunnerTool, context: ToolContext) -> None:
    """Missing code parameter returns an error immediately (no subprocess)."""
    result = await tool.execute(context)
    assert not result.success
    assert "code is required" in (result.error or "")


@pytest.mark.asyncio
async def test_runner_raises_with_pipe(tool: PythonRunnerTool, context: ToolContext) -> None:
    """The subprocess script reads code from stdin.  Verify it works end-to-end."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(
            json.dumps({
                "stdout": "hello from repl\n",
                "stderr": "",
                "return": "None",
                "error": None,
            }).encode(),
            b"",
        ),
    )

    with patch.object(
        asyncio,
        "create_subprocess_exec",
        new=AsyncMock(return_value=mock_process),
    ):
        result = await tool.execute(context, code="print('hello from repl')")

    assert result.success
    assert "hello from repl" in result.output


@pytest.mark.asyncio
async def test_runner_timeout(tool: PythonRunnerTool, context: ToolContext) -> None:
    """A timeout is reported as a failure."""
    mock_process = MagicMock()
    mock_process.returncode = None
    mock_process.communicate = AsyncMock(side_effect=TimeoutError)
    mock_process.kill = MagicMock()

    with patch.object(
        asyncio,
        "create_subprocess_exec",
        new=AsyncMock(return_value=mock_process),
    ):
        result = await tool.execute(context, code="while True: pass", timeout=1)

    assert not result.success
    assert "timed out" in (result.error or "").lower()
    mock_process.kill.assert_called_once()


@pytest.mark.asyncio
async def test_runner_nonzero_exit(tool: PythonRunnerTool, context: ToolContext) -> None:
    """Non-zero return code is reported as a failure."""
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"error output"))

    with patch.object(
        asyncio,
        "create_subprocess_exec",
        new=AsyncMock(return_value=mock_process),
    ):
        result = await tool.execute(context, code="exit(1)")

    assert not result.success
    assert "exited with code 1" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_runner_error_in_code(tool: PythonRunnerTool, context: ToolContext) -> None:
    """Python runtime errors inside the subprocess are reported as failures."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(
            json.dumps({
                "stdout": "",
                "stderr": "",
                "return": "None",
                "error": "NameError: name 'x' is not defined",
            }).encode(),
            b"",
        ),
    )

    with patch.object(
        asyncio,
        "create_subprocess_exec",
        new=AsyncMock(return_value=mock_process),
    ):
        result = await tool.execute(context, code="print(x)")

    assert not result.success
    assert "NameError" in (result.error or "")


@pytest.mark.asyncio
async def test_runner_with_vars(tool: PythonRunnerTool, context: ToolContext) -> None:
    """Variables passed via the `vars` parameter are available in the subprocess globals."""
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(
            json.dumps({
                "stdout": "42\n",
                "stderr": "",
                "return": "None",
                "error": None,
            }).encode(),
            b"",
        ),
    )

    with patch.object(
        asyncio,
        "create_subprocess_exec",
        new=AsyncMock(return_value=mock_process),
    ):
        result = await tool.execute(
            context,
            code="print(x)",
            vars=json.dumps({"x": 42}),
        )

    assert result.success
    assert "42" in result.output


@pytest.mark.asyncio
async def test_runner_invalid_vars_json(tool: PythonRunnerTool, context: ToolContext) -> None:
    """Invalid JSON in the `vars` parameter produces a clear error."""
    result = await tool.execute(context, code="print(1)", vars="not-json")
    assert not result.success
    assert "Invalid vars JSON" in (result.error or "")


@pytest.mark.asyncio
async def test_runner_gated_outside_development(tool: PythonRunnerTool) -> None:
    """Outside development with sandbox disabled, execution is refused."""
    settings = MagicMock(spec=Settings)
    settings.environment = Environment.PRODUCTION
    prod_context = ToolContext(
        conversation_id="test-conv",
        settings=settings,
    )

    result = await tool.execute(prod_context, code="print(1)")
    assert not result.success
    assert "disabled outside development" in (result.error or "").lower()
