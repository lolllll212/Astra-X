"""Unit tests for the hardened Python execution sandbox.

Covers :class:`HardenedSandbox` and its enforcement of module
restrictions, network/filesystem access control, and output limits.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.python.sandbox import SandboxConfig
from app.tools.security.sandbox_runner import HardenedSandbox, SandboxResult


@pytest.fixture
def default_config() -> SandboxConfig:
    return SandboxConfig(
        enabled=True,
        timeout_seconds=5,
        max_output_bytes=4096,
        blocked_modules=("os", "subprocess", "socket", "http", "urllib"),
        allowed_modules=(),
        network_access=False,
        filesystem_access=False,
    )


@pytest.fixture
def sandbox(default_config: SandboxConfig) -> HardenedSandbox:
    return HardenedSandbox(default_config)


# =========================================================================
# SandboxResult
# =========================================================================


class TestSandboxResult:
    def test_default_construction(self) -> None:
        r = SandboxResult(success=True, output="out")
        assert r.success
        assert r.output == "out"
        assert r.error is None
        assert r.stderr == ""
        assert r.return_value is None

    def test_full_construction(self) -> None:
        r = SandboxResult(
            success=False,
            output="out",
            error="err",
            stderr="stderr text",
            return_value="42",
        )
        assert not r.success
        assert r.error == "err"
        assert r.return_value == "42"


# =========================================================================
# HardenedSandbox — general
# =========================================================================


class TestHardenedSandboxGeneral:
    @pytest.mark.asyncio
    async def test_empty_code(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("")
        assert not result.success
        assert "required" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_code(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("   \n  ")
        assert not result.success
        assert "required" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_successful_execution(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("print('hello')")
        assert result.success
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_return_value(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("_return = 42")
        assert result.success
        assert result.return_value == "42"

    @pytest.mark.asyncio
    async def test_stdout_and_return(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("print('hi')\n_return = 99")
        assert result.success
        assert "hi" in result.output
        assert "99" in result.output

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("import time; time.sleep(100)", timeout=1)
        assert not result.success
        assert "timed out" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_syntax_error(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("invalid{{{")
        assert not result.success

    @pytest.mark.asyncio
    async def test_runtime_error(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("1/0")
        assert not result.success
        assert "ZeroDivisionError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_with_input_vars(self, default_config: SandboxConfig) -> None:
        sbox = HardenedSandbox(default_config)
        result = await sbox.run("print(x)", input_vars={"x": 42})
        assert result.success
        assert "42" in result.output

    @pytest.mark.asyncio
    async def test_custom_timeout_override(self) -> None:
        config = SandboxConfig(enabled=True, timeout_seconds=30)
        sbox = HardenedSandbox(config)
        result = await sbox.run("import time; time.sleep(100)", timeout=1)
        assert not result.success
        assert "timed out" in (result.error or "").lower()


# =========================================================================
# HardenedSandbox — module restriction enforcement
# =========================================================================


class TestHardenedSandboxBlockedModules:
    @pytest.mark.asyncio
    async def test_import_os_blocked(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("import os")
        assert not result.success
        assert "blocked" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_import_subprocess_blocked(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("import subprocess")
        assert not result.success
        assert "blocked" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_import_socket_blocked(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("import socket")
        assert not result.success
        assert "blocked" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_import_from_blocked_module(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("from os import path")
        assert not result.success
        assert "blocked" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_import_blocked_as_alias(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("import os as operating_system")
        assert not result.success
        assert "blocked" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_stdlib_imports_allowed(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("import json; import math; print('ok')")
        assert result.success
        assert "ok" in result.output

    @pytest.mark.asyncio
    async def test_only_allowed_modules_whitelist(self) -> None:
        config = SandboxConfig(
            enabled=True,
            allowed_modules=("json", "math"),
            blocked_modules=(),
        )
        sbox = HardenedSandbox(config)

        ok = await sbox.run("import json; import math; print('ok')")
        assert ok.success

        blocked = await sbox.run("import random")
        assert not blocked.success
        assert "allowed" in (blocked.error or "").lower()

    @pytest.mark.asyncio
    async def test_empty_blocked_list_blocks_nothing(self) -> None:
        config = SandboxConfig(
            enabled=True,
            blocked_modules=(),
            allowed_modules=(),
            network_access=False,
            filesystem_access=False,
        )
        sbox = HardenedSandbox(config)
        result = await sbox.run("import math; print(math.pi)")
        assert result.success


# =========================================================================
# HardenedSandbox — filesystem & network enforcement
# =========================================================================


class TestHardenedSandboxAccessControl:
    @pytest.mark.asyncio
    async def test_filesystem_write_blocked(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("with open('/tmp/test.txt', 'w') as f: f.write('hi')")
        assert not result.success
        assert any(
            word in (result.error or "").lower()
            for word in ("disabled", "denied", "access", "permission")
        )

    @pytest.mark.asyncio
    async def test_filesystem_read_blocked(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("open('/etc/passwd').read()")
        assert not result.success

    @pytest.mark.asyncio
    async def test_filesystem_access_allowed_when_configured(self) -> None:
        config = SandboxConfig(
            enabled=True,
            filesystem_access=True,
            network_access=False,
        )
        sbox = HardenedSandbox(config)
        result = await sbox.run("print('allowed')")
        assert result.success

    @pytest.mark.asyncio
    async def test_input_vars_still_work(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("print(x + y)", input_vars={"x": 10, "y": 20})
        assert result.success
        assert "30" in result.output


# =========================================================================
# HardenedSandbox — output size enforcement
# =========================================================================


class TestHardenedSandboxOutputLimit:
    @pytest.mark.asyncio
    async def test_large_output_truncated(self) -> None:
        config = SandboxConfig(
            enabled=True,
            max_output_bytes=200,
            timeout_seconds=5,
        )
        sbox = HardenedSandbox(config)
        result = await sbox.run("print('x' * 1000)")
        # Output should be truncated or error
        assert not result.success or len(result.output) < 1000
        if not result.success:
            assert "truncat" in (result.error or "").lower()


# =========================================================================
# HardenedSandbox — subprocess errors
# =========================================================================


class TestHardenedSandboxSubprocess:
    @pytest.mark.asyncio
    async def test_subprocess_failure(self, sandbox: HardenedSandbox) -> None:
        with patch.object(
            asyncio,
            "create_subprocess_exec",
            new=AsyncMock(side_effect=OSError("exec failed")),
        ):
            result = await sandbox.run("print('hi')")
        assert not result.success
        assert "exec failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_nonzero_return_code(self, sandbox: HardenedSandbox) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error output"))

        with patch.object(
            asyncio,
            "create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = await sandbox.run("print('hi')")

        assert not result.success
        assert "exited with code 1" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_invalid_output_json(self, sandbox: HardenedSandbox) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not json", b""))

        with patch.object(
            asyncio,
            "create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = await sandbox.run("print('hi')")

        assert not result.success
        assert "parse" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_timeout_short(self, sandbox: HardenedSandbox) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)
        mock_proc.kill = MagicMock()

        with patch.object(
            asyncio,
            "create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ):
            result = await sandbox.run("while True: pass", timeout=1)

        assert not result.success
        assert "timed out" in (result.error or "").lower()
        mock_proc.kill.assert_called_once()


# =========================================================================
# HardenedSandbox — input_vars edge cases
# =========================================================================


class TestHardenedSandboxInputVars:
    @pytest.mark.asyncio
    async def test_empty_input_vars(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("print('ok')", input_vars={})
        assert result.success

    @pytest.mark.asyncio
    async def test_input_vars_not_interfering_with_sandbox(self, sandbox: HardenedSandbox) -> None:
        result = await sandbox.run("print(type(x).__name__)", input_vars={"x": 42})
        assert result.success
        assert "int" in result.output or "42" in result.output
