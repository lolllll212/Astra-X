"""Hardened sandbox runner for secure Python code execution in a subprocess.

The :class:`HardenedSandbox` enforces the constraints declared in
:class:`~app.tools.python.sandbox.SandboxConfig` by generating an
instrumented runner script that intercepts imports, overrides builtins,
and caps output.  All constraints are applied *inside* the subprocess
at the Python level.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any

from app.tools.errors import SandboxViolation
from app.tools.python.sandbox import SandboxConfig

SANDBOX_HEADER = r'''
import json as _json, sys as _sys, traceback as _tb, io as _io, builtins as _builtins_mod

_StringIO = _io.StringIO
_sys_stdin_read = _sys.stdin.read
_sys_stdout = _sys.stdout
_sys_stderr = _sys.stderr
_tb_format_exc = _tb.format_exc
_json_dumps = _json.dumps
_json_loads = _json.loads
_sorted = sorted
_str = str
_len = len

SANDBOX_BLOCKED = {blocked!r}
SANDBOX_ALLOWED = {allowed!r}
SANDBOX_NETWORK = {network!r}
SANDBOX_FILESYSTEM = {filesystem!r}
SANDBOX_MAX_OUTPUT = {max_output!r}

_sandbox_original_import = _builtins_mod.__import__

def _sandbox_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base in SANDBOX_BLOCKED:
        raise ImportError(
            "Module '" + name + "' is blocked by the security sandbox. "
            "Blocked modules: " + _str(_sorted(SANDBOX_BLOCKED))
        )
    if SANDBOX_ALLOWED and base not in SANDBOX_ALLOWED:
        raise ImportError(
            "Module '" + name + "' is not in the allowed modules list. "
            "Allowed modules: " + _str(_sorted(SANDBOX_ALLOWED))
        )
    return _sandbox_original_import(name, *args, **kwargs)

_builtins_mod.__import__ = _sandbox_import

class _SandboxSocketPlaceholder:
    def __getattr__(self, _name):
        raise RuntimeError("Network access is disabled by the security sandbox")

# Network / filesystem access control
if not SANDBOX_NETWORK:
    _sys.modules['socket'] = _SandboxSocketPlaceholder()
    _sys.modules['http'] = _SandboxSocketPlaceholder()
    _sys.modules['urllib'] = _SandboxSocketPlaceholder()

    _original_open = _builtins_mod.open
    def _sandbox_open(*args, **kwargs):
        if not SANDBOX_FILESYSTEM:
            permission = "Network" if 'http' in _str(args[0]) or '://' in _str(args[0]) else "Filesystem"
            raise PermissionError(
                permission + " access is disabled by the security sandbox"
            )
        return _original_open(*args, **kwargs)
    _builtins_mod.open = _sandbox_open

if not SANDBOX_FILESYSTEM:
    _builtins_mod.open = lambda *args, **kwargs: (_ for _ in ()).throw(
        PermissionError("Filesystem access is disabled by the security sandbox")
    )

_code_ = _sys_stdin_read()
_result = {{'stdout': '', 'stderr': '', 'return': None, 'error': None}}
try:
    _stdout = _StringIO()
    _stderr = _StringIO()
    _sys.stdout = _stdout
    _sys.stderr = _stderr
    _locals = {{}}
    _globals = {vars}
    for _key in ('exec', 'eval', 'compile', 'open'):
        _globals.pop(_key, None)
    exec(compile(_code_, '<sandbox>', 'exec'), _globals, _locals)
    _result['stdout'] = _stdout.getvalue()
    _result['stderr'] = _stderr.getvalue()
    _result['return'] = _str(_locals.get('_return', None))
except BaseException:
    _result['error'] = _tb_format_exc()
finally:
    _sys.stdout = _sys_stdout
    _sys.stderr = _sys_stderr

_output = _json_dumps(_result)
if _len(_output) > SANDBOX_MAX_OUTPUT:
    _result['stdout'] = _result.get('stdout', '')[:SANDBOX_MAX_OUTPUT // 2]
    _result['stderr'] = _result.get('stderr', '')[:SANDBOX_MAX_OUTPUT // 2]
    if not _result.get('error'):
        _result['error'] = 'Output truncated (exceeded ' + _str(SANDBOX_MAX_OUTPUT) + ' bytes)'
    _output = _json_dumps(_result)
print(_output)
'''


@dataclass(frozen=True)
class SandboxResult:
    """Result of a sandboxed Python execution.

    Attributes:
        success: Whether execution completed without errors.
        output: Captured stdout text.
        error: Error message if execution failed.
        stderr: Captured stderr text (from the subprocess itself).
        return_value: The value of ``_return`` if set by user code.
    """

    success: bool
    output: str
    error: str | None = None
    stderr: str = ""
    return_value: str | None = None


class HardenedSandbox:
    """Enforces :class:`SandboxConfig` constraints on Python subprocess execution.

    Usage::

        config = SandboxConfig(enabled=True, blocked_modules=("os",), network_access=False)
        sandbox = HardenedSandbox(config)
        result = await sandbox.run("print('hello')", timeout=10)
        print(result.output)
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    async def run(
        self,
        code: str,
        *,
        input_vars: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> SandboxResult:
        """Execute *code* inside a subprocess with sandbox constraints.

        Args:
            code: The Python source to execute.
            input_vars: Variables to inject into the global scope of the
                executed code.
            timeout: Execution timeout in seconds. Falls back to the
                config default if ``None``.

        Returns:
            The sandbox execution result.
        """
        if not code.strip():
            return SandboxResult(success=False, output="", error="code is required")

        effective_timeout = timeout if timeout is not None else self._config.timeout_seconds
        if effective_timeout < 1:
            effective_timeout = self._config.timeout_seconds
        if effective_timeout > 120:
            effective_timeout = self._config.timeout_seconds

        runner_code = SANDBOX_HEADER.format(
            blocked=self._config.blocked_modules,
            allowed=self._config.allowed_modules,
            network=self._config.network_access,
            filesystem=self._config.filesystem_access,
            max_output=self._config.max_output_bytes,
            vars=json.dumps(input_vars or {}),
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                runner_code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return SandboxResult(success=False, output="", error=str(exc))

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=code.encode("utf-8")),
                timeout=effective_timeout,
            )
        except TimeoutError:
            proc.kill()
            return SandboxResult(
                success=False,
                output="",
                error=f"Execution timed out after {effective_timeout}s",
            )

        stderr_text = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0 and proc.returncode is not None:
            return SandboxResult(
                success=False,
                output="",
                error=f"Process exited with code {proc.returncode}",
                stderr=stderr_text,
            )

        try:
            result_data = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return SandboxResult(
                success=False,
                output="",
                error="Failed to parse REPL output",
                stderr=stderr_text,
            )

        output_lines = []
        if result_data.get("stdout"):
            output_lines.append(result_data["stdout"])
        return_val = result_data.get("return")
        if return_val and return_val != "None":
            output_lines.append(f"Return value: {return_val}")

        combined = "\n".join(output_lines) if output_lines else "(no output)"

        if result_data.get("error"):
            return SandboxResult(
                success=False,
                error=result_data["error"],
                output=combined,
                stderr=result_data.get("stderr", ""),
                return_value=return_val if return_val != "None" else None,
            )

        return SandboxResult(
            success=True,
            output=combined,
            stderr=stderr_text,
            return_value=return_val if return_val != "None" else None,
        )


def _build_blocked_import_checker(
    blocked_modules: tuple[str, ...],
    allowed_modules: tuple[str, ...],
) -> str:
    """Generate Python code that installs a guarded ``__import__``.

    This is an internal helper used when building custom runner scripts.
    The returned string is valid Python that, when executed in a
    subprocess, overrides the builtin ``__import__`` to enforce the given
    module allow/deny lists.
    """
    lines = [
        "_sandbox_original_import = __builtins__.__import__ if isinstance(__builtins__, dict) else __builtins__.__import__",
        "",
        "def _sandbox_import(name, *args, **kwargs):",
        "    base = name.split('.')[0]",
    ]

    blocked_json = json.dumps(list(blocked_modules))
    allowed_json = json.dumps(list(allowed_modules))

    lines.append(f"    if base in {blocked_json}:")
    lines.append(f'    raise ImportError(f"Module " + name + " is blocked by the sandbox")')

    if allowed_modules:
        lines.append(f"    if {allowed_json} and base not in {allowed_json}:")
        lines.append(f'    raise ImportError(f"Module " + name + " is not in allowed modules")')

    lines.append("    return _sandbox_original_import(name, *args, **kwargs)")
    lines.append("")
    lines.append(
        "_builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__"
    )
    lines.append("_builtins['__import__'] = _sandbox_import")

    return "\n".join(lines)
