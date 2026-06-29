"""Run Python code in a subprocess with a timeout.

WARNING: This tool executes arbitrary Python code.  Enable only in
sandboxed environments or when the caller is fully trusted.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from app.config.settings import Environment
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.python.sandbox import SandboxConfig
from app.tools.result import ToolResult


class PythonRunnerTool(Tool):
    @property
    def name(self) -> str:
        return "python_repl"

    @property
    def description(self) -> str:
        return "Execute Python code and return stdout, stderr, and the return value. Supports passing input variables."

    @property
    def capabilities(self) -> list[str]:
        return ["execute_python"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(name="code", type_="string", description="Python code to execute", required=True),
                ToolParameter(name="timeout", type_="integer", description="Execution timeout in seconds", required=False, default=10),
                ToolParameter(name="vars", type_="string", description="JSON object of variables to inject into the global scope", required=False),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        code: str = kwargs.get("code", "")
        timeout: int = int(kwargs.get("timeout", 10))
        vars_json: str | None = kwargs.get("vars")

        if not code:
            return ToolResult(success=False, error="code is required")
        if timeout < 1 or timeout > 120:
            return ToolResult(success=False, error="timeout must be between 1 and 120")

        # Sandbox gate: refuse unsandboxed execution outside development.
        env = getattr(context.settings, "environment", None)
        sandbox = SandboxConfig()
        if env is not Environment.DEVELOPMENT and not sandbox.enabled:
            return ToolResult(
                success=False,
                error=(
                    "Python execution is disabled outside development without a "
                    "sandbox. Set environment=development or configure a sandbox."
                ),
            )

        import_vars: dict[str, Any] = {}
        if vars_json:
            try:
                import_vars = json.loads(vars_json)
            except json.JSONDecodeError as exc:
                return ToolResult(success=False, error=f"Invalid vars JSON: {exc}")

        runner_code = "\n".join([
            "import json, sys, traceback",
            "_code_ = sys.stdin.read()",
            "_result = {'stdout': '', 'stderr': '', 'return': None, 'error': None}",
            "try:",
            "    import io",
            "    _stdout = io.StringIO()",
            "    _stderr = io.StringIO()",
            "    sys.stdout = _stdout",
            "    sys.stderr = _stderr",
            "    _locals = {}",
            f"    _globals = {json.dumps(import_vars)}",
            "    exec(compile(_code_, '<python_repl>', 'exec'), _globals, _locals)",
            "    _result['stdout'] = _stdout.getvalue()",
            "    _result['stderr'] = _stderr.getvalue()",
            "    _result['return'] = str(_locals.get('_return', None))",
            "except BaseException:",
            "    _result['error'] = traceback.format_exc()",
            "finally:",
            "    sys.stdout = sys.__stdout__",
            "    sys.stderr = sys.__stderr__",
            "print(json.dumps(_result))",
        ])

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                runner_code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=code.encode("utf-8")),
                    timeout=timeout,
                )
            except TimeoutError:
                proc.kill()
                return ToolResult(success=False, error=f"Execution timed out after {timeout}s")
        except OSError as exc:
            return ToolResult(success=False, error=str(exc))

        if proc.returncode != 0:
            return ToolResult(
                success=False,
                error=f"Process exited with code {proc.returncode}",
                metadata={"stderr": stderr.decode("utf-8", errors="replace")},
            )

        try:
            result_data = json.loads(stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return ToolResult(
                success=False,
                error="Failed to parse REPL output",
                metadata={"raw_stdout": stdout.decode("utf-8", errors="replace")},
            )

        output_lines = []
        if result_data.get("stdout"):
            output_lines.append(result_data["stdout"])
        if result_data.get("return") and result_data["return"] != "None":
            output_lines.append(f"Return value: {result_data['return']}")

        combined = "\n".join(output_lines) if output_lines else "(no output)"

        if result_data.get("error"):
            return ToolResult(
                success=False,
                error=result_data["error"],
                output=combined,
                metadata={"stderr": result_data.get("stderr", "")},
            )

        return ToolResult(
            success=True,
            output=combined,
            metadata={"stderr": result_data.get("stderr", ""), "return": result_data.get("return")},
        )
