"""Run Python code in a subprocess with sandbox enforcement.

Uses :class:`~app.tools.security.sandbox_runner.HardenedSandbox` to
enforce module restrictions, network/filesystem access control, and
output limits defined in :class:`~app.tools.python.sandbox.SandboxConfig`.

WARNING: Enabling Python execution without a sandbox is dangerous.
Outside development, execution is blocked unless at least one sandbox
policy is enforced.
"""

from __future__ import annotations

import json
from typing import Any

from app.config.settings import Environment
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.python.sandbox import SandboxConfig
from app.tools.result import ToolResult
from app.tools.security.sandbox_runner import HardenedSandbox


class PythonRunnerTool(Tool):
    """Execute Python code in a sandboxed subprocess.

    The tool enforces module restrictions, network/filesystem access
    control, and output limits via :class:`HardenedSandbox`.  Outside
    development environments the sandbox must be explicitly enabled.
    """

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

        sandbox_runner = HardenedSandbox(sandbox)
        result = await sandbox_runner.run(
            code,
            input_vars=import_vars,
            timeout=timeout,
        )

        if not result.success:
            return ToolResult(
                success=False,
                error=result.error or "Unknown error",
                output=result.output,
                metadata={"stderr": result.stderr} if result.stderr else None,
            )

        return ToolResult(
            success=True,
            output=result.output,
            metadata={"stderr": result.stderr, "return": result.return_value} if result.return_value else None,
        )
