"""n8n automation tool - drive an n8n instance via its CLI, REST API, and webhooks.

n8n is a workflow-automation platform (https://n8n.io). This tool lets Astra-X
create/import/export/execute workflows, inspect executions, trigger webhooks,
and call the public REST API of a running instance. A local checkout
(default ``C:\\Users\\Ashut\\Downloads\\n8n-master``) or a global ``n8n`` on PATH
provides the CLI; the server is reached over HTTP (default port 5678).
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.n8n.models import N8NConfig, N8NStatus
from app.tools.result import ToolResult

N8N_HOME_ENV = "N8N_HOME"
N8N_HOME_DEFAULT = r"C:\Users\Ashut\Downloads\n8n-master"
N8N_BASE_URL_ENV = "N8N_BASE_URL"
N8N_BASE_URL_DEFAULT = "http://127.0.0.1:5678"
N8N_API_KEY_ENV = "N8N_API_KEY"
N8N_CLI_PATH_ENV = "N8N_CLI_PATH"


class N8NTool(Tool):
    """n8n - workflow automation: import/export/execute workflows, inspect executions, trigger webhooks, and call the REST API."""

    @property
    def name(self) -> str:
        return "n8n"

    @property
    def description(self) -> str:
        return "n8n workflow automation: list/import/export/execute workflows, inspect executions, trigger webhooks, and call the n8n public REST API. Drives a local n8n instance (CLI + http://127.0.0.1:5678) for automation, scheduling, and integrations."

    @property
    def capabilities(self) -> list[str]:
        return ["automation", "workflow_orchestration", "webhooks", "task_scheduling"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action: get_status, list_workflows, get_workflow, import_workflow, export_workflow, execute, list_executions, get_execution, trigger_webhook, api_request, run_cli",
                    required=True,
                    enum=["get_status", "list_workflows", "get_workflow", "import_workflow", "export_workflow", "execute", "list_executions", "get_execution", "trigger_webhook", "api_request", "run_cli"],
                ),
                ToolParameter(
                    name="workflow_id",
                    type_="string",
                    description="Workflow ID (for get_workflow, execute, list_executions filter)",
                    required=False,
                ),
                ToolParameter(
                    name="workflow_file",
                    type_="string",
                    description="Path to a workflow JSON file (import_workflow / execute --file / export --output)",
                    required=False,
                ),
                ToolParameter(
                    name="execution_id",
                    type_="string",
                    description="Execution ID (for get_execution)",
                    required=False,
                ),
                ToolParameter(
                    name="webhook_path",
                    type_="string",
                    description="Webhook path appended to base_url/webhook (trigger_webhook), e.g. 'my-hook' or 'my-hook/param'",
                    required=False,
                ),
                ToolParameter(
                    name="payload",
                    type_="object",
                    description="JSON payload for trigger_webhook or api_request body",
                    required=False,
                    default={},
                ),
                ToolParameter(
                    name="method",
                    type_="string",
                    description="HTTP method for api_request (GET, POST, PUT, PATCH, DELETE)",
                    required=False,
                    default="GET",
                ),
                ToolParameter(
                    name="path",
                    type_="string",
                    description="API path for api_request relative to /api/v1, e.g. 'workflows' or 'workflows/<id>'",
                    required=False,
                ),
                ToolParameter(
                    name="n8n_command",
                    type_="string",
                    description="Raw n8n CLI arguments for run_cli, e.g. 'export:workflow --all'",
                    required=False,
                ),
                ToolParameter(
                    name="active",
                    type_="boolean",
                    description="Filter/set active workflows (list_workflows / export_workflow)",
                    required=False,
                ),
                ToolParameter(
                    name="all",
                    type_="boolean",
                    description="Export all workflows (export_workflow)",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="limit",
                    type_="integer",
                    description="Max items to return (list_workflows / list_executions)",
                    required=False,
                    default=50,
                ),
                ToolParameter(
                    name="base_url",
                    type_="string",
                    description="Override n8n base URL (default http://127.0.0.1:5678)",
                    required=False,
                ),
            ],
        )

    def __init__(self) -> None:
        self._home = self._find_home()
        found = self._find_n8n_cli()
        self._cli_path = found
        self._config = N8NConfig(
            base_url=os.environ.get(N8N_BASE_URL_ENV, N8N_BASE_URL_DEFAULT),
            api_key=os.environ.get(N8N_API_KEY_ENV, ""),
            cli_path=found,
            home=self._home,
        )

    def _find_home(self) -> str:
        """Locate an n8n checkout (env override, then the default path)."""
        candidates = [os.environ.get(N8N_HOME_ENV) or "", N8N_HOME_DEFAULT]
        for cand in candidates:
            if cand and (Path(cand) / "packages" / "cli" / "bin").is_dir():
                return cand
        return os.environ.get(N8N_HOME_ENV, "")

    def _find_n8n_cli(self) -> str:
        """Find a runnable n8n CLI: explicit env, local checkout, then PATH."""
        candidates: list[str] = []
        env = os.environ.get(N8N_CLI_PATH_ENV, "")
        if env:
            candidates.append(env)
        if self._home:
            candidates.append(str(Path(self._home) / "packages" / "cli" / "bin" / "n8n.cmd"))
            candidates.append(str(Path(self._home) / "packages" / "cli" / "bin" / "n8n"))
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            candidates.append(str(Path(appdata) / "npm" / "n8n.cmd"))
        candidates.extend(["n8n", "n8n.cmd"])
        for candidate in candidates:
            if os.path.sep in candidate or "/" in candidate:
                if os.path.isfile(candidate):
                    return candidate
            else:
                from shutil import which

                if which(candidate):
                    return which(candidate) or candidate
        return ""

    def _cli_prefix(self) -> list[str]:
        """Build the argv prefix needed to launch the n8n CLI on this platform."""
        cli = self._cli_path or "n8n"
        if cli.lower().endswith((".cmd", ".bat")) and os.name == "nt":
            return ["cmd", "/c", cli]
        return [cli]

    async def _run_cli(self, args: list[str], cwd: str = "", timeout_sec: float = 120.0) -> tuple[int, str, str]:
        prefix = self._cli_prefix()
        env = dict(os.environ)
        if self._home:
            env.setdefault(N8N_HOME_ENV, self._home)
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *prefix, *args,
                    cwd=cwd or (self._home or None),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout_sec,
            )
        except FileNotFoundError:
            return 127, "", f"n8n CLI not found (tried: {' '.join(prefix)})"
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except TimeoutError:
            proc.kill()
            return -1, "", f"n8n CLI timed out after {timeout_sec:.0f}s"
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace").strip(),
            stderr.decode("utf-8", errors="replace").strip(),
        )

    async def _http(self, method: str, url: str, body: dict | None = None, use_api_key: bool = False) -> tuple[int, str]:
        try:
            import httpx
        except ImportError:
            return 0, "httpx not installed; run: pip install httpx"
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if use_api_key and self._config.api_key:
            headers["X-N8N-API-KEY"] = self._config.api_key
        try:
            async with httpx.AsyncClient(timeout=self._config.timeout_sec) as client:
                resp = await client.request(method.upper(), url, json=body, headers=headers)
                return resp.status_code, resp.text
        except Exception as exc:
            return 0, f"request failed: {exc}"

    def _base_url(self, kwargs: dict) -> str:
        return (kwargs.get("base_url") or self._config.base_url).rstrip("/")

    async def _server_running(self, base_url: str) -> bool:
        status, _ = await self._http("GET", f"{base_url}/healthz")
        if status == 200:
            return True
        status, _ = await self._http("GET", f"{base_url}/rest/settings")
        return status in (200, 401, 403)

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        if action == "get_status":
            return await self._get_status(kwargs)
        if action == "list_workflows":
            return await self._list_workflows(kwargs)
        if action == "get_workflow":
            return await self._get_workflow(kwargs)
        if action == "import_workflow":
            return await self._import_workflow(kwargs)
        if action == "export_workflow":
            return await self._export_workflow(kwargs)
        if action == "execute":
            return await self._execute_workflow(kwargs)
        if action == "list_executions":
            return await self._list_executions(kwargs)
        if action == "get_execution":
            return await self._get_execution(kwargs)
        if action == "trigger_webhook":
            return await self._trigger_webhook(kwargs)
        if action == "api_request":
            return await self._api_request(kwargs)
        if action == "run_cli":
            return await self._run_cli_action(kwargs)
        return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _get_status(self, kwargs: dict) -> ToolResult:
        from shutil import which

        version = ""
        if self._cli_path:
            rc, out, err = await self._run_cli(["--version"], timeout_sec=30.0)
            version = (out or err).strip() if rc == 0 else ""
        base_url = self._base_url(kwargs)
        running = await self._server_running(base_url)
        status = N8NStatus(
            cli_found=bool(self._cli_path),
            cli_path=self._cli_path,
            version=version,
            node_found=bool(which("node")),
            server_running=running,
            base_url=base_url,
            api_key_set=bool(self._config.api_key),
            home=self._home,
        )
        hint = ""
        if not status.cli_found:
            hint = (
                "n8n CLI not found. Set N8N_CLI_PATH, install n8n globally "
                "(npm install -g n8n), or build the local checkout at "
                f"{self._home or N8N_HOME_DEFAULT}."
            )
        elif not status.server_running:
            hint = f"n8n server not reachable at {base_url}. Start it with action 'run_cli' command 'start'."
        return ToolResult(
            success=True,
            output=json.dumps({
                "cli_found": status.cli_found,
                "cli_path": status.cli_path,
                "version": status.version,
                "node_found": status.node_found,
                "server_running": status.server_running,
                "base_url": status.base_url,
                "api_key_set": status.api_key_set,
                "home": status.home,
                "install_hint": hint,
            }, indent=2),
        )

    async def _list_workflows(self, kwargs: dict) -> ToolResult:
        limit = int(kwargs.get("limit", 50))
        base_url = self._base_url(kwargs)
        if await self._server_running(base_url) and self._config.api_key:
            status, text = await self._http(
                "GET", f"{base_url}/api/v1/workflows?limit={limit}", use_api_key=True
            )
            if status == 200:
                return ToolResult(success=True, output=text[:12000], metadata={"source": "api", "status": status})
        if not self._cli_path:
            return ToolResult(success=False, error="n8n CLI not found and REST API unavailable (no running server / API key).")
        rc, out, err = await self._run_cli(["list:workflow"], timeout_sec=60.0)
        if rc == 0:
            return ToolResult(success=True, output=out[:12000] or "(no workflows)", metadata={"source": "cli"})
        return ToolResult(success=False, error=err or f"n8n CLI exited {rc}")

    async def _get_workflow(self, kwargs: dict) -> ToolResult:
        workflow_id = kwargs.get("workflow_id", "")
        if not workflow_id:
            return ToolResult(success=False, error="workflow_id is required for get_workflow")
        base_url = self._base_url(kwargs)
        if await self._server_running(base_url) and self._config.api_key:
            status, text = await self._http(
                "GET", f"{base_url}/api/v1/workflows/{workflow_id}", use_api_key=True
            )
            if status == 200:
                return ToolResult(success=True, output=text[:12000], metadata={"source": "api", "status": status})
            return ToolResult(success=False, error=f"n8n API returned {status}: {text[:2000]}")
        return ToolResult(success=False, error="get_workflow requires a running n8n server and N8N_API_KEY")

    async def _import_workflow(self, kwargs: dict) -> ToolResult:
        if not self._cli_path:
            return ToolResult(success=False, error="n8n CLI not found; cannot import workflow")
        workflow_file = kwargs.get("workflow_file", "")
        if not workflow_file:
            return ToolResult(success=False, error="workflow_file is required for import_workflow")
        rc, out, err = await self._run_cli(["import:workflow", f"--input={workflow_file}"], timeout_sec=120.0)
        if rc == 0:
            return ToolResult(success=True, output=out[:12000] or "workflow imported")
        return ToolResult(success=False, error=err or f"n8n CLI exited {rc}")

    async def _export_workflow(self, kwargs: dict) -> ToolResult:
        if not self._cli_path:
            return ToolResult(success=False, error="n8n CLI not found; cannot export workflow")
        args = ["export:workflow"]
        if kwargs.get("all"):
            args.append("--all")
        elif kwargs.get("workflow_id"):
            args.append(f"--id={kwargs['workflow_id']}")
        else:
            return ToolResult(success=False, error="export_workflow requires 'all' or 'workflow_id'")
        workflow_file = kwargs.get("workflow_file", "")
        if workflow_file:
            args.append(f"--output={workflow_file}")
        if kwargs.get("active") is not None:
            args.append(f"--onlyActive" if kwargs["active"] else "--onlyActive=false")
        rc, out, err = await self._run_cli(args, timeout_sec=120.0)
        if rc == 0:
            return ToolResult(success=True, output=out[:12000] or (f"exported to {workflow_file}" if workflow_file else "workflow exported"))
        return ToolResult(success=False, error=err or f"n8n CLI exited {rc}")

    async def _execute_workflow(self, kwargs: dict) -> ToolResult:
        if not self._cli_path:
            return ToolResult(success=False, error="n8n CLI not found; cannot execute workflow")
        args = ["execute"]
        if kwargs.get("workflow_file"):
            args.extend(["--file", kwargs["workflow_file"]])
        elif kwargs.get("workflow_id"):
            args.extend(["--id", kwargs["workflow_id"]])
        else:
            return ToolResult(success=False, error="execute requires 'workflow_id' or 'workflow_file'")
        rc, out, err = await self._run_cli(args, timeout_sec=300.0)
        if rc == 0:
            return ToolResult(success=True, output=out[:12000] or "(no output)")
        return ToolResult(success=False, error=err or f"n8n CLI exited {rc}")

    async def _list_executions(self, kwargs: dict) -> ToolResult:
        base_url = self._base_url(kwargs)
        limit = int(kwargs.get("limit", 50))
        if not self._config.api_key:
            return ToolResult(success=False, error="list_executions requires N8N_API_KEY and a running server")
        url = f"{base_url}/api/v1/executions?limit={limit}"
        if kwargs.get("workflow_id"):
            url += f"&workflowId={kwargs['workflow_id']}"
        status, text = await self._http("GET", url, use_api_key=True)
        if status == 200:
            return ToolResult(success=True, output=text[:12000], metadata={"status": status})
        return ToolResult(success=False, error=f"n8n API returned {status}: {text[:2000]}")

    async def _get_execution(self, kwargs: dict) -> ToolResult:
        execution_id = kwargs.get("execution_id", "")
        if not execution_id:
            return ToolResult(success=False, error="execution_id is required for get_execution")
        if not self._config.api_key:
            return ToolResult(success=False, error="get_execution requires N8N_API_KEY and a running server")
        base_url = self._base_url(kwargs)
        status, text = await self._http(
            "GET", f"{base_url}/api/v1/executions/{execution_id}?includeData=true", use_api_key=True
        )
        if status == 200:
            return ToolResult(success=True, output=text[:12000], metadata={"status": status})
        return ToolResult(success=False, error=f"n8n API returned {status}: {text[:2000]}")

    async def _trigger_webhook(self, kwargs: dict) -> ToolResult:
        webhook_path = (kwargs.get("webhook_path") or "").lstrip("/")
        if not webhook_path:
            return ToolResult(success=False, error="webhook_path is required for trigger_webhook")
        base_url = self._base_url(kwargs)
        payload = kwargs.get("payload") or None
        method = "GET" if payload is None else "POST"
        status, text = await self._http(method, f"{base_url}/webhook/{webhook_path}", body=payload)
        if status == 0:
            return ToolResult(success=False, error=text)
        ok = 200 <= status < 300
        return ToolResult(
            success=ok,
            output=text[:12000] or f"HTTP {status}",
            error=None if ok else f"n8n webhook returned {status}: {text[:2000]}",
            metadata={"status": status, "method": method, "webhook": webhook_path},
        )

    async def _api_request(self, kwargs: dict) -> ToolResult:
        path = (kwargs.get("path") or "").lstrip("/")
        if not path:
            return ToolResult(success=False, error="path is required for api_request")
        if not self._config.api_key:
            return ToolResult(success=False, error="api_request requires N8N_API_KEY and a running server")
        base_url = self._base_url(kwargs)
        method = (kwargs.get("method") or "GET").upper()
        payload = kwargs.get("payload") or None
        status, text = await self._http(method, f"{base_url}/api/v1/{path}", body=payload, use_api_key=True)
        ok = 200 <= status < 300
        return ToolResult(
            success=ok,
            output=text[:12000] or f"HTTP {status}",
            error=None if ok else f"n8n API returned {status}: {text[:2000]}",
            metadata={"status": status, "method": method, "path": path},
        )

    async def _run_cli_action(self, kwargs: dict) -> ToolResult:
        command = (kwargs.get("n8n_command") or "").strip()
        if not command:
            return ToolResult(success=False, error="n8n_command is required for run_cli")
        if not self._cli_path:
            return ToolResult(success=False, error="n8n CLI not found; set N8N_CLI_PATH or install n8n")
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return ToolResult(success=False, error=f"Invalid n8n_command quoting: {exc}")
        rc, out, err = await self._run_cli(parts, timeout_sec=300.0)
        if rc == 0:
            return ToolResult(
                success=True,
                output=out[:12000] or "(no output)",
                metadata={"exit_code": rc, "command": command},
            )
        return ToolResult(
            success=False,
            error=err or f"n8n CLI exited {rc}",
            output=out[:12000],
            metadata={"exit_code": rc, "command": command},
        )
