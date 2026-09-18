"""BrowserSkill automation tool - controls browser via bsk CLI, daemon, and extension."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any

from app.tools.base import Tool
from app.tools.browserskill.models import (
    BrowserAction,
    BrowserActionResult,
    BrowserSession,
    BrowserSkillCommand,
    BSKConfig,
    HumanInLoopReason,
    HumanInLoopRequest,
)
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class BrowserSkillTool(Tool):
    """BrowserSkill automation - reuse real login state, keep working uninterrupted, support any agent, built-in human-in-loop."""

    @property
    def name(self) -> str:
        return "browserskill"

    @property
    def description(self) -> str:
        return "Automate browser tasks using BrowserSkill: bsk CLI, daemon, and browser extension. Reuse real login state, run in separate Agent Window, human-in-loop for captcha/login/confirmations."

    @property
    def capabilities(self) -> list[str]:
        return ["browser_automation", "web_interaction", "human_in_loop", "session_management"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action to perform: start_session, execute_action, execute_sequence, get_session, close_session, request_human_help, resolve_human_help, get_status",
                    required=True,
                    enum=["start_session", "execute_action", "execute_sequence", "get_session", "close_session", "request_human_help", "resolve_human_help", "get_status"],
                ),
                ToolParameter(
                    name="session_id",
                    type_="string",
                    description="Existing session ID (for actions on existing session)",
                    required=False,
                ),
                ToolParameter(
                    name="config",
                    type_="object",
                    description="BSKConfig for new session: daemon_port, headless, user_data_dir, auto_connect",
                    required=False,
                ),
                ToolParameter(
                    name="command",
                    type_="string",
                    description="Browser command: navigate, click, type, select, scroll, wait, screenshot, extract, execute_js, get_cookies, set_cookies, tab_new, tab_close, tab_switch, tab_list, download, upload, hover, drag_drop, press_key, get_text, get_html, get_attribute, set_attribute, evaluate",
                    required=False,
                ),
                ToolParameter(
                    name="selector",
                    type_="string",
                    description="CSS/XPath selector for element interactions",
                    required=False,
                ),
                ToolParameter(
                    name="value",
                    type_="string",
                    description="Value for type, select, or other input commands",
                    required=False,
                ),
                ToolParameter(
                    name="url",
                    type_="string",
                    description="URL for navigate command",
                    required=False,
                ),
                ToolParameter(
                    name="options",
                    type_="object",
                    description="Additional options for the command",
                    required=False,
                    default={},
                ),
                ToolParameter(
                    name="tab_id",
                    type_="string",
                    description="Target tab ID (uses active tab if not specified)",
                    required=False,
                ),
                ToolParameter(
                    name="timeout_ms",
                    type_="integer",
                    description="Command timeout in milliseconds",
                    required=False,
                    default=30000,
                ),
                ToolParameter(
                    name="sequence",
                    type_="array",
                    description="Array of actions for execute_sequence",
                    required=False,
                ),
                ToolParameter(
                    name="human_in_loop_reason",
                    type_="string",
                    description="Reason for human-in-loop: captcha, login, confirmation_dialog, two_factor, custom",
                    required=False,
                ),
                ToolParameter(
                    name="human_in_loop_message",
                    type_="string",
                    description="Message to show to human",
                    required=False,
                ),
                ToolParameter(
                    name="human_in_loop_context",
                    type_="object",
                    description="Additional context for human-in-loop request",
                    required=False,
                    default={},
                ),
                ToolParameter(
                    name="resolution",
                    type_="string",
                    description="Human resolution for human-in-loop request",
                    required=False,
                ),
            ],
        )

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}
        found = self._find_bsk_cli()
        self._bsk_found = found is not None
        self._bsk_cli_path = found or "bsk"  # Fallback to PATH
        self._bsk_version = ""

    def _find_bsk_cli(self) -> str | None:
        """Find bsk CLI executable; return None when not installed."""
        # Check common locations
        candidates = [
            "bsk",
            "bsk.exe",
            os.path.expanduser("~/.bsk/bin/bsk"),
            os.path.expanduser("~/.bsk/bin/bsk.exe"),
            "/usr/local/bin/bsk",
            "/opt/bsk/bin/bsk",
        ]
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    self._bsk_version = result.stdout.decode("utf-8", errors="replace").strip()
                    return candidate
            except Exception:
                continue
        return None

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")

        if action == "start_session":
            return await self._start_session(context, kwargs)
        elif action == "execute_action":
            return await self._execute_action(context, kwargs)
        elif action == "execute_sequence":
            return await self._execute_sequence(context, kwargs)
        elif action == "get_session":
            return await self._get_session(context, kwargs)
        elif action == "close_session":
            return await self._close_session(context, kwargs)
        elif action == "request_human_help":
            return await self._request_human_help(context, kwargs)
        elif action == "resolve_human_help":
            return await self._resolve_human_help(context, kwargs)
        elif action == "get_status":
            return await self._get_status(context, kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _get_status(self, context: ToolContext, kwargs: dict) -> ToolResult:
        daemon_running = False
        if self._bsk_found:
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [self._bsk_cli_path, "daemon", "status"],
                    capture_output=True,
                    timeout=5,
                )
                daemon_running = result.returncode == 0
            except Exception:
                daemon_running = False
        return ToolResult(
            success=True,
            output=json.dumps({
                "bsk_found": self._bsk_found,
                "bsk_path": self._bsk_cli_path,
                "bsk_version": self._bsk_version,
                "daemon_running": daemon_running,
                "active_sessions": len(self._sessions),
                "install_hint": "" if self._bsk_found else (
                    "bsk CLI not found. Install BrowserSkill (bsk) and ensure it is on PATH "
                    "or at ~/.bsk/bin/bsk, then retry."
                ),
            }, indent=2),
        )

    async def _start_session(self, context: ToolContext, kwargs: dict) -> ToolResult:
        if not self._bsk_found:
            return ToolResult(
                success=False,
                error=(
                    "bsk CLI not found: browser automation requires the BrowserSkill (bsk) "
                    "daemon/CLI installed and on PATH (or ~/.bsk/bin/bsk). "
                    "Run action 'get_status' to check availability."
                ),
            )
        config_data = kwargs.get("config", {})
        config = BSKConfig(
            daemon_port=config_data.get("daemon_port", 9222),
            headless=config_data.get("headless", False),
            user_data_dir=config_data.get("user_data_dir", ""),
            auto_connect=config_data.get("auto_connect", True),
            request_timeout_ms=config_data.get("request_timeout_ms", 60000),
            max_retries=config_data.get("max_retries", 3),
        )

        session = BrowserSession(config=config)

        # Start bsk daemon if not running
        daemon_started = await self._start_daemon(config)
        if not daemon_started:
            return ToolResult(
                success=False,
                error=(
                    f"Failed to start bsk daemon via '{self._bsk_cli_path}'. "
                    "Ensure the BrowserSkill daemon is installed and the browser extension is available."
                ),
            )

        # Connect to extension
        connected = await self._connect_extension(config)
        session.extension_connected = connected

        self._sessions[session.id] = session

        return ToolResult(
            success=True,
            output=f"BrowserSkill session started: {session.id}",
            metadata={
                "session_id": session.id,
                "daemon_port": config.daemon_port,
                "extension_connected": connected,
            },
        )

    async def _start_daemon(self, config: BSKConfig) -> bool:
        """Start bsk daemon if not already running."""
        try:
            # Check if daemon is already running
            result = await asyncio.to_thread(
                subprocess.run,
                [self._bsk_cli_path, "daemon", "status"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True

            # Start daemon
            cmd = [self._bsk_cli_path, "daemon", "start", "--port", str(config.daemon_port)]
            if config.headless:
                cmd.append("--headless")
            if config.user_data_dir:
                cmd.extend(["--user-data-dir", config.user_data_dir])

            await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait a moment for daemon to start
            await asyncio.sleep(2)

            # Verify it's running
            result = await asyncio.to_thread(
                subprocess.run,
                [self._bsk_cli_path, "daemon", "status"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _connect_extension(self, config: BSKConfig) -> bool:
        """Connect to BrowserSkill browser extension."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [self._bsk_cli_path, "extension", "connect", "--port", str(config.daemon_port)],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _execute_action(self, context: ToolContext, kwargs: dict) -> ToolResult:
        session_id = kwargs.get("session_id")
        if not session_id or session_id not in self._sessions:
            return ToolResult(success=False, error="Invalid or missing session_id")

        session = self._sessions[session_id]

        command_str = kwargs.get("command", "")
        try:
            command = BrowserSkillCommand(command_str)
        except ValueError:
            return ToolResult(success=False, error=f"Unknown command: {command_str}")

        action = BrowserAction(
            command=command,
            selector=kwargs.get("selector"),
            value=kwargs.get("value"),
            url=kwargs.get("url"),
            options=kwargs.get("options", {}),
            tab_id=kwargs.get("tab_id") or session.active_tab_id,
            timeout_ms=kwargs.get("timeout_ms", 30000),
        )

        result = await self._run_browser_action(session, action)
        session.last_activity = datetime.utcnow()

        return ToolResult(
            success=result.success,
            output=json.dumps(result.output) if result.output else "",
            error=result.error,
            metadata={
                "action_id": action.id,
                "duration_ms": result.duration_ms,
                "session_id": session_id,
            },
        )

    async def _run_browser_action(self, session: BrowserSession, action: BrowserAction) -> BrowserActionResult:
        """Execute a single browser action via bsk CLI."""
        start_time = time.monotonic()

        try:
            cmd = [self._bsk_cli_path, "action", action.command.value]

            if action.selector:
                cmd.extend(["--selector", action.selector])
            if action.value:
                cmd.extend(["--value", action.value])
            if action.url:
                cmd.extend(["--url", action.url])
            if action.tab_id:
                cmd.extend(["--tab", action.tab_id])
            if action.options:
                cmd.extend(["--options", json.dumps(action.options)])
            cmd.extend(["--port", str(session.config.daemon_port)])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=action.timeout_ms / 1000,
            )

            duration_ms = int((time.monotonic() - start_time) * 1000)

            if process.returncode == 0:
                output = json.loads(stdout.decode()) if stdout else None
                return BrowserActionResult(
                    action_id=action.id,
                    success=True,
                    output=output,
                    duration_ms=duration_ms,
                )
            else:
                error = stderr.decode() if stderr else f"Command failed with code {process.returncode}"
                return BrowserActionResult(
                    action_id=action.id,
                    success=False,
                    error=error,
                    duration_ms=duration_ms,
                )

        except TimeoutError:
            return BrowserActionResult(
                action_id=action.id,
                success=False,
                error=f"Action timed out after {action.timeout_ms}ms",
                duration_ms=action.timeout_ms,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return BrowserActionResult(
                action_id=action.id,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def _execute_sequence(self, context: ToolContext, kwargs: dict) -> ToolResult:
        session_id = kwargs.get("session_id")
        if not session_id or session_id not in self._sessions:
            return ToolResult(success=False, error="Invalid or missing session_id")

        session = self._sessions[session_id]
        sequence = kwargs.get("sequence", [])

        results = []
        for action_data in sequence:
            action = BrowserAction(
                command=BrowserSkillCommand(action_data.get("command", "navigate")),
                selector=action_data.get("selector"),
                value=action_data.get("value"),
                url=action_data.get("url"),
                options=action_data.get("options", {}),
                tab_id=action_data.get("tab_id") or session.active_tab_id,
                timeout_ms=action_data.get("timeout_ms", 30000),
            )

            result = await self._run_browser_action(session, action)
            results.append({
                "action_id": action.id,
                "command": action.command.value,
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "duration_ms": result.duration_ms,
            })

            if not result.success and action_data.get("stop_on_failure", True):
                break

            session.last_activity = datetime.utcnow()

        overall_success = all(r["success"] for r in results)

        return ToolResult(
            success=overall_success,
            output=json.dumps(results),
            metadata={"session_id": session_id, "actions_executed": len(results)},
        )

    async def _get_session(self, context: ToolContext, kwargs: dict) -> ToolResult:
        session_id = kwargs.get("session_id")
        if not session_id or session_id not in self._sessions:
            return ToolResult(success=False, error="Invalid or missing session_id")

        session = self._sessions[session_id]
        return ToolResult(
            success=True,
            output=json.dumps({
                "session_id": session.id,
                "tabs": [{"id": t.id, "url": t.url, "title": t.title, "window_type": t.window_type.value, "is_active": t.is_active} for t in session.tabs],
                "active_tab_id": session.active_tab_id,
                "extension_connected": session.extension_connected,
                "started_at": session.started_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
            }),
        )

    async def _close_session(self, context: ToolContext, kwargs: dict) -> ToolResult:
        session_id = kwargs.get("session_id")
        if not session_id or session_id not in self._sessions:
            return ToolResult(success=False, error="Invalid or missing session_id")

        # Close all tabs via bsk
        await asyncio.to_thread(
            subprocess.run,
            [self._bsk_cli_path, "session", "close", "--session", session_id],
            capture_output=True,
            timeout=10,
        )

        del self._sessions[session_id]

        return ToolResult(success=True, output=f"Session {session_id} closed")

    async def _request_human_help(self, context: ToolContext, kwargs: dict) -> ToolResult:
        session_id = kwargs.get("session_id")
        if not session_id or session_id not in self._sessions:
            return ToolResult(success=False, error="Invalid or missing session_id")

        session = self._sessions[session_id]

        reason_str = kwargs.get("human_in_loop_reason", "custom")
        try:
            reason = HumanInLoopReason(reason_str)
        except ValueError:
            reason = HumanInLoopReason.CUSTOM

        request = HumanInLoopRequest(
            reason=reason,
            message=kwargs.get("human_in_loop_message", "Human intervention required"),
            context=kwargs.get("human_in_loop_context", {}),
            tab_id=kwargs.get("tab_id") or session.active_tab_id,
        )

        session.metadata.setdefault("human_in_loop_requests", []).append(request)

        # In a real implementation, this would notify the user via UI
        # For now, we return the request for the caller to handle
        return ToolResult(
            success=True,
            output=f"Human-in-loop requested: {request.id}",
            metadata={
                "request_id": request.id,
                "reason": reason.value,
                "message": request.message,
                "context": request.context,
            },
        )

    async def _resolve_human_help(self, context: ToolContext, kwargs: dict) -> ToolResult:
        session_id = kwargs.get("session_id")
        request_id = kwargs.get("request_id")
        resolution = kwargs.get("resolution", "")

        if not session_id or session_id not in self._sessions:
            return ToolResult(success=False, error="Invalid or missing session_id")

        session = self._sessions[session_id]

        requests = session.metadata.get("human_in_loop_requests", [])
        request = next((r for r in requests if r.id == request_id), None)

        if not request:
            return ToolResult(success=False, error=f"Human-in-loop request not found: {request_id}")

        request.resolved_at = datetime.utcnow()
        request.resolution = resolution

        return ToolResult(
            success=True,
            output=f"Human-in-loop request {request_id} resolved",
            metadata={"resolution": resolution},
        )
