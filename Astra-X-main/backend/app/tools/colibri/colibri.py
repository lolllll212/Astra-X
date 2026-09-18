"""Colibri inference engine tool - run frontier MoE models (744B-2.8T) on consumer hardware via AI memory multitiering."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any

from app.tools.base import Tool
from app.tools.colibri.models import (
    ColibriBackend,
    ColibriConfig,
    ColibriRunResult,
    ColibriServeSession,
    ColibriStatus,
)
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class ColibriTool(Tool):
    """Colibri - tiny engine, immense model. Run frontier MoE models (744B-2.8T) on consumer and heterogeneous hardware by treating VRAM, RAM, and storage as a single multitier inference hierarchy."""

    @property
    def name(self) -> str:
        return "colibri"

    @property
    def description(self) -> str:
        return "Colibri inference engine: chat/serve/run GLM-5.2 (744B), GLM-5.3, Inkling (975B), Kimi K3 (2.8T), DeepSeek V4/V4.1 Flash, Qwen3.8/3.6, OLMoE. AI memory multitiering streams experts from VRAM/RAM/storage. Also: info, plan, doctor, mirror, bench, convert, build."

    @property
    def capabilities(self) -> list[str]:
        return ["frontier_inference", "moe_model_running", "multitier_memory", "local_serving"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action: run, chat, serve, stop_serve, web, info, plan, doctor, mirror, bench, build, get_status",
                    required=True,
                    enum=["run", "chat", "serve", "stop_serve", "web", "info", "plan", "doctor", "mirror", "bench", "build", "get_status"],
                ),
                ToolParameter(
                    name="prompt",
                    type_="string",
                    description="Prompt / text input for run or chat",
                    required=False,
                ),
                ToolParameter(
                    name="model_dir",
                    type_="string",
                    description="Path to Colibri model directory (or set COLI_MODEL)",
                    required=False,
                ),
                ToolParameter(
                    name="mirror_dir",
                    type_="string",
                    description="Second copy of the model on another drive for dual-SSD striping",
                    required=False,
                ),
                ToolParameter(
                    name="ram",
                    type_="number",
                    description="RAM budget in GB (auto-sizes expert cache)",
                    required=False,
                ),
                ToolParameter(
                    name="repin",
                    type_="integer",
                    description="Adapt RAM/VRAM experts every N tokens",
                    required=False,
                ),
                ToolParameter(
                    name="topp",
                    type_="number",
                    description="Adaptive expert top-p",
                    required=False,
                ),
                ToolParameter(
                    name="topk",
                    type_="integer",
                    description="Fixed top-k",
                    required=False,
                ),
                ToolParameter(
                    name="ngen",
                    type_="integer",
                    description="Maximum response tokens",
                    required=False,
                ),
                ToolParameter(
                    name="cap",
                    type_="integer",
                    description="Cache slots per layer",
                    required=False,
                ),
                ToolParameter(
                    name="backend",
                    type_="string",
                    description="Backend: auto, cpu, cuda, metal, vulkan",
                    required=False,
                    default="auto",
                ),
                ToolParameter(
                    name="host",
                    type_="string",
                    description="Serve host (default 127.0.0.1)",
                    required=False,
                    default="127.0.0.1",
                ),
                ToolParameter(
                    name="port",
                    type_="integer",
                    description="Serve port (default 8000)",
                    required=False,
                    default=8000,
                ),
                ToolParameter(
                    name="bench_tasks",
                    type_="array",
                    description="Benchmark task names for bench action",
                    required=False,
                    default=[],
                ),
                ToolParameter(
                    name="extra_args",
                    type_="array",
                    description="Extra CLI args passed verbatim",
                    required=False,
                    default=[],
                ),
            ],
        )

    def __init__(self) -> None:
        self._cli_path = self._find_coli()
        self._serve_sessions: dict[int, ColibriServeSession] = {}
        self._status = ColibriStatus(cli_found=bool(self._cli_path), cli_path=self._cli_path or "")

    def _find_coli(self) -> str:
        """Locate the coli launcher: env var, common download paths, or PATH."""
        candidates: list[str] = []
        env = os.environ.get("COLI_PATH", "")
        if env:
            candidates.append(env)
        for root in (r"C:\Users\Ashut\Downloads\colibri-main\c", r"C:\Users\Ashut\Downloads\colibri-main"):
            candidates.append(os.path.join(root, "coli.cmd"))
            candidates.append(os.path.join(root, "coli"))
        candidates.extend(["coli", "coli.cmd"])
        for candidate in candidates:
            if candidate.lower().endswith(".cmd") or candidate.lower().endswith(".bat"):
                if os.path.isfile(candidate):
                    return candidate
            elif os.path.isfile(candidate) or shutil.which(candidate):
                return candidate
        return "coli"

    def _build_config(self, kwargs: dict) -> ColibriConfig:
        return ColibriConfig(
            model_dir=kwargs.get("model_dir", os.environ.get("COLI_MODEL", "")),
            mirror_dir=kwargs.get("mirror_dir", os.environ.get("COLI_MODEL_MIRROR", "")),
            ram_budget_gb=kwargs.get("ram", 0),
            repin_interval_tokens=kwargs.get("repin", 0),
            topp=kwargs.get("topp", 0),
            topk=kwargs.get("topk", 0),
            ngen=kwargs.get("ngen", 0),
            cap=kwargs.get("cap", 0),
            backend=ColibriBackend(kwargs.get("backend", "auto")) if kwargs.get("backend") else ColibriBackend.AUTO,
            extra_args=kwargs.get("extra_args", []),
        )

    def _apply_config(self, cmd: list[str], config: ColibriConfig) -> list[str]:
        if config.model_dir:
            cmd.extend(["--model", config.model_dir])
        if config.mirror_dir:
            cmd.extend(["--model-mirror", config.mirror_dir])
        if config.ram_budget_gb > 0:
            cmd.extend(["--ram", str(int(config.ram_budget_gb))])
        if config.repin_interval_tokens > 0:
            cmd.extend(["--repin", str(config.repin_interval_tokens)])
        if config.topp > 0:
            cmd.extend(["--topp", str(config.topp)])
        if config.topk > 0:
            cmd.extend(["--topk", str(config.topk)])
        if config.ngen > 0:
            cmd.extend(["--ngen", str(config.ngen)])
        if config.cap > 0:
            cmd.extend(["--cap", str(config.cap)])
        if config.backend != ColibriBackend.AUTO:
            cmd.extend(["--backend", config.backend.value])
        for arg in config.extra_args:
            cmd.append(arg)
        return cmd

    async def _run_cli(self, args: list[str], timeout_sec: float = 300.0) -> tuple[int, str, str]:
        """Run the coli launcher and capture output."""
        cmd = [self._cli_path, *args]
        env = dict(os.environ)
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            ),
            timeout=timeout_sec,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
        except TimeoutError:
            proc.kill()
            return -1, "", "timed out"

    async def _process_alive(self, proc: asyncio.subprocess.Process) -> bool:
        try:
            await asyncio.wait_for(proc.wait(), timeout=0.1)
            return False
        except TimeoutError:
            return True

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        if action == "run":
            return await self._run(kwargs)
        elif action == "chat":
            return await self._run(kwargs, chat=True)
        elif action == "serve":
            return await self._serve(kwargs)
        elif action == "stop_serve":
            return await self._stop_serve(kwargs)
        elif action == "web":
            return await self._web(kwargs)
        elif action == "info":
            return await self._info(kwargs)
        elif action == "plan":
            return await self._plan(kwargs)
        elif action == "doctor":
            return await self._doctor(kwargs)
        elif action == "mirror":
            return await self._mirror(kwargs)
        elif action == "bench":
            return await self._bench(kwargs)
        elif action == "build":
            return await self._build(kwargs)
        elif action == "get_status":
            return await self._get_status(kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _run(self, kwargs: dict, chat: bool = False) -> ToolResult:
        prompt = kwargs.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, error="prompt is required for run/chat")

        config = self._build_config(kwargs)
        args = ["run", prompt] if not chat else ["chat", "--prompt", prompt]
        args = self._apply_config(args, config)

        rc, out, err = await self._run_cli(args, timeout_sec=1200.0)

        result = ColibriRunResult(
            ok=rc == 0,
            output=out.strip() if rc == 0 else err.strip(),
            error=None if rc == 0 else (err.strip() or f"exit {rc}"),
            engine_stdout=out,
            engine_stderr=err,
        )
        if result.ok:
            return ToolResult(
                success=True,
                output=result.output[:12000],
                metadata={"tokens": result.tokens, "tps": result.tps, "ttfs_ms": result.ttfs_ms},
            )
        return ToolResult(success=False, error=result.error)

    async def _serve(self, kwargs: dict) -> ToolResult:
        config = self._build_config(kwargs)
        host = kwargs.get("host", "127.0.0.1")
        port = int(kwargs.get("port", 8000))
        if not config.model_dir:
            return ToolResult(success=False, error="model_dir is required for serve (or set COLI_MODEL)")

        existing = self._serve_sessions.get(port)
        if existing and existing.process and await self._process_alive(existing.process):
            return ToolResult(success=False, error=f"Colibri serve already running on port {port}")

        args = ["serve", "--host", host, "--port", str(port)]
        args = self._apply_config(args, config)

        env = dict(os.environ)
        env["COLI_SERVE_PORT"] = str(port)
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to start colibri serve: {exc}")

        await asyncio.sleep(6)
        if not await self._process_alive(proc):
            _, err = await proc.communicate()
            return ToolResult(success=False, error=err.decode("utf-8", errors="replace").strip() or "colibri serve exited immediately")

        session = ColibriServeSession(host=host, port=port, pid=proc.pid, model_dir=config.model_dir, process=proc)
        self._serve_sessions[port] = session
        return ToolResult(
            success=True,
            output=f"Colibri OpenAI-compatible server started on http://{host}:{port}",
            metadata={"port": port, "pid": proc.pid, "model": config.model_dir},
        )

    async def _stop_serve(self, kwargs: dict) -> ToolResult:
        port = int(kwargs.get("port", 8000))
        session = self._serve_sessions.get(port)
        if not session or not session.process:
            return ToolResult(success=False, error=f"No colibri serve session on port {port}")

        proc = session.process
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=10)
        except Exception:
            proc.kill()
        del self._serve_sessions[port]
        return ToolResult(success=True, output=f"Colibri serve stopped on port {port}")

    async def _web(self, kwargs: dict) -> ToolResult:
        config = self._build_config(kwargs)
        host = kwargs.get("host", "127.0.0.1")
        port = int(kwargs.get("port", 8000))
        if not config.model_dir:
            return ToolResult(success=False, error="model_dir is required for web (or set COLI_MODEL)")

        args = ["web", "--host", host, "--port", str(port)]
        args = self._apply_config(args, config)
        env = dict(os.environ)
        env["SERVE"] = "1"
        env["COLI_SERVE_PORT"] = str(port)
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to start colibri web: {exc}")

        await asyncio.sleep(6)
        if not await self._process_alive(proc):
            _, err = await proc.communicate()
            return ToolResult(success=False, error=err.decode("utf-8", errors="replace").strip() or "colibri web exited immediately")

        session = ColibriServeSession(host=host, port=port, pid=proc.pid, model_dir=config.model_dir, process=proc)
        self._serve_sessions[port] = session
        return ToolResult(
            success=True,
            output=f"Colibri web dashboard served on http://{host}:{port}",
            metadata={"port": port, "pid": proc.pid, "model": config.model_dir},
        )

    async def _info(self, kwargs: dict) -> ToolResult:
        rc, out, err = await self._run_cli(["info"], timeout_sec=120.0)
        if rc == 0:
            return ToolResult(success=True, output=out.strip()[:12000])
        return ToolResult(success=False, error=err.strip() or f"exit {rc}")

    async def _plan(self, kwargs: dict) -> ToolResult:
        config = self._build_config(kwargs)
        if not config.model_dir:
            return ToolResult(success=False, error="model_dir is required for plan (or set COLI_MODEL)")
        args = self._apply_config(["plan"], config)
        rc, out, err = await self._run_cli(args, timeout_sec=600.0)
        if rc == 0:
            return ToolResult(success=True, output=out.strip()[:12000])
        return ToolResult(success=False, error=err.strip() or f"exit {rc}")

    async def _doctor(self, kwargs: dict) -> ToolResult:
        config = self._build_config(kwargs)
        args = self._apply_config(["doctor"], config)
        rc, out, err = await self._run_cli(args, timeout_sec=300.0)
        if rc == 0:
            return ToolResult(success=True, output=out.strip()[:12000])
        return ToolResult(success=False, error=err.strip() or f"exit {rc}")

    async def _mirror(self, kwargs: dict) -> ToolResult:
        config = self._build_config(kwargs)
        if not config.model_dir or not config.mirror_dir:
            return ToolResult(success=False, error="model_dir and mirror_dir are required for mirror")
        args = self._apply_config(["mirror"], config)
        rc, out, err = await self._run_cli(args, timeout_sec=600.0)
        if rc == 0:
            return ToolResult(success=True, output=out.strip()[:12000])
        return ToolResult(success=False, error=err.strip() or f"exit {rc}")

    async def _bench(self, kwargs: dict) -> ToolResult:
        config = self._build_config(kwargs)
        tasks = kwargs.get("bench_tasks", [])
        args = ["bench", *list(tasks)]
        args = self._apply_config(args, config)
        rc, out, err = await self._run_cli(args, timeout_sec=3600.0)
        if rc == 0:
            return ToolResult(success=True, output=out.strip()[:12000])
        return ToolResult(success=False, error=err.strip() or f"exit {rc}")

    async def _build(self, kwargs: dict) -> ToolResult:
        rc, out, err = await self._run_cli(["build"], timeout_sec=3600.0)
        if rc == 0:
            return ToolResult(success=True, output=out.strip()[:12000])
        return ToolResult(success=False, error=err.strip() or f"exit {rc}")

    async def _get_status(self, kwargs: dict) -> ToolResult:
        cli_known = self._cli_path and (os.path.sep in self._cli_path or os.path.isfile(self._cli_path))
        self._status.cli_found = bool(cli_known)
        self._status.cli_path = self._cli_path or ""
        living = []
        for s in self._serve_sessions.values():
            if s.process and await self._process_alive(s.process):
                living.append(s)
        self._status.active_serve = living[0] if living else None
        return ToolResult(
            success=True,
            output=json.dumps({
                "cli_found": self._status.cli_found,
                "cli_path": self._status.cli_path,
                "built": self._status.built,
                "supported_families": ["GLM-5.2/5.3", "GLM-5.3-Flash", "Inkling", "Kimi K3", "DeepSeek V4 Flash", "DeepSeek V4.1 Flash", "Qwen3.8-Flash-Next", "Qwen3.6", "OLMoE"],
                "detected_backends": self._status.detected_backends,
                "active_serve": [{"host": s.host, "port": s.port, "pid": s.pid} for s in living],
                "model_hint": os.environ.get("COLI_MODEL", ""),
            }, indent=2),
        )
