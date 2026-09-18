"""Ghidra Software Reverse Engineering Framework tool - disassembly, decompilation, graphing, scripting."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.ghidra.models import (
    GhidraAnalysisResult,
    GhidraConfig,
    GhidraFunction,
    GhidraProject,
    GhidraScriptLang,
    GhidraScriptResult,
    GhidraStatus,
)
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult

HEADLESS_DECOMPILE_POSTSCRIPT = r"""
import ghidra.app.decompiler.DecompInterface;
import ghidra.program.model.listing.Function;
import ghidra.app.script.GhidraScript;
import java.io.PrintWriter;

public class DecompileFunctions extends GhidraScript {
    @Override
    public void run() throws Exception {
        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);
        PrintWriter out = getOutWriter();
        FunctionIterator it = currentProgram.getFunctionManager().getFunctions(true);
        FileWriter fw = new FileWriter(getScriptArgs()[0]);
        while (it.hasNext() && monitor.isCancelled() == false) {
            Function f = it.next();
            out.println("FUNC|" + f.getName() + "|0x" + f.getEntryPoint());
            DecompileResults res = decomp.decompileFunction(f, 30, monitor);
            if (res != null && res.decompileCompleted()) {
                out.println("---DECOMPILE---");
                out.println(res.getDecompiledFunction().getC());
            }
        }
        fw.close();
        decomp.dispose();
    }
}
"""

HEADLESS_LIST_FUNCTIONS_POSTSCRIPT = r"""
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;

public class ListFunctions extends GhidraScript {
    @Override
    public void run() throws Exception {
        PrintWriter out = getOutWriter();
        FunctionIterator it = currentProgram.getFunctionManager().getFunctions(true);
        while (it.hasNext() && monitor.isCancelled() == false) {
            Function f = it.next();
            out.println("FUNC|" + f.getName() + "|0x" + f.getEntryPoint() + "|" + f.getBody().getNumAddresses());
        }
    }
}
"""


class GhidraTool(Tool):
    """Ghidra - NSA's software reverse engineering framework: disassembly, assembly, decompilation, graphing, scripting on Windows/macOS/Linux, headless or interactive."""

    @property
    def name(self) -> str:
        return "ghidra"

    @property
    def description(self) -> str:
        return "Ghidra SRE framework: analyze binaries headless (analyzeHeadless), list functions, decompile to C, extract strings/imports/exports, run Ghidra scripts (Java/Python/PyGhidra). Supports user-interactive and automated modes."

    @property
    def capabilities(self) -> list[str]:
        return ["reverse_engineering", "decompilation", "binary_analysis", "malware_analysis", "scripting"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action: analyze, list_functions, decompile, run_script, get_status",
                    required=True,
                    enum=["analyze", "list_functions", "decompile", "run_script", "get_status"],
                ),
                ToolParameter(
                    name="ghidra_home",
                    type_="string",
                    description="Path to Ghidra installation root (default: GHIDRA_INSTALL_DIR or auto-detect)",
                    required=False,
                ),
                ToolParameter(
                    name="binary",
                    type_="string",
                    description="Path to the binary/executable to analyze",
                    required=False,
                ),
                ToolParameter(
                    name="project_dir",
                    type_="string",
                    description="Directory to store the Ghidra project (default: temp)",
                    required=False,
                ),
                ToolParameter(
                    name="project_name",
                    type_="string",
                    description="Ghidra project name",
                    required=False,
                ),
                ToolParameter(
                    name="function_name",
                    type_="string",
                    description="Function name filter for decompile/list_functions",
                    required=False,
                ),
                ToolParameter(
                    name="script_path",
                    type_="string",
                    description="Path to a Ghidra script (.java, .py) to run",
                    required=False,
                ),
                ToolParameter(
                    name="script_args",
                    type_="array",
                    description="Arguments passed to the script",
                    required=False,
                    default=[],
                ),
                ToolParameter(
                    name="script_language",
                    type_="string",
                    description="Script language: java, python (PyGhidra)",
                    required=False,
                    default="java",
                ),
                ToolParameter(
                    name="analysis_mode",
                    type_="string",
                    description="Analysis mode: headless, scripted, batch",
                    required=False,
                    default="headless",
                ),
                ToolParameter(
                    name="analysis_timeout_sec",
                    type_="integer",
                    description="Per-file analysis timeout in seconds",
                    required=False,
                    default=600,
                ),
                ToolParameter(
                    name="max_memory_mb",
                    type_="string",
                    description="Max JVM memory, e.g. -Xmx4G",
                    required=False,
                ),
                ToolParameter(
                    name="extract_strings",
                    type_="boolean",
                    description="Extract defined strings from the binary",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="language",
                    type_="string",
                    description="Ghidra language ID, e.g. x86:LE:64:default",
                    required=False,
                ),
            ],
        )

    def __init__(self) -> None:
        self._config = GhidraConfig()
        self._projects: dict[str, GhidraProject] = {}
        self._status = GhidraStatus()

    def _find_ghidra_home(self, kwargs: dict) -> str:
        home = kwargs.get("ghidra_home", "") or os.environ.get("GHIDRA_INSTALL_DIR", "") or self._config.ghidra_home
        if home and Path(home).is_dir():
            return str(Path(home))
        env_default = os.environ.get("GHIDRA_INSTALL_DIR", "")
        if env_default:
            return str(Path(env_default))
        mounts = ["C:\\", "C:\\", "D:\\", "C:\\Users\\Ashut\\Downloads"]
        for mount in mounts:
            p = Path(mount)
            if p.is_dir():
                for cand in p.iterdir():
                    try:
                        if cand.is_dir() and cand.name.lower().startswith("ghidra"):
                            return str(cand)
                    except (PermissionError, OSError):
                        continue
        return ""

    def _get_analyze_headless_path(self, home: str) -> str:
        if not home:
            return ""
        bat = os.path.join(home, "support", "analyzeHeadless.bat")
        if os.path.isfile(bat):
            return bat
        sh = os.path.join(home, "support", "analyzeHeadless")
        if os.path.isfile(sh):
            return sh
        return ""

    def _check_java(self) -> str:
        try:
            r = subprocess.run(["java", "-version"], capture_output=True, timeout=10)
            return r.stderr.decode("utf-8", errors="replace").split("\n")[0]
        except Exception:
            return ""

    def _ensure_project(self, project_dir: str, project_name: str) -> GhidraProject:
        key = f"{project_dir}/{project_name}"
        if key in self._projects:
            return self._projects[key]
        Path(project_dir).mkdir(parents=True, exist_ok=True)
        project = GhidraProject(name=project_name, dir=project_dir)
        self._projects[key] = project
        return project

    def _build_headless_cmd(self, home: str, configurations: dict) -> list[str]:
        headless = self._get_analyze_headless_path(home)
        project_dir = configurations.get("project_dir") or tempfile.mkdtemp(prefix="ghidra_")
        project_name = configurations.get("project_name") or "astra_analysis"
        self._ensure_project(project_dir, project_name)

        cmd = [headless, project_dir, project_name]
        if configurations.get("import_binary"):
            cmd.extend(["-import", configurations["import_binary"]])
        if configurations.get("process_binary"):
            cmd.extend(["-process", configurations["process_binary"]])
        if configurations.get("language"):
            cmd.extend(["-language", configurations["language"]])
        timeout = configurations.get("analysis_timeout_sec", 600)
        cmd.extend(["-analysisTimeoutPerFile", str(timeout)])
        if configurations.get("max_memory_mb"):
            cmd.extend(["-java", configurations["max_memory_mb"]])
        if configurations.get("postscript"):
            cmd.extend(["-postScript", configurations["postscript"]])
            if configurations.get("postscript_args"):
                cmd.extend(["-postScriptArg"] + configurations["postscript_args"])
        if configurations.get("script_path"):
            cmd.extend(["-scriptPath", configurations["script_path"]])
        return cmd

    async def _run_cmd(self, cmd: list[str], timeout_sec: float) -> tuple[int, str, str]:
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=timeout_sec,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
        except TimeoutError:
            return -1, "", "timed out"

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        if action == "analyze":
            return await self._analyze(kwargs)
        elif action == "list_functions":
            return await self._list_functions(kwargs)
        elif action == "decompile":
            return await self._decompile(kwargs)
        elif action == "run_script":
            return await self._run_script(kwargs)
        elif action == "get_status":
            return await self._get_status()
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _analyze(self, kwargs: dict) -> ToolResult:
        home = self._find_ghidra_home(kwargs)
        if not home or not self._get_analyze_headless_path(home):
            return ToolResult(
                success=False,
                error="Ghidra not found. Install Ghidra (JDK 25 + ghidra_<ver>_PUBLIC_<date>.zip) and set GHIDRA_INSTALL_DIR.",
            )
        binary = kwargs.get("binary", "")
        if not binary or not Path(binary).is_file():
            return ToolResult(success=False, error="binary path is required and must exist")

        start = time.monotonic()
        configuration = {
            "project_dir": kwargs.get("project_dir", ""),
            "project_name": kwargs.get("project_name", "astra_analysis"),
            "import_binary": binary,
            "language": kwargs.get("language", ""),
            "analysis_timeout_sec": kwargs.get("analysis_timeout_sec", 600),
            "max_memory_mb": kwargs.get("max_memory_mb", ""),
            "extract_strings": kwargs.get("extract_strings", False),
        }
        cmd = self._build_headless_cmd(home, configuration)
        rc, out, err = await self._run_cmd(cmd, timeout_sec=1800.0)

        result = GhidraAnalysisResult(
            ok=rc == 0,
            output=out[-12000:] if out else err,
            error=None if rc == 0 else (err[-4000:] or f"exit {rc}"),
            project_path=configuration["project_dir"],
            duration_sec=round(time.monotonic() - start, 2),
            stdout=out,
            stderr=err,
        )
        if result.ok:
            return ToolResult(
                success=True,
                output=result.output,
                metadata={
                    "project_dir": result.project_path,
                    "duration_sec": result.duration_sec,
                    "binary": binary,
                },
            )
        return ToolResult(success=False, error=result.error or "Ghidra analysis failed")

    async def _list_functions(self, kwargs: dict) -> ToolResult:
        home = self._find_ghidra_home(kwargs)
        if not home or not self._get_analyze_headless_path(home):
            return ToolResult(success=False, error="Ghidra not found. Install Ghidra and set GHIDRA_INSTALL_DIR.")

        binary = kwargs.get("binary", "")
        if not binary or not Path(binary).is_file():
            return ToolResult(success=False, error="binary path is required and must exist")

        postscript_dir = tempfile.mkdtemp(prefix="ghidra_script_")
        ps_path = os.path.join(postscript_dir, "ListFunctions.java")
        Path(ps_path).write_text(HEADLESS_LIST_FUNCTIONS_POSTSCRIPT, encoding="utf-8")

        configuration = {
            "project_dir": tempfile.mkdtemp(prefix="ghidra_proj_"),
            "project_name": "lf",
            "import_binary": binary,
            "postscript": os.path.splitext(os.path.basename(ps_path))[0],
            "postscript_args": [],
            "script_path": postscript_dir,
            "analysis_timeout_sec": kwargs.get("analysis_timeout_sec", 600),
        }
        cmd = self._build_headless_cmd(home, configuration)
        rc, out, err = await self._run_cmd(cmd, timeout_sec=1800.0)

        function_lines = []
        for line in out.splitlines():
            if line.startswith("FUNC|"):
                function_lines.append(line)
        functions = []
        for line in function_lines:
            parts = line.split("|")
            if len(parts) >= 4:
                functions.append(GhidraFunction(name=parts[1], entry=parts[2], size=int(parts[3]) if parts[3].isdigit() else 0))

        if rc != 0:
            return ToolResult(success=False, error=err[-4000:] or f"exit {rc}", metadata={"functions_found": len(functions)})

        return ToolResult(
            success=True,
            output=json.dumps([f.__dict__ for f in functions], indent=2, default=str),
            metadata={"functions_found": len(functions)},
        )

    async def _decompile(self, kwargs: dict) -> ToolResult:
        home = self._find_ghidra_home(kwargs)
        if not home or not self._get_analyze_headless_path(home):
            return ToolResult(success=False, error="Ghidra not found. Install Ghidra and set GHIDRA_INSTALL_DIR.")

        binary = kwargs.get("binary", "")
        if not binary or not Path(binary).is_file():
            return ToolResult(success=False, error="binary path is required and must exist")

        postscript_dir = tempfile.mkdtemp(prefix="ghidra_script_")
        ps_path = os.path.join(postscript_dir, "DecompileFunctions.java")
        out_buf = os.path.join(postscript_dir, "decompiled.txt")
        Path(ps_path).write_text(HEADLESS_DECOMPILE_POSTSCRIPT, encoding="utf-8")

        configuration = {
            "project_dir": tempfile.mkdtemp(prefix="ghidra_proj_"),
            "project_name": "dec",
            "import_binary": binary,
            "postscript": os.path.splitext(os.path.basename(ps_path))[0],
            "postscript_args": [out_buf],
            "script_path": postscript_dir,
            "analysis_timeout_sec": kwargs.get("analysis_timeout_sec", 600),
        }
        cmd = self._build_headless_cmd(home, configuration)
        rc, out, err = await self._run_cmd(cmd, timeout_sec=1800.0)

        decompiled_path = out_buf if os.path.isfile(out_buf) else None
        result = GhidraAnalysisResult(
            ok=rc == 0,
            output=(Path(decompiled_path).read_text(encoding="utf-8", errors="replace") if decompiled_path else out)[-12000:],
            error=None if rc == 0 else (err[-4000:] or f"exit {rc}"),
            project_path=configuration["project_dir"],
        )
        if result.ok:
            return ToolResult(success=True, output=result.output or "No decompiled output.")
        return ToolResult(success=False, error=result.error)

    async def _run_script(self, kwargs: dict) -> ToolResult:
        home = self._find_ghidra_home(kwargs)
        if not home or not self._get_analyze_headless_path(home):
            return ToolResult(success=False, error="Ghidra not found. Install Ghidra and set GHIDRA_INSTALL_DIR.")

        binary = kwargs.get("binary", "")
        if not binary or not Path(binary).is_file():
            return ToolResult(success=False, error="binary path is required and must exist")

        script_path = kwargs.get("script_path", "")
        if not script_path or not Path(script_path).is_file():
            return ToolResult(success=False, error="script_path is required and must exist")

        script_dir = str(Path(script_path).parent)
        script_basename = Path(script_path).stem
        language = kwargs.get("script_language", "java")
        if language == GhidraScriptLang.PYTHON.value:
            script_basename = script_basename + ".py"

        configuration = {
            "project_dir": tempfile.mkdtemp(prefix="ghidra_proj_"),
            "project_name": "scr",
            "import_binary": binary,
            "postscript": script_basename,
            "postscript_args": kwargs.get("script_args", []),
            "script_path": script_dir,
            "analysis_timeout_sec": kwargs.get("analysis_timeout_sec", 600),
        }
        cmd = self._build_headless_cmd(home, configuration)
        rc, out, err = await self._run_cmd(cmd, timeout_sec=1800.0)

        result = GhidraScriptResult(
            ok=rc == 0,
            output=out[-12000:] if out else err,
            error=None if rc == 0 else (err[-4000:] or f"exit {rc}"),
            stdout=out,
            stderr=err,
        )
        if result.ok:
            return ToolResult(success=True, output=result.output)
        return ToolResult(success=False, error=result.error)

    async def _get_status(self) -> ToolResult:
        home = self._find_ghidra_home({})
        headless = self._get_analyze_headless_path(home)
        version = ""
        if headless:
            _, out, err = await self._run_cmd([headless], timeout_sec=60.0)
            version = (out + err)[:200]

        self._status = GhidraStatus(
            installed=bool(headless),
            ghidra_home=home,
            analyze_headless_path=headless,
            version=version,
            java_version=self._check_java(),
            projects=list(self._projects.values()),
        )
        return ToolResult(
            success=True,
            output=json.dumps({
                "installed": self._status.installed,
                "ghidra_home": self._status.ghidra_home,
                "analyze_headless_path": self._status.analyze_headless_path,
                "java_version": self._status.java_version,
                "version_hint": self._status.version,
                "projects": [p.__dict__ for p in self._status.projects],
            }, indent=2, default=str),
            metadata={"installed": self._status.installed},
        )
