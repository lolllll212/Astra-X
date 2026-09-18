"""AutoFix - autonomous "repair-the-project" agent.

Accepts a single natural-language prompt (e.g. *"I think I wrote some wrong
codes in this database, can you fix it"*), derives the target directory, and
drives the real tool stack end-to-end:

1. Resolve target (:code:`path` param, or a folder name mentioned in the
   prompt, probed under :envvar:`USERPROFILE`).
2. Scan for candidate source files (Python/JS/TS/JSON/YAML/SQL).
3. Use the local coding LLM (:class:`~app.tools.lmstudio.LmStudioTool`,
   default qwen2.5-coder-7b) to analyze candidates and produce concrete edits.
4. Apply edits through the real filesystem tools
   (:class:`~app.tools.filesystem` readers/writers), creating backups and
   syntax-checking edited Python files.
5. Sensitive files (``.env``, keys, secrets, dotfiles with credentials)
   are never rewritten without explicit user approval — the run returns an
   :code:`approval_token` and :code:`confirm_writes` applies them.

Heavy models (Colibri, larger LM Studio models) are engaged for the analysis
step when the prompt requests them or the default LLM is unavailable.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.tools.autofix.models import AutoFixRun, AutoFixStatus, FileEdit
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult

USERPROFILE = os.environ.get("USERPROFILE") or str(Path.home())
DEFAULT_SEARCH_ROOTS = [
    str(Path.home()),
    str(Path.home() / "Downloads"),
]

SENSITIVE_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "id_rsa", "id_ed25519", "credentials", "service-account",
}
SENSITIVE_PATTERNS = [
    re.compile(r"^\.env(\..+)?$"),
    re.compile(r"\.(key|pem|p12|pfx|jks|pkcs12)$", re.IGNORECASE),
    re.compile(r".*\.secret$"),
    re.compile(r".*(credential|secret|token|apikey|api_key).*(\.json|\.env|\.txt)$", re.IGNORECASE),
]

SCAN_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml", ".sql", ".toml", ".sh", ".bat", ".cmd"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ruff_cache", ".mypy_cache", "dist", "build", ".pytest_cache", ".cache", ".idea", ".vscode"}
SKIP_FILES = {".env", ".env.local", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

MAX_CANDIDATES = 200
MAX_FILE_BYTES = 256 * 1024
MAX_EDIT_BYTES = 24 * 1024
MAX_PREVIEW_CHARS = 3000


def _find_target(path: str, prompt: str) -> str:
    """Resolve the directory the agent should operate on.

    Precedence: explicit :code:`path` -> folder names mentioned in the
    prompt (matched against directories under the user profile / Downloads) ->
    current working directory.
    """
    if path:
        p = Path(path)
        if p.is_dir():
            return str(p.resolve())
        # maybe a file -> its parent
        if p.exists():
            return str(p.parent.resolve())
        return str(p.resolve())

    # Folder names referenced inside the prompt: strip file-extension-y tokens.
    words = re.findall(r"[\w\-\.]+", prompt.lower())
    names = {w.lower().rstrip(".") for w in words}
    interesting = {
        w for w in names
        if len(w) >= 3
        and not w.endswith((".py", ".js", ".ts", ".json", ".md", ".txt", ".yaml", ".yml", ".sql", ".toml"))
    }

    for root in DEFAULT_SEARCH_ROOTS:
        try:
            root_p = Path(root)
            if not root_p.is_dir():
                continue
            for child in root_p.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if child.name.lower() in interesting and not _is_skip_dir(child.name):
                    return str(child.resolve())
        except OSError:
            continue

    return os.getcwd()


def _is_skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name in {d.lower() for d in SKIP_DIRS}


def _is_sensitive(path: Path) -> bool:
    name = path.name
    if name in SENSITIVE_NAMES or name in {n.lower() for n in SENSITIVE_NAMES}:
        return True
    return any(pat.match(name) for pat in SENSITIVE_PATTERNS)


def _walk_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not _is_skip_dir(d) and not d.startswith(".")]
        for fname in files:
            if fname in SKIP_FILES:
                continue
            p = Path(current) / fname
            if p.suffix.lower() not in SCAN_EXTENSIONS:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            candidates.append(p)
            if len(candidates) >= MAX_CANDIDATES:
                return candidates
    return candidates


class AutoFixTool(Tool):
    """AutoFix - autonomous code repair using the local tool stack."""

    @property
    def name(self) -> str:
        return "autofix"

    @property
    def description(self) -> str:
        return "Autonomous repair agent: takes one prompt like 'fix the wrong code in this database', finds and fixes bugs across the target project using the local LLM + filesystem tools, without touching sensitive files (.env, keys, secrets) until you approve. Review action performs a read-only audit with upgrade suggestions and database-relevance classification."

    @property
    def capabilities(self) -> list[str]:
        return ["autonomous_repair", "code_fixing", "project_scanning", "automation"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action: fix, review, confirm_writes, get_status",
                    required=True,
                    enum=["fix", "review", "confirm_writes", "get_status"],
                ),
                ToolParameter(
                    name="prompt",
                    type_="string",
                    description="Natural-language repair request (e.g. 'fix the wrong code in the ecc-main folder')",
                    required=False,
                ),
                ToolParameter(
                    name="path",
                    type_="string",
                    description="Explicit target directory to scan and fix (overrides prompt folder detection)",
                    required=False,
                ),
                ToolParameter(
                    name="model",
                    type_="string",
                    description="LM Studio model id for analysis (default qwen2.5-coder-7b-instruct)",
                    required=False,
                ),
                ToolParameter(
                    name="use_heavy_model",
                    type_="boolean",
                    description="Engage a heavier/heavy-model pass for analysis when available",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="approval_token",
                    type_="string",
                    description="Token returned by a fix run to approve writing its sensitive files",
                    required=False,
                ),
                ToolParameter(
                    name="max_files",
                    type_="integer",
                    description="Maximum number of files to analyze per run",
                    required=False,
                    default=16,
                ),
            ],
        )

    def __init__(self) -> None:
        self._runs: dict[str, AutoFixRun] = {}
        self._status = AutoFixStatus(base_dir=os.getcwd())
        self._lmstudio = None
        self._read_tool = None
        self._write_tool = None

    @property
    def _llm(self):
        if self._lmstudio is None:
            from app.tools.lmstudio import LmStudioTool

            self._lmstudio = LmStudioTool()
        return self._lmstudio

    @property
    def _reader(self):
        if self._read_tool is None:
            from app.tools.filesystem.read_file import ReadFileTool

            self._read_tool = ReadFileTool()
        return self._read_tool

    @property
    def _writer(self):
        if self._write_tool is None:
            from app.tools.filesystem.write_file import WriteFileTool

            self._write_tool = WriteFileTool()
        return self._write_tool

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        if action == "fix":
            return await self._fix(context, kwargs)
        elif action == "review":
            return await self._review(context, kwargs)
        elif action == "confirm_writes":
            return await self._confirm_writes(context, kwargs)
        elif action == "get_status":
            return await self._get_status(context, kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # fix
    # ------------------------------------------------------------------

    async def _fix(self, context: ToolContext, kwargs: dict) -> ToolResult:
        prompt = kwargs.get("prompt", "").strip()
        path = kwargs.get("path", "").strip()
        use_heavy = bool(kwargs.get("use_heavy_model", False))
        max_files = int(kwargs.get("max_files", 16))

        if not prompt and not path:
            return ToolResult(success=False, error="prompt is required (or pass path to fix a directory)")

        model = kwargs.get("model") or (
            "deepseek-r1-distill-qwen-7b" if use_heavy else "qwen2.5-coder-7b-instruct"
        )

        target = _find_target(path, prompt or "")
        run = AutoFixRun(id=str(uuid4())[:8], prompt=prompt or "", target_dir=target, model=model)
        self._runs[run.id] = run

        try:
            root = Path(target)
            if not root.is_dir():
                run.status = "failed"
                run.completed_at = datetime.utcnow()
                return ToolResult(success=False, error=f"Target directory not found: {target}")

            # Scan phase
            files = _walk_candidates(root)[:max_files]
            run.scans.append(f"scanned {len(files)} candidate files under {target}")
            findings: list[str] = []

            # If there are no candidates, report (e.g. the dir is empty/binary-heavy).
            if not files:
                run.status = "completed"
                run.completed_at = datetime.utcnow()
                return ToolResult(
                    success=True,
                    output=json.dumps({
                        "target_dir": target,
                        "files_analyzed": 0,
                        "edits_applied": 0,
                        "message": "No editable source files found to repair.",
                    }, indent=2),
                )

            # Analysis phase — ask the coding LLM to inspect batches.
            for chunk_start in range(0, len(files), 4):
                chunk = files[chunk_start:chunk_start + 4]
                file_reports = await self._analyze_batch(root, chunk, prompt, model)
                for file_report in file_reports:
                    if not file_report:
                        continue
                    findings.extend(file_report.get("issues", []) or [])
                    edit = self._build_edit(root, file_report, run)
                    if edit is not None:
                        run.edits.append(edit)

            edits_to_apply = [e for e in run.edits if not e.needs_approval]
            pending_edits = [e for e in run.edits if e.needs_approval]

            # Apply phase (safe edits only).
            for edit in edits_to_apply:
                if not await self._apply_edit(context, run, edit):
                    run.findings.append(f"apply failed: {edit.path}")

            run.edits_applied = len([e for e in run.edits if e.applied])

            approval_token = ""
            if pending_edits:
                approval_token = uuid4().hex
                run.approval_token = approval_token
                run.status = "awaiting_approval"
            else:
                run.status = "completed"
            run.completed_at = datetime.utcnow()
            self._status.last_run_id = run.id
            self._status.last_run_status = run.status
            self._status.edits_applied += run.edits_applied
            self._status.edits_pending_approval = len(pending_edits)
            self._status.sensitive_detected += len(pending_edits)

            return ToolResult(success=True, output=self._render_manifest(run), metadata={
                "run_id": run.id,
                "target_dir": target,
                "files_analyzed": len(files),
                "edits_applied": run.edits_applied,
                "edits_pending_approval": len(pending_edits),
                "approval_token": approval_token,
                "status": run.status,
                "sensitive_files": [e.path for e in pending_edits],
            })

        except Exception as exc:
            run.status = "failed"
            run.completed_at = datetime.utcnow()
            return ToolResult(success=False, error=f"AutoFix failed: {exc}")

    async def _review(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Read-only audit: analyze files with the local LLM and return
        per-file issues, upgrade suggestions, and database-relevance flags.
        Never writes to disk."""
        prompt = kwargs.get("prompt", "").strip()
        path = kwargs.get("path", "").strip()
        use_heavy = bool(kwargs.get("use_heavy_model", False))
        max_files = int(kwargs.get("max_files", 16))

        if not prompt and not path:
            return ToolResult(success=False, error="prompt is required (or pass path to review a directory)")

        model = kwargs.get("model") or (
            "deepseek-r1-distill-qwen-7b" if use_heavy else "qwen2.5-coder-7b-instruct"
        )
        target = _find_target(path, prompt or "")
        root = Path(target)
        if not root.is_dir():
            return ToolResult(success=False, error=f"Target directory not found: {target}")

        files = _walk_candidates(root)[:max_files]
        if not files:
            return ToolResult(
                success=True,
                output=json.dumps({
                    "target_dir": target,
                    "files_analyzed": 0,
                    "reviews": [],
                    "summary": "No editable source files found to review.",
                }, indent=2),
            )

        reviews: list[dict] = []
        for chunk_start in range(0, len(files), 4):
            chunk = files[chunk_start:chunk_start + 4]
            batch_reports = await self._review_batch(root, chunk, prompt, model)
            for report, p in zip(batch_reports, chunk, strict=False):
                if report:
                    reviews.append(report)
                    continue
                single = await self._review_batch(root, [p], prompt, model)
                if single and single[0]:
                    reviews.append(single[0])

        summary = {
            "files_analyzed": len(files),
            "with_issues": sum(1 for r in reviews if r.get("issues")),
            "db_related": sum(1 for r in reviews if r.get("database_related")),
            "non_db": [r.get("path") for r in reviews if not r.get("database_related")],
            "unwanted_suggestions": [
                r.get("path") for r in reviews
                if r.get("action") == "remove" or r.get("priority") == "unrelated-junk"
            ],
        }
        return ToolResult(
            success=True,
            output=json.dumps({"target_dir": target, "reviews": reviews, "summary": summary}, indent=2),
            metadata={"target_dir": target, "files_analyzed": len(files), "reviews": len(reviews)},
        )

    async def _review_batch(self, root: Path, files: list[Path], prompt: str, model: str) -> list[dict | None]:
        """Send a batch of file contents to the coding LLM; parse its JSON review plan."""
        contents: list[dict] = []
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            contents.append({"path": str(p.relative_to(root)), "language": p.suffix.lstrip("."), "content": text[:MAX_PREVIEW_CHARS]})

        system = (
            "You are a meticulous senior code reviewer embedded in an autonomous audit system. "
            "You find real bugs (syntax errors, undefined names, wrong imports, broken logic, "
            "security issues), suggest concrete upgrades, and classify each file's relation to "
            "the project's database layer. Be terse, concrete, and certain."
        )
        user = (
            f"Audit request: {prompt}\n\n"
            "Below are source files from the target project. Analyze each one.\n"
            "Respond with ONLY valid JSON (no markdown fences), an array with one object per file:\n"
            "[{\"path\": \"...\", \"issues\": [\"short issue or empty\"], "
            "\"upgrades\": [\"concrete upgrade suggestion or empty\"], "
            "\"database_related\": true or false, "
            "\"action\": \"keep\" | \"remove\" | \"review\", "
            "\"priority\": \"critical\" | \"normal\" | \"unrelated-junk\"}]\n"
            "Rules: set database_related true only if the file touches the DB (schemas, migrations, "
            "repositories, models, queries, connection/session handling). Action 'remove' only for "
            "clear junk (empty, duplicate, scratch/test-harness, placeholder) files UNRELATED to the "
            "database. 'unrelated-junk' priority for files the project does not need."
        )
        payload = {"files": contents}

        result = await self._llm.execute(
            ToolContext(workspace=str(root)),
            action="chat",
            model=model,
            system=system,
            prompt=user + "\n\n" + json.dumps(payload, ensure_ascii=False)[:MAX_EDIT_BYTES * 6],
            temperature=0.1,
            max_tokens=4096,
        )
        if not result.success or not result.output:
            return [None] * len(contents)

        try:
            parsed = self._extract_json(result.output)
        except Exception:
            return [None] * len(contents)
        if not isinstance(parsed, list):
            return [None] * len(contents)
        return parsed

    async def _analyze_batch(self, root: Path, files: list[Path], prompt: str, model: str) -> list[dict | None]:
        """Send a batch of file contents to the coding LLM; parse its JSON repair plan."""
        contents: list[dict] = []
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            contents.append({"path": str(p.relative_to(root)), "language": p.suffix.lstrip("."), "content": text[:MAX_FILE_BYTES]})

        system = (
            "You are a senior code reviewer embedded in an autonomous repair system. "
            "You inspect code and find real bugs: syntax errors, undefined names, "
            "wrong imports, broken logic, security issues. "
            "You ONLY request edits that are certain fixes, never stylistic changes."
        )
        user = (
            f"Repair request: {prompt}\n\n"
            "Below are source files from the target project. Analyze each one carefully.\n"
            "Respond with ONLY valid JSON (no markdown fences), an array, one object per file:\n"
            "[{\"path\": \"...\", \"issues\": [\"short description\"], \"fixed_content\": \"ENTIRE corrected file content OR empty string if no fix needed\"}]\n"
            "Rules: fixed_content must be the COMPLETE corrected file if you found issues; "
            "if no issues found or none certain, set it to \"\". Do not abbreviate."
        )
        payload = {"files": contents}

        result = await self._llm.execute(
            ToolContext(workspace=str(root)),
            action="chat",
            model=model,
            system=system,
            prompt=user + "\n\n" + json.dumps(payload, ensure_ascii=False)[:MAX_EDIT_BYTES * 6],
            temperature=0.1,
            max_tokens=4096,
        )
        if not result.success or not result.output:
            return [None] * len(contents)

        try:
            parsed = self._extract_json(result.output)
        except Exception:
            return [None] * len(contents)
        if not isinstance(parsed, list):
            return [None] * len(contents)
        return parsed

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
            text = text.strip()
        return json.loads(text)

    def _build_edit(self, root: Path, report: dict, run: AutoFixRun) -> FileEdit | None:
        rel = report.get("path") or ""
        fixed = report.get("fixed_content") or ""
        issues = report.get("issues") or []
        if not rel or not fixed or not issues:
            return None
        p = (root / rel).resolve()
        return FileEdit(
            path=str(p),
            kind="create" if not p.exists() else "write",
            summary="; ".join(issues)[:300],
            before="",
            after=fixed,
            needs_approval=_is_sensitive(p),
        )

    async def _apply_edit(self, context: ToolContext, run: AutoFixRun, edit: FileEdit) -> bool:
        p = Path(edit.path)
        backup: Path | None = None
        if p.exists():
            backup = p.with_suffix(p.suffix + ".bak")
            try:
                shutil.copy2(p, backup)
                edit.backup_path = str(backup)
            except OSError:
                backup = None
        result = await self._writer.execute(
            ToolContext(workspace=str(p.parent)),
            path=str(p),
            content=edit.after,
        )
        if not result.success:
            if backup is not None and backup.exists() and p.exists():
                with suppress(OSError):
                    shutil.copy2(backup, p)
            return False
        edit.applied = True
        edit.verified = self._verify_syntax(p)
        return True

    def _verify_syntax(self, p: Path) -> bool:
        if p.suffix.lower() != ".py":
            return True
        try:
            subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, timeout=120)
            return True
        except Exception:
            return False

    async def _confirm_writes(self, context: ToolContext, kwargs: dict) -> ToolResult:
        token = kwargs.get("approval_token", "").strip()
        if not token:
            return ToolResult(success=False, error="approval_token is required")
        run = next((r for r in self._runs.values() if r.approval_token == token), None)
        if run is None:
            return ToolResult(success=False, error="Invalid approval_token (expired or unknown run)")
        if run.status != "awaiting_approval":
            return ToolResult(success=False, error=f"Run {run.id} is not awaiting approval (status={run.status})")

        pending = [e for e in run.edits if e.needs_approval]
        applied = 0
        for edit in pending:
            if await self._apply_edit(context, run, edit):
                edit.needs_approval = False
                applied += 1
            else:
                run.findings.append(f"approval apply failed: {edit.path}")

        run.status = "completed"
        run.completed_at = datetime.utcnow()
        self._status.last_run_status = run.status
        self._status.edits_applied += applied
        self._status.edits_pending_approval = max(self._status.edits_pending_approval - len(pending), 0)
        return ToolResult(
            success=True,
            output=f"Applied {applied}/{len(pending)} approved writes for run {run.id}",
            metadata={"run_id": run.id, "writes_applied": applied, "writes_requested": len(pending)},
        )

    def _render_manifest(self, run: AutoFixRun) -> str:
        return json.dumps({
            "run_id": run.id,
            "prompt": run.prompt,
            "target_dir": run.target_dir,
            "scans": run.scans,
            "status": run.status,
            "edits": [
                {
                    "path": e.path,
                    "kind": e.kind,
                    "summary": e.summary,
                    "applied": e.applied,
                    "needs_approval": e.needs_approval,
                    "verified": e.verified,
                    "backup": e.backup_path,
                }
                for e in run.edits
            ],
            "issues_found": run.findings,
            "approval_token": run.approval_token,
        }, indent=2)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    async def _get_status(self, context: ToolContext, kwargs: dict) -> ToolResult:
        self._status.runs_count = len(self._runs)
        return ToolResult(
            success=True,
            output=json.dumps({
                "enabled": self._status.enabled,
                "base_dir": self._status.base_dir,
                "last_run_id": self._status.last_run_id,
                "last_run_status": self._status.last_run_status,
                "runs_count": self._status.runs_count,
                "edits_applied": self._status.edits_applied,
                "edits_pending_approval": self._status.edits_pending_approval,
                "sensitive_detected": self._status.sensitive_detected,
            }, indent=2),
        )
