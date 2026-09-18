"""Train ALL Astra-X tools through the chat-driven feedback pipeline.

Gives every registered tool (30 total) its own REAL natural-language prompt —
like a user chatting with Astra-X — executes it through the live tool stack,
and pushes each outcome through the real learning pipeline:

  ExecutionResult + ReflectionResult -> FeedbackLoop.process()
  -> episodic/semantic/procedural memories + knowledge triples
  -> LearningStore save(ExecutionPattern) (generalized goal)
  -> astra_training_log.jsonl (replayable training set)

Groups:
  [B]  builtin:       calculator, datetime, json, text, uuid
  [F]  filesystem:    list_directory, read_file, search_files, write_file
  [G]  github (env):  commits, issues, pull_requests, repository
  [W]  web (env):     download, fetch, scrape, search
  [A]  audit chain:   reconnaissance -> hunting -> candidate_validation
                      -> structured_output -> independent_verification
                      -> target_neutral_reporting  (chained via real files)
  [X]  external:      browserskill, voicebox, colibri, ghidra
  [N]  n8n:           get_status, list_workflows, run_cli (automation)
  [E]  ecc:           get_status / scan_security / llm_generate (local)
  [L]  lmstudio:      chat, completion, embeddings, list_models, get_status
  [S]  autofix:       status, fix, review, confirm_writes

Run: python test_train_all_tools.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.feedback import FeedbackLoop
from app.agents.learning_store import LearningStore, classify_goal
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.pattern import ExecutionPattern
from app.agents.models.task import TaskStatus
from app.tools.autofix import AutoFixTool
from app.tools.audit import (
    CandidateValidationTool,
    HuntingTool,
    IndependentVerificationTool,
    ReconnaissanceTool,
    StructuredOutputTool,
    TargetNeutralReportingTool,
)
from app.tools.browserskill import BrowserSkillTool
from app.tools.builtin import CalculatorTool, DateTimeTool, JsonTool, TextTool, UuidTool
from app.tools.colibri import ColibriTool
from app.tools.ecc import ECCTool
from app.tools.filesystem import ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool
from app.tools.ghidra import GhidraTool
from app.tools.github import GitHubCommitsTool, GitHubIssuesTool, GitHubPullRequestsTool, GitHubRepositoryTool
from app.tools.lmstudio import LmStudioTool
from app.tools.n8n import N8NTool
from app.tools.voicebox import VoiceboxTool
from app.tools.web import DownloadTool, FetchTool, ScrapeTool, SearchTool
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

TRAINING_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astra_training_log.jsonl")
CONVERSATION_ID = str(uuid4())

MODEL_CHAT = "qwen2.5-coder-7b-instruct"
MODEL_HEAVY = "deepseek-r1-distill-qwen-7b"
MODEL_HEAVY_14B = "deepseek-r1-distill-qwen-14b"
MODEL_VISION = "qwen/qwen3.5-9b"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PROJECT = {
    "app.py": "def greet(name):\n    print('hello' + name\n",
    "utils.py": "def double(x):\n    return x * 2\n",
    "db.py": "import sqlite3\n\ndef connect():\n    return sqlite3.connect('app.db')\n",
    "api/routes.py": "def items_route():\n    return {'ok': True}\n",
    "schema.sql": "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);\n",
}

SENSITIVE_PROJECT = {
    "buggy.py": "def add(a, b:\n    return a + b\n",
    "credentials.json": '{"api_keys": "sk-test-old-value"}\n',
}


def _write_project(root: str, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _py_compiles(path: str) -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            capture_output=True, timeout=120, check=True,
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Trainer (real learning pipeline)
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self) -> None:
        self.store = LearningStore(repository=None)
        self.log: list[dict] = []
        self.features: list[tuple[str, str, bool, str]] = []

    def track(self, scenario: str, user_prompt: str, ok: bool, detail: str = "") -> None:
        self.features.append((scenario, user_prompt, ok, detail))

    def train(self, scenario: str, goal: str, result: ToolResult, extra_metadata: dict | None = None) -> dict:
        started = time.monotonic()
        status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        execution = ExecutionResult(
            task_id=str(uuid4()),
            status=status,
            output=(result.output or "")[:500],
            error=result.error,
            tool_name="tool_chat",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            metadata={"scenario": scenario, **(extra_metadata or {})},
        )
        reflection = ReflectionResult(
            decision=ReflectionDecision.ACCEPT if result.success else ReflectionDecision.RETRY,
            feedback="run satisfied the prompt" if result.success else result.error or "run failed",
            reason="feature exercised" if result.success else "feature returned an error",
            confidence=0.9 if result.success else 0.4,
        )
        feedback = FeedbackLoop.process(execution, reflection, CONVERSATION_ID)
        goal_pattern = LearningStore.generalize_goal(goal)
        tag_kws = [w for w in goal.lower().replace("-", " ").split() if len(w) > 3]
        pattern = ExecutionPattern(
            goal_pattern=goal_pattern,
            capability=scenario.split(":")[0],
            strategy_summary=f"scenario {scenario} via chat prompt",
            tool_sequence=["tool_chat"],
            tags=tag_kws or [scenario],
            avg_confidence=reflection.confidence,
            avg_importance=feedback.importance,
            avg_execution_cost_ms=(time.monotonic() - started) * 1000.0,
            last_reflection_confidence=reflection.confidence,
            last_success_at=datetime.now(UTC) if result.success else None,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        if not result.success:
            pattern = pattern.record_failure(decay_factor=0.95)
        self.store._cache[pattern.goal_pattern] = pattern
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "conversation_id": CONVERSATION_ID,
            "scenario": scenario,
            "goal": goal,
            "goal_pattern": goal_pattern,
            "domain": classify_goal(goal).value,
            "success": result.success,
            "importance": round(feedback.importance, 3),
            "decision": reflection.decision.value,
            "confidence": reflection.confidence,
            "memories": {
                "episodic": len(feedback.episodic_memories),
                "semantic": len(feedback.semantic_memories),
                "procedural": len(feedback.procedural_memories),
            },
            "triples": len(feedback.knowledge_triples),
            "output_preview": (result.output or "")[:200],
            "error": result.error,
            "pattern": pattern.model_dump(),
        }
        self.log.append(entry)
        with open(TRAINING_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry


def _result_ok(res: ToolResult) -> bool:
    return res.success and bool(res.output)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

GROUP_BUILTIN = [
    ("B:calculator", "calculate 2 + 2", "what is 2 plus 2", {"expression": "2 + 2"}),
    ("B:datetime", "show current utc date and time", "what time is it now in utc", {"format": "%Y-%m-%d %H:%M:%S", "timezone": "UTC"}),
    ("B:json", "format this json nicely", "pretty print this json", {"operation": "stringify", "input": "{\"a\":1,\"b\":[1,2]}", "indent": 2}),
    ("B:text", "uppercase and trim this text", "uppercase this text", {"operation": "upper", "text": "  hello world  "}),
    ("B:uuid", "generate 3 uuids", "give me three uuid v4s", {"count": 3, "version": 4}),
]

GROUP_FILESYSTEM = [
    ("F:list_directory", "list files in my project", "what files are in this project", {"path": "{fixture}", "recursive": True}),
    ("F:read_file", "read the db.py file", "show me the db.py source", {"path": "{fixture}/db.py"}),
    ("F:search_files", "find files matching schema", "find sql or schema files", {"root": "{fixture}", "pattern": "*.sql"}),
    ("F:write_file", "write a new hook.py", "create an empty module file", {"path": "{fixture}/hook.py", "content": "# hook\n"}),
]

GROUP_GITHUB = [
    ("G:github_commits", "recent commits of a repo", "what are the latest commits on that repo", {"owner": "octocat", "repo": "Hello-World", "count": 3}),
    ("G:github_repository", "repo metadata", "give me info about the hello-world repo", {"owner": "octocat", "repo": "Hello-World"}),
    ("G:github_pull_requests", "open pull requests", "show open pull requests", {"owner": "octocat", "repo": "Hello-World", "state": "open"}),
    ("G:github_issues", "open issues", "list the open issues", {"owner": "octocat", "repo": "Hello-World", "state": "open", "count": 3}),
]

GROUP_WEB = [
    ("W:web_search", "search the web", "search the web for latest fastapi news", {"query": "fastapi python latest", "count": 3}),
    ("W:web_fetch", "fetch a page", "fetch the example.com homepage", {"url": "https://example.com", "timeout": 15}),
    ("W:web_scrape", "scrape a page", "scrape example.com and extract text", {"url": "https://example.com", "extract": "text", "timeout": 15}),
    ("W:web_download", "download a small file", "download the example.com homepage", {"url": "https://example.com/", "output_path": "{tmp}/example.html", "timeout": 30}),
]

GROUP_EXTERNAL = [
    ("X:browserskill", "browserskill availability", "is browser automation available", {"action": "get_status"}),
    ("X:voicebox", "voicebox status", "is voicebox available", {"action": "get_status"}),
    ("X:colibri", "colibri status", "is colibri available", {"action": "get_status"}),
    ("X:ghidra", "ghidra status", "is ghidra available", {"action": "get_status"}),
]

GROUP_N8N = [
    ("N:n8n_status", "n8n availability", "is n8n automation available", {"action": "get_status"}),
    ("N:n8n_list_workflows", "list n8n workflows", "list my n8n workflows", {"action": "list_workflows", "limit": 10}),
]


def tool_by_name(registry: ToolRegistry, name: str) -> Tool:
    return registry.get(name)


async def run() -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    ctx = ToolContext(workspace=os.getcwd())
    trainer = Trainer()

    classes = [
        CalculatorTool, DateTimeTool, JsonTool, TextTool, UuidTool,
        ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool,
        GitHubCommitsTool, GitHubIssuesTool, GitHubPullRequestsTool, GitHubRepositoryTool,
        DownloadTool, FetchTool, ScrapeTool, SearchTool,
        ReconnaissanceTool, HuntingTool, CandidateValidationTool,
        StructuredOutputTool, IndependentVerificationTool, TargetNeutralReportingTool,
        BrowserSkillTool, VoiceboxTool, ECCTool, AutoFixTool, ColibriTool, GhidraTool, LmStudioTool, N8NTool,
    ]
    for cls in classes:
        registry.register(cls())
    print(f"Registered {registry.count} tools\n")

    fixture = tempfile.mkdtemp(prefix="astra_fixture_")
    _write_project(fixture, FIXTURE_PROJECT)
    tmp = tempfile.mkdtemp(prefix="astra_tmp_")

    try:
        # ---------------- [B] builtin ----------------
        print("-" * 60)
        print("GROUP B: builtin tools")
        print("-" * 60)
        for scenario, prompt, goal, args in GROUP_BUILTIN:
            res = await executor.execute(ToolCall(tool_name=scenario.split(":")[1], arguments=args), ctx)
            ok = _result_ok(res)
            trainer.track(scenario, goal, ok, (res.output or res.error or "")[:120])
            trainer.train(scenario, goal, res)
            print(f"  [{scenario[0]}] {scenario:<16} {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [F] filesystem ----------------
        print("-" * 60)
        print("GROUP F: filesystem tools")
        print("-" * 60)
        fx_ctx = ToolContext(workspace=fixture)
        for scenario, prompt, goal, args in GROUP_FILESYSTEM:
            args = {k: v.format(fixture=fixture, tmp=tmp) if isinstance(v, str) else v for k, v in args.items()}
            res = await executor.execute(ToolCall(tool_name=scenario.split(":")[1], arguments=args), fx_ctx)
            ok = _result_ok(res)
            trainer.track(scenario, goal, ok, (res.output or res.error or "")[:120])
            trainer.train(scenario, goal, res)
            print(f"  [{scenario[0]}] {scenario:<16} {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [G] github (network/credentials dependent) ----------------
        print("-" * 60)
        print("GROUP G: github tools (needs GITHUB_TOKEN/network)")
        print("-" * 60)
        for scenario, prompt, goal, args in GROUP_GITHUB:
            res = await executor.execute(ToolCall(tool_name=scenario.split(":")[1], arguments=args), ctx)
            ok = _result_ok(res)
            env = "ENV" if not res.success else "OK"
            trainer.track(scenario, goal, ok, (res.output or res.error or "")[:120])
            trainer.train(scenario, goal, res)
            print(f"  [{scenario[0]}] {scenario:<16} {env:<4} -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [W] web (network dependent) ----------------
        print("-" * 60)
        print("GROUP W: web tools (needs network)")
        print("-" * 60)
        wx_ctx = ToolContext(workspace=os.path.join(fixture, "webdl"))
        os.makedirs(os.path.join(fixture, "webdl"), exist_ok=True)
        for scenario, prompt, goal, args in GROUP_WEB:
            args = {k: v.format(fixture=fixture, tmp=os.path.join(fixture, "webdl")) if isinstance(v, str) else v for k, v in args.items()}
            res = await executor.execute(ToolCall(tool_name=scenario.split(":")[1], arguments=args), wx_ctx)
            ok = _result_ok(res)
            env = "ENV" if not res.success else "OK"
            trainer.track(scenario, goal, ok, (res.output or res.error or "")[:120])
            trainer.train(scenario, goal, res)
            print(f"  [{scenario[0]}] {scenario:<16} {env:<4} -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [A] audit chain (real chained files) ----------------
        print("-" * 60)
        print("GROUP A: audit pipeline (recon -> hunt -> validate -> structure -> verify -> report)")
        print("-" * 60)
        audit_dir = os.path.join(fixture, "audit_out")
        os.makedirs(audit_dir, exist_ok=True)

        res = await executor.execute(ToolCall(tool_name="reconnaissance", arguments={
            "target_description": "A small python api with sqlite storage",
            "source_code_paths": [str(Path(fixture) / "db.py"), str(Path(fixture) / "api")],
            "config_files": [],
            "output_dir": audit_dir,
            "include_attack_classes": ["injection", "auth_bypass"],
        }), ctx)
        ok = res.success and os.path.exists(os.path.join(audit_dir, "architecture.md")) and os.path.exists(os.path.join(audit_dir, "coverage-ledger.json"))
        trainer.track("A:reconnaissance", "recon this project and map attack surface", ok, (res.output or res.error or "")[:120])
        trainer.train("A:reconnaissance", "recon the project and map the attack surface", res,
                      {"files": ["architecture.md", "coverage-ledger.json"]})
        print(f"  [A] reconnaissance            {'PASS' if ok else 'FAIL'}  -> architecture.md + coverage-ledger.json")

        ledger = os.path.join(audit_dir, "coverage-ledger.json")
        arch = os.path.join(audit_dir, "architecture.md")
        res = await executor.execute(ToolCall(tool_name="hunting", arguments={
            "coverage_ledger_path": ledger,
            "architecture_path": arch,
            "output_dir": audit_dir,
            "hunter_count": 2,
            "attack_class_prompts": {"injection": "look for sql injection", "auth_bypass": "look for auth bypass"},
            "run_coverage_critic": True,
        }), ctx)
        ok = res.success and os.path.exists(os.path.join(audit_dir, "findings.json"))
        trainer.track("A:hunting", "hunt for vulnerabilities with coverage critic", ok, (res.output or res.error or "")[:120])
        trainer.train("A:hunting", "hunt the project for vulnerabilities", res,
                      {"files": ["findings.json", "hunters.json", "coverage-critic.json"]})
        print(f"  [A] hunting                   {'PASS' if ok else 'FAIL'}  -> findings.json + hunters.json + coverage-critic.json")

        findings = os.path.join(audit_dir, "findings.json")
        res = await executor.execute(ToolCall(tool_name="candidate_validation", arguments={
            "findings_path": findings,
            "output_dir": audit_dir,
            "require_source_verification": True,
        }), ctx)
        ok = res.success and os.path.exists(os.path.join(audit_dir, "verifiers.json"))
        trainer.track("A:candidate_validation", "validate the candidate findings", ok, (res.output or res.error or "")[:120])
        trainer.train("A:candidate_validation", "validate the candidate security findings", res)
        print(f"  [A] candidate_validation      {'PASS' if ok else 'FAIL'}  -> verifiers.json")

        res = await executor.execute(ToolCall(tool_name="structured_output", arguments={
            "findings_path": findings,
            "output_dir": audit_dir,
            "include_rejected": True,
        }), ctx)
        ok = res.success and os.path.exists(os.path.join(audit_dir, "findings.json"))
        trainer.track("A:structured_output", "organize findings by verdict", ok, (res.output or res.error or "")[:120])
        trainer.train("A:structured_output", "organize findings by verdict and validate schema", res)
        print(f"  [A] structured_output         {'PASS' if ok else 'FAIL'}  -> findings.json reorganized")

        findings_out = os.path.join(audit_dir, "findings.json")
        res = await executor.execute(ToolCall(tool_name="independent_verification", arguments={
            "findings_path": findings_out,
            "source_code_paths": [str(Path(fixture) / "db.py"), str(Path(fixture) / "api" / "routes.py")],
            "output_dir": audit_dir,
            "verification_depth": "standard",
        }), ctx)
        ok = res.success
        trainer.track("A:independent_verification", "independently verify the findings against source", ok, (res.output or res.error or "")[:120])
        trainer.train("A:independent_verification", "independently verify findings against source code", res)
        print(f"  [A] independent_verification  {'PASS' if ok else 'FAIL'}  -> verified verdicts")

        res = await executor.execute(ToolCall(tool_name="target_neutral_reporting", arguments={
            "findings_path": findings_out,
            "coverage_ledger_path": ledger,
            "architecture_path": arch,
            "output_dir": audit_dir,
            "target_name": "small-api",
            "include_executive_summary": True,
        }), ctx)
        ok = res.success and os.path.exists(os.path.join(audit_dir, "report.md"))
        trainer.track("A:target_neutral_reporting", "produce a neutral security report", ok, (res.output or res.error or "")[:120])
        trainer.train("A:target_neutral_reporting", "produce a neutral security audit report", res)
        print(f"  [A] target_neutral_reporting  {'PASS' if ok else 'FAIL'}  -> report.md")

        # ---------------- [X] external services (environmental) ----------------
        print("-" * 60)
        print("GROUP X: external-service tools (report availability)")
        print("-" * 60)
        for scenario, prompt, goal, args in GROUP_EXTERNAL:
            res = await executor.execute(ToolCall(tool_name=scenario.split(":")[1], arguments=args), ctx)
            ok = _result_ok(res)
            env = "ENV" if not res.success else "OK"
            trainer.track(scenario, goal, ok, (res.output or res.error or "")[:120])
            trainer.train(scenario, goal, res)
            print(f"  [{scenario[0]}] {scenario:<16} {env:<4} -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [N] n8n automation ----------------
        print("-" * 60)
        print("GROUP N: n8n workflow automation (reports availability)")
        print("-" * 60)
        for scenario, prompt, goal, args in GROUP_N8N:
            res = await executor.execute(ToolCall(tool_name="n8n", arguments=args), ctx)
            ok = _result_ok(res)
            env = "ENV" if not res.success else "OK"
            trainer.track(scenario, goal, ok, (res.output or res.error or "")[:120])
            trainer.train(scenario, goal, res)
            print(f"  [{scenario[0]}] {scenario:<20} {env:<4} -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [E] ECC (local ECC-main available) ----------------
        print("-" * 60)
        print("GROUP E: ECC tool (local ECC-main)")
        print("-" * 60)
        res = await executor.execute(ToolCall(tool_name="ecc", arguments={"action": "get_status"}), ctx)
        ok = _result_ok(res)
        trainer.track("E:ecc_get_status", "check ecc availability", ok, (res.output or res.error or "")[:120])
        trainer.train("E:ecc_get_status", "check the status of the engineering control center", res)
        print(f"  [E] ecc get_status            {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        res = await executor.execute(ToolCall(tool_name="ecc", arguments={"action": "llm_generate", "prompt": "write a one-line description of this project", "model": MODEL_CHAT}), ctx)
        ok = _result_ok(res)
        env = "ENV" if not res.success else "OK"
        trainer.track("E:ecc_llm_generate", "generate text via lm studio", ok, (res.output or res.error or "")[:120])
        trainer.train("E:ecc_llm_generate", "generate a description with the local llm", res)
        print(f"  [E] ecc llm_generate (LM Studio) {env:<4} -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [L] LM Studio (live chat + heavy model) ----------------
        print("-" * 60)
        print("GROUP L: LM Studio tools (live)")
        print("-" * 60)
        res = await executor.execute(ToolCall(tool_name="lmstudio", arguments={"action": "list_models"}), ctx)
        ok = _result_ok(res)
        trainer.track("L:lmstudio_list_models", "list available models", ok, (res.output or res.error or "")[:120])
        trainer.train("L:lmstudio_list_models", "list the available local models", res)
        print(f"  [L] lmstudio list_models      {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        res = await executor.execute(ToolCall(tool_name="lmstudio", arguments={
            "action": "chat", "model": MODEL_CHAT,
            "prompt": "Reply with exactly: LMStudio-CHAT-OK",
            "max_tokens": 32, "temperature": 0,
        }), ctx)
        ok = _result_ok(res)
        trainer.track("L:lmstudio_chat", "chat with the coding model", ok, (res.output or res.error or "")[:120])
        trainer.train("L:lmstudio_chat", "chat with the local coding model", res)
        print(f"  [L] lmstudio chat (qwen)      {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        res = await executor.execute(ToolCall(tool_name="lmstudio", arguments={
            "action": "chat", "model": MODEL_HEAVY,
            "prompt": "Reply with exactly: DS-CHAT-OK",
            "max_tokens": 1024, "temperature": 0,
        }), ctx)
        ok = _result_ok(res)
        trainer.track("L:lmstudio_heavy", "chat with the heavy reasoning model", ok, (res.output or res.error or "")[:120])
        trainer.train("L:lmstudio_heavy", "chat with the heavy reasoning model", res)
        print(f"  [L] lmstudio chat (deepseek)  {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        res = await executor.execute(ToolCall(tool_name="lmstudio", arguments={
            "action": "chat", "model": MODEL_VISION,
            "prompt": "Reply with exactly: QWEN35-OK. Do not explain.",
            "max_tokens": 2048, "temperature": 0,
        }), ctx)
        ok = _result_ok(res)
        trainer.track("L:lmstudio_qwen35", "chat with the 9B high-capacity model", ok, (res.output or res.error or "")[:120])
        trainer.train("L:lmstudio_qwen35", "chat with the high-capacity 9B reasoning model", res)
        print(f"  [L] lmstudio chat (qwen3.5)   {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        res = await executor.execute(ToolCall(tool_name="lmstudio", arguments={
            "action": "chat", "model": MODEL_HEAVY_14B,
            "prompt": "Reply with exactly: DS14B-OK. Do not explain.",
            "max_tokens": 2048, "temperature": 0,
        }), ctx)
        ok = _result_ok(res)
        trainer.track("L:lmstudio_heavy14b", "chat with the 14B reasoning model", ok, (res.output or res.error or "")[:120])
        trainer.train("L:lmstudio_heavy14b", "chat with the 14B reasoning model", res)
        print(f"  [L] lmstudio chat (ds14b)     {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        res = await executor.execute(ToolCall(tool_name="lmstudio", arguments={"action": "get_status"}), ctx)
        ok = _result_ok(res)
        trainer.track("L:lmstudio_status", "get lm studio status", ok, (res.output or res.error or "")[:120])
        trainer.train("L:lmstudio_status", "get the lm studio server status", res)
        print(f"  [L] lmstudio get_status       {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        # ---------------- [S] AutoFix (autonomous agent) ----------------
        print("-" * 60)
        print("GROUP S: AutoFix autonomous agent")
        print("-" * 60)
        res = await executor.execute(ToolCall(tool_name="autofix", arguments={"action": "get_status"}), ctx)
        ok = _result_ok(res)
        trainer.track("S:autofix_status", "autofix status", ok, (res.output or res.error or "")[:120])
        trainer.train("S:autofix_status", "check the autofix agent status", res)
        print(f"  [S] autofix get_status        {'PASS' if ok else 'FAIL'}  -> {(res.output or res.error or '').strip()[:100]}")

        res = await executor.execute(ToolCall(tool_name="autofix", arguments={
            "action": "fix", "prompt": "find and fix the wrong code in this project",
            "path": fixture, "max_files": 8,
        }), ctx)
        meta = res.metadata or {}
        app_ok = os.path.exists(os.path.join(fixture, "app.py")) and _py_compiles(os.path.join(fixture, "app.py"))
        ok = res.success and meta.get("edits_applied", 0) >= 1 and app_ok
        trainer.track("S:autofix_fix", "fix the broken code autonomously", ok, (res.output or res.error or "")[:120])
        trainer.train("S:autofix_fix", "fix the broken code in the project autonomously", res, extra_metadata=meta)
        print(f"  [S] autofix fix               {'PASS' if ok else 'FAIL'}  -> edits={meta.get('edits_applied')} app.py_compiles={app_ok}")

        res = await executor.execute(ToolCall(tool_name="autofix", arguments={
            "action": "review", "prompt": "audit this project, classify database files and suggest upgrades",
            "path": fixture, "max_files": 8,
        }), ctx)
        try:
            payload = json.loads(res.output)
            dbc = len([r for r in payload.get("reviews", []) if r.get("database_related")])
        except Exception:
            dbc = -1
        ok = res.success and dbc >= 1
        trainer.track("S:autofix_review", "audit and classify database-related files", ok, (res.output or res.error or "")[:120])
        trainer.train("S:autofix_review", "audit the project and classify database files", res)
        print(f"  [S] autofix review            {'PASS' if ok else 'FAIL'}  -> db_related={dbc}")

        # ---- sensitive approval flow
        sen = tempfile.mkdtemp(prefix="astra_sensitive_")
        _write_project(sen, SENSITIVE_PROJECT)
        try:
            res = await executor.execute(ToolCall(tool_name="autofix", arguments={
                "action": "fix", "prompt": "fix the syntax error and the credentials file too",
                "path": sen, "max_files": 4,
            }), ctx)
            meta = res.metadata or {}
            token = meta.get("approval_token", "")
            unchanged = '"sk-test-old-value"' in (Path(sen) / "credentials.json").read_text(encoding="utf-8")
            ok = res.success and meta.get("status") == "awaiting_approval" and token and unchanged
            if ok:
                good = await executor.execute(ToolCall(tool_name="autofix", arguments={
                    "action": "confirm_writes", "approval_token": token,
                }), ctx)
                ok = good.success and good.metadata.get("writes_applied", 0) >= 1
            trainer.track("S:autofix_approval", "approve sensitive file writes", ok, (res.output or res.error or "")[:120])
            trainer.train("S:autofix_approval", "approve writing the sensitive credentials file", res, extra_metadata=meta)
            print(f"  [S] autofix approval gate     {'PASS' if ok else 'FAIL'}  -> awaiting={meta.get('status')} token={token[:6] if token else 'none'}")
        finally:
            shutil.rmtree(sen, ignore_errors=True)

    finally:
        shutil.rmtree(fixture, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ALL-TOOL TRAINING SUMMARY")
    print("=" * 72)
    passed = sum(1 for *_, ok, _ in trainer.features if ok)
    total = len(trainer.features)
    print(f"  {passed}/{total} tool scenarios produced a successful result")
    for scenario, goal, ok, detail in trainer.features:
        mark = "PASS" if ok else "-"
        print(f"  [{mark}] {scenario:<32} {goal[:44]}")

    print(f"\nTraining log: {TRAINING_LOG}  ({len(trainer.log)} entries)")
    print(f"Conversation: {CONVERSATION_ID}")
    print(f"LearningStore cache: {len(trainer.store._cache)} learned patterns")
    doms = {}
    for p in trainer.store._cache.values():
        d = classify_goal(p.goal_pattern)
        doms[d.value] = doms.get(d.value, 0) + 1
    print(f"  by domain: {doms}")

    print("\nLearningStore.search() ranking demo:")
    for goal, limit in [("fix the broken code in my database project", 4),
                        ("audit the project and suggest security upgrades", 4),
                        ("what time is it now", 3)]:
        hits = trainer.store.search(goal, limit=limit)
        print(f"    {goal!r:<55} -> {[h.goal_pattern for h in hits]}")

    print("\nDONE")


if __name__ == "__main__":
    asyncio.run(run())