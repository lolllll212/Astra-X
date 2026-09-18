"""Chat-driven Astra-X feature-test + training harness.

Gives the Astra-X agent (the AutoFix autonomous repair agent) a series of
DIFFERENT natural-language prompts, exactly like a user chatting with it,
so every feature is exercised against the live tool stack:

  [1] get_status                     - status reporting (no LLM)
  [2] fix (explicit path)            - real repair via live LLM + write tool
  [3] fix (prompt folder discovery)  - folder name found from the prompt
  [4] review                         - read-only audit + DB-relevance classification
  [5] sensitive-file approval flow   - fix + confirm_writes with approval token
  [6] heavy model pass               - use_heavy_model (deepseek-r1-distill-qwen-7b)
  [7] custom model override          - explicit model parameter
  [8] validation guardrail           - missing prompt rejected

After every run, the outcome is pushed through the REAL learning pipeline:

  * ExecutionResult + ReflectionResult built from the tool result
  * FeedbackLoop.process()  -> importance + episodic/semantic/procedural memories
  * LearningStore.save(ExecutionPattern) via generalize_goal()
  * Training log appended to astra_training_log.jsonl (replayable)

Run: python test_train_astra_chat.py
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
from app.agents.learning_store import LearningStore
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.pattern import ExecutionPattern
from app.agents.models.task import TaskStatus
from app.tools.autofix import AutoFixTool
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult

USERPROFILE = os.environ.get("USERPROFILE") or str(Path.home())
TRAINING_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astra_training_log.jsonl")
CONVERSATION_ID = str(uuid4())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUGGY_PROJECT_FILES = {
    "app.py": "def greet(name):\n    print('hello' + name\n",
    "utils.py": "def double(x):\n    return x * 2\n",
    "db.py": "import sqlite3\n\ndef connect():\n    return sqlite3.connect('app.db')\n",
}

REVIEW_PROJECT_FILES = {
    "db/repository.py": "import sqlite3\n\nclass Repository:\n    def fetch(self):\n        conn = sqlite3.connect('app.db')\n        return conn.execute('select * from items').fetchall()\n",
    "api/routes.py": "def items_route():\n    return {'ok': True}\n",
    "junk.harness.py": "print('test harness scratch')\n",
    "schema.sql": "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);\n",
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
# Learning pipeline
# ---------------------------------------------------------------------------

class Trainer:
    """Feeds every chat run through the real feedback loop + learning store."""

    def __init__(self) -> None:
        self.store = LearningStore(repository=None)
        self.log: list[dict] = []

    def train(
        self,
        goal: str,
        scenario: str,
        result: ToolResult,
        tool_name: str = "autofix",
        extra_metadata: dict | None = None,
    ) -> dict:
        started = time.monotonic()
        status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED

        execution = ExecutionResult(
            task_id=str(uuid4()),
            status=status,
            output=(result.output or "")[:500],
            error=result.error,
            tool_name=tool_name,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            metadata={
                "scenario": scenario,
                **(extra_metadata or {}),
            },
        )
        decision = (
            ReflectionDecision.ACCEPT
            if result.success else ReflectionDecision.RETRY
        )
        reflection = ReflectionResult(
            decision=decision,
            feedback="run satisfied the prompt" if result.success else result.error or "run failed",
            reason="feature exercised successfully" if result.success else "feature returned an error",
            confidence=0.9 if result.success else 0.4,
        )

        feedback = FeedbackLoop.process(execution, reflection, CONVERSATION_ID)

        goal_pattern = LearningStore.generalize_goal(goal)
        tag_kws = [w for w in goal.lower().replace("-", " ").split() if len(w) > 3]
        pattern = ExecutionPattern(
            goal_pattern=goal_pattern,
            capability=tool_name,
            strategy_summary=f"Used {tool_name} action for scenario: {scenario}",
            tool_sequence=[tool_name],
            tags=tag_kws or ["autofix", "chat"],
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
            "success": result.success,
            "importance": round(feedback.importance, 3),
            "decision": decision.value,
            "confidence": reflection.confidence,
            "memories": {
                "episodic": len(feedback.episodic_memories),
                "semantic": len(feedback.semantic_memories),
                "procedural": len(feedback.procedural_memories),
            },
            "triples": len(feedback.knowledge_triples),
            "output_preview": (result.output or "")[:200],
            "error": result.error,
            "metadata": execution.metadata,
            "pattern": pattern.model_dump(),
        }
        self.log.append(entry)
        with open(TRAINING_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return entry


# ---------------------------------------------------------------------------
# Feature scenarios
# ---------------------------------------------------------------------------

async def run() -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    tool = AutoFixTool()
    registry.register(tool)
    ctx = ToolContext(workspace=os.getcwd())
    trainer = Trainer()

    print("=" * 72)
    print("ASTRA-X CHAT FEATURE TEST + TRAINING (live LM Studio)")
    print("=" * 72)

    results: list[tuple[int, str, bool]] = []

    # [1] get_status — status reporting, no LLM
    res = await executor.execute(ToolCall(tool_name=tool.name, arguments={"action": "get_status"}), ctx)
    ok = res.success and '"runs_count"' in res.output
    results.append((1, "get_status", ok))
    trainer.train("show me the current status of the agent", "get_status", res)
    print(f"\n[1] get_status -> {'PASS' if ok else 'FAIL'}")
    print(f"    {res.output.strip()}")

    # [2] fix with explicit path — live LLM repair of a temp project
    root = tempfile.mkdtemp(prefix="astra_chat_fix_")
    _write_project(root, BUGGY_PROJECT_FILES)
    try:
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments={
            "action": "fix",
            "prompt": "please find and fix the wrong code in my small python project",
            "path": root,
            "max_files": 8,
        }), ctx)
        meta = res.metadata or {}
        app_ok = os.path.exists(os.path.join(root, "app.py")) and _py_compiles(os.path.join(root, "app.py"))
        ok = res.success and meta.get("edits_applied", 0) >= 1 and app_ok
        results.append((2, "fix (explicit path)", ok))
        trainer.train("fix the broken python functions in my project", "fix_path", res, extra_metadata=meta)
        print(f"\n[2] fix (explicit path) -> {'PASS' if ok else 'FAIL'}")
        print(f"    files_analyzed={meta.get('files_analyzed')} edits_applied={meta.get('edits_applied')} status={meta.get('status')}")
        print(f"    app.py compiles cleanly after repair: {app_ok}")
        bak = Path(root) / "app.py.bak"
        print(f"    backup created: {bak.exists()}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # [3] fix via prompt-driven folder discovery (folder under user profile)
    probe = os.path.join(USERPROFILE, "autofix-discovery-probe")
    if os.path.exists(probe):
        shutil.rmtree(probe, ignore_errors=True)
    os.makedirs(probe, exist_ok=True)
    (Path(probe) / "broken.py").write_text("def echo(txt:\n    return txt\n", encoding="utf-8")
    try:
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments={
            "action": "fix",
            "prompt": "can you fix the broken code in the autofix-discovery-probe folder",
            "max_files": 4,
        }), ctx)
        meta = res.metadata or {}
        ok = res.success and "autofix-discovery-probe" in meta.get("target_dir", "")
        ok = ok and os.path.exists(os.path.join(probe, "broken.py")) and _py_compiles(os.path.join(probe, "broken.py"))
        results.append((3, "fix (folder discovery from prompt)", ok))
        trainer.train("fix the broken code in the autofix-discovery-probe folder", "fix_discovery", res, extra_metadata=meta)
        print(f"\n[3] fix (folder discovery from prompt) -> {'PASS' if ok else 'FAIL'}")
        print(f"    target resolved via prompt: {meta.get('target_dir')}")
        print(f"    broken.py compiles cleanly after repair: {ok}")
    finally:
        shutil.rmtree(probe, ignore_errors=True)

    # [4] review — read-only audit + DB-relevance classification
    root = tempfile.mkdtemp(prefix="astra_chat_review_")
    _write_project(root, REVIEW_PROJECT_FILES)
    try:
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments={
            "action": "review",
            "prompt": "audit this project and classify which files belong to the database layer and suggest upgrades",
            "path": root,
            "max_files": 8,
        }), ctx)
        ok = res.success
        try:
            payload = json.loads(res.output)
            reviews = payload.get("reviews", [])
            db_related = [r for r in reviews if r.get("database_related")]
            ok = ok and len(reviews) >= 3 and len(db_related) >= 1
        except Exception:
            ok = False
        results.append((4, "review + db classification", ok))
        trainer.train("audit the project and classify database files and suggest upgrades", "review", res, extra_metadata=res.metadata)
        print(f"\n[4] review + DB classification -> {'PASS' if ok else 'FAIL'}")
        print(f"    files_analyzed={res.metadata and res.metadata.get('files_analyzed')}")
        if res.success:
            try:
                payload = json.loads(res.output)
                print(f"    reviews={len(payload.get('reviews', []))} "
                      f"with_issues={payload.get('summary', {}).get('with_issues')} "
                      f"db_related={payload.get('summary', {}).get('db_related')}")
                print(f"    non_db={payload.get('summary', {}).get('non_db')}")
            except Exception:
                print("    (output not JSON-parsable)")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # [5] sensitive-file approval flow — fix + confirm_writes
    root = tempfile.mkdtemp(prefix="astra_chat_sensitive_")
    _write_project(root, {
        "buggy.py": "def add(a, b:\n    return a + b\n",
        "credentials.json": '{"api_keys": "sk-test-old-value"}\n',
    })
    try:
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments={
            "action": "fix",
            "prompt": "fix the syntax problem and also fix the credentials file if it has an issue",
            "path": root,
            "max_files": 4,
        }), ctx)
        meta = res.metadata or {}
        token = meta.get("approval_token", "")
        creds_before = (Path(root) / "credentials.json").read_text(encoding="utf-8")
        safe_applied = meta.get("edits_applied", 0)
        awaiting = meta.get("status") == "awaiting_approval"
        sensitive_gated = "credentials.json" in str(meta.get("sensitive_files", []))
        unchanged_before_approval = '"sk-test-old-value"' in creds_before
        ok = res.success and awaiting and token and sensitive_gated and safe_applied >= 1 and unchanged_before_approval

        if ok and token:
            bad = await executor.execute(ToolCall(tool_name=tool.name, arguments={
                "action": "confirm_writes", "approval_token": "deadbeef",
            }), ctx)
            ok_write = (not bad.success) and ("Invalid approval_token" in (bad.error or ""))
            good = await executor.execute(ToolCall(tool_name=tool.name, arguments={
                "action": "confirm_writes", "approval_token": token,
            }), ctx)
            ok_write = ok_write and good.success and good.metadata.get("writes_applied", 0) >= 1
            ok = ok and ok_write
        results.append((5, "sensitive-file approval flow", ok))
        trainer.train("fix the credentials file and the broken function", "fix_approval", res, extra_metadata=meta)
        print(f"\n[5] sensitive-file approval flow -> {'PASS' if ok else 'FAIL'}")
        print(f"    edits_applied(safe)={safe_applied} awaiting_approval={awaiting} token={token[:8] if token else 'none'}...")
        print(f"    credentials.json untouched before approval: {unchanged_before_approval}")
        print(f"    credentials.json final: {(Path(root) / 'credentials.json').read_text(encoding='utf-8').strip()}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # [6] heavy model pass — use_heavy_model=True
    root = tempfile.mkdtemp(prefix="astra_chat_heavy_")
    (Path(root) / "heavy.py").write_text("def compute():\n    return [i for i in range(5) if i % 2]]\n", encoding="utf-8")
    try:
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments={
            "action": "fix",
            "prompt": "fix the syntax error in this file",
            "path": root,
            "use_heavy_model": True,
            "max_files": 4,
        }), ctx)
        meta = res.metadata or {}
        ok = res.success and meta.get("edits_applied", 0) >= 1
        checked = os.path.exists(os.path.join(root, "heavy.py")) and _py_compiles(os.path.join(root, "heavy.py"))
        ok = ok and checked
        results.append((6, "heavy model pass", ok))
        trainer.train("fix the syntax error in this file with the heavy model", "fix_heavy", res, extra_metadata=meta)
        print(f"\n[6] heavy model pass (deepseek-r1-distill-qwen-7b) -> {'PASS' if ok else 'FAIL'}")
        print(f"    edits_applied={meta.get('edits_applied')} compiles_clean={checked}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # [7] custom model override
    root = tempfile.mkdtemp(prefix="astra_chat_model_")
    (Path(root) / "calc.py").write_text("def calc(x, y):\n    return x * y\n", encoding="utf-8")
    try:
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments={
            "action": "review",
            "prompt": "review this tiny project",
            "path": root,
            "model": "deepseek-coder-6.7b-instruct",
            "max_files": 4,
        }), ctx)
        ok = res.success
        results.append((7, "custom model override", ok))
        trainer.train("review the tiny calc project", "review_model_override", res, extra_metadata=res.metadata)
        print(f"\n[7] custom model override (deepseek-coder-6.7b) -> {'PASS' if ok else 'FAIL'}")
        print(f"    files_analyzed={res.metadata and res.metadata.get('files_analyzed')} reviews={res.metadata and res.metadata.get('reviews')}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # [8] validation guardrail
    res = await executor.execute(ToolCall(tool_name=tool.name, arguments={"action": "fix"}), ctx)
    ok = (not res.success) and ("prompt is required" in (res.error or ""))
    results.append((8, "validation guardrail", ok))
    trainer.train("fix everything by yourself without any information", "fix_missing_args", res)
    print(f"\n[8] validation guardrail (missing prompt/path) -> {'PASS' if ok else 'FAIL'}")
    print(f"    error: {res.error}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FEATURE TEST SUMMARY")
    print("=" * 72)
    all_ok = True
    for num, name, ok in results:
        all_ok = all_ok and ok
        print(f"  [{num}] {name:<45} {'PASS' if ok else 'FAIL'}")
    print(f"\nTraining log: {TRAINING_LOG}")
    print(f"Conversation: {CONVERSATION_ID}")
    print(f"Patterns in LearningStore cache: {len(trainer.store._cache)}")
    for gp in list(trainer.store._cache)[:12]:
        pat = trainer.store._cache[gp]
        print(f"    - {gp}  (conf={pat.avg_confidence}, total={pat.total_count})")

    # Show the domain classifier + search working on the learned patterns
    print("\nLearningStore.search() demo (multi-factor ranking):")
    for goal, limit in [("fix broken code in database", 3), ("audit and upgrade the project", 3)]:
        hits = trainer.store.search(goal, limit=limit)
        print(f"    goal={goal!r} domain={ train_classify(goal).value} -> {[h.goal_pattern for h in hits]}")

    print("\n" + ("ALL ASTRA-X FEATURES PASSED AND LEARNED" if all_ok else "SOME FEATURES FAILED"))
    return 0 if all_ok else 1


def train_classify(goal):
    from app.agents.learning_store import classify_goal
    return classify_goal(goal)


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))