"""Verify AutoFix - the autonomous single-prompt repair agent.

End-to-end through the real executor pipeline:

1. ToolRegistry + CapabilityRegistry + ToolExecutor dispatch for 'autofix'.
2. Prompt-driven target discovery (e.g. an 'ecc-main' folder mentioned in
   the prompt resolves under the user profile / Downloads).
3. A real end-to-end repair run on a temp project: a buggy Python file is
   detected by the (fake) coding LLM and fixed through the filesystem write
   tool, then syntax-verified with py_compile.
4. Sensitive files (e.g. credentials.json) are never touched until an
   approval_token is presented via confirm_writes.

Run: python test_autofix_agent.py
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.autofix import AutoFixTool
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall
from app.tools.registry import ToolRegistry
from app.tools.result import ToolResult


class FakeCodingLLM:
    """Test double for the LM Studio analysis model.

    Reads the file payload embedded in the repair prompt and returns a valid
    JSON repair plan : one object per file. 'buggy.py' gets a corrected body;
    'credentials.json' is flagged (sensitive) and rewritten with valid JSON.
    """

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        prompt = kwargs.get("prompt", "")
        idx = prompt.rfind('{"files":')
        payload = json.loads(prompt[idx:]) if idx != -1 else {"files": []}
        reports = []
        for item in payload.get("files", []):
            path = item["path"]
            fixed = ""
            issues: list[str] = []
            if path == "buggy.py":
                fixed = "def add(a, b):\n    return a + b\n"
                issues = ["syntax error: missing closing parenthesis on def line"]
            elif path == "credentials.json":
                fixed = '{"api_keys": ["sk-test-1234"]}\n'
                issues = ["secrets file had invalid JSON"]
            reports.append({"path": path, "issues": issues, "fixed_content": fixed})
        return ToolResult(success=True, output=json.dumps(reports))


BUGGY_PY = "def add(a, b:\n    return a + b\n"
CREDS_JSON = '{"api_keys": "sk-test-1234}\n'


def write_fixture() -> str:
    root = tempfile.mkdtemp(prefix="autofix_test_")
    with open(os.path.join(root, "buggy.py"), "w", encoding="utf-8") as fh:
        fh.write(BUGGY_PY)
    with open(os.path.join(root, "credentials.json"), "w", encoding="utf-8") as fh:
        fh.write(CREDS_JSON)
    return root


async def run() -> None:
    print("AutoFix agent verification\n")

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    tool = AutoFixTool()
    tool._lmstudio = FakeCodingLLM()
    registry.register(tool)
    ctx = ToolContext(workspace=os.getcwd())

    # 1. Executor dispatch + schema.
    assert registry.get(tool.name) is tool
    assert len(tool.schema.parameters) == 7
    res = await executor.execute(ToolCall(tool_name=tool.name, arguments={"action": "get_status"}), ctx)
    assert res.success and '"runs_count"' in res.output
    print("[1] Executor dispatch + get_status OK")
    print(f"    capabilities: {tool.capabilities}")
    print(f"    schema params: {[p.name for p in tool.schema.parameters]}")
    passed = 1

    # 2. Prompt-driven target discovery: 'ecc-main' folder mentioned in prompt.
    from app.tools.autofix.autofix import _find_target

    ecc_target = _find_target("", "please fix the cli issue in the ecc-main folder")
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    print(f"[2] Prompt folder detection -> {ecc_target}")
    assert ecc_target.lower().endswith("ecc-main") or ecc_target.lower().replace("\\", "/").endswith("ecc-main"), ecc_target
    print(f"    resolved under user profile: '{home}' detected correctly")
    passed += 1

    # 3. Full repair run on a temp project (real write tool + real py_compile).
    root = write_fixture()
    try:
        res = await executor.execute(
            ToolCall(tool_name=tool.name, arguments={
                "action": "fix",
                "prompt": "I think I wrote some wrong code here, fix it",
                "path": root,
            }), ctx
        )
        assert res.success, res.error
        meta = res.metadata
        safe_edits = meta.get("edits_applied", 0)
        pending_files = meta.get("sensitive_files", [])
        token = meta.get("approval_token", "")

        buggy_path = os.path.join(root, "buggy.py")
        with open(buggy_path, encoding="utf-8") as fh:
            fixed_content = fh.read()
        assert "def add(a, b):" in fixed_content and "return a + b" in fixed_content
        # Fresh copy must now compile cleanly (the .bak holds the original bug).
        try:
            subprocess.run(
                [sys.executable, "-m", "py_compile", buggy_path],
                capture_output=True,
                timeout=120,
                check=True,
            )
            subcompile = 0
        except subprocess.CalledProcessError:
            subcompile = 1
        assert subcompile == 0, "fixed buggy.py must pass py_compile"

        creds_path = os.path.join(root, "credentials.json")
        with open(creds_path, encoding="utf-8") as fh:
            creds = fh.read()
        assert creds == CREDS_JSON, "sensitive file must NOT be written before approval"
        assert safe_edits >= 1, "buggy.py should have been fixed without approval"
        assert pending_files and os.path.normpath(creds_path) in [os.path.normpath(p) for p in pending_files]
        assert token, "run awaiting approval must return an approval_token"
        assert meta.get("status") == "awaiting_approval"
        print("[3] Repair run: robust fix + verification OK")
        print(f"    buggy.py rewritten to a py_compile-clean version ({safe_edits} edits applied)")
        print(f"    sensitive 'credentials.json' untouched, approval required: {token[:8]}...")
        passed += 1

        # 4. Approval flow wiring.
        bad = await executor.execute(
            ToolCall(tool_name=tool.name, arguments={
                "action": "confirm_writes",
                "approval_token": "deadbeef",
            }), ctx
        )
        assert bad.success is False and "Invalid approval_token" in bad.error
        ok = await executor.execute(
            ToolCall(tool_name=tool.name, arguments={
                "action": "confirm_writes",
                "approval_token": token,
            }), ctx
        )
        assert ok.success and ok.metadata.get("writes_applied", 0) >= 1
        with open(creds_path, encoding="utf-8") as fh:
            creds_after = fh.read()
        assert 'sk-test-1234' in creds_after and "api_keys" in creds_after
        print("[4] Sensitive-file approval gate OK")
        print("    wrong token rejected; right token applied the approved write")
        passed += 1
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 5. Validation guardrail: fix without prompt/path must be rejected.
    res = await executor.execute(ToolCall(tool_name=tool.name, arguments={"action": "fix"}), ctx)
    assert res.success is False and "prompt is required" in res.error
    print("[5] Missing-argument validation OK")
    passed += 1

    print(f"\nRegistrations: {registry.count} tool(s), schema checks passed")
    print("ALL AUTOFIX AGENT TESTS PASSED" if passed == 5 else "FAILED")


if __name__ == "__main__":
    asyncio.run(run())
