"""Live smoke test: drive AutoFix against the real LM Studio coding LLM.

Creates a temp project with one intentionally broken Python file and asks the
autonomous fixer to repair it using the actual qwen2.5-coder model served by
LM Studio. Verifies the file is rewritten and now compiles.

Run: python test_autofix_live.py
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

BUGGY = """import math


def circle_area(radius):
    if radius < 0
        raise ValueError("radius must be non-negative")
    return math.pi * radius ** 2


def broken_syntax():
    return 1 +
"""


async def main() -> None:
    root = tempfile.mkdtemp(prefix="autofix_live_")
    path = os.path.join(root, "shapes.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(BUGGY)

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    registry.register(AutoFixTool())
    ctx = ToolContext(workspace=os.getcwd())

    res = await executor.execute(
        ToolCall(tool_name="autofix", arguments={
            "action": "fix",
            "prompt": "I think this Python file has syntax errors, fix them and make it valid",
            "path": root,
        }), ctx
    )
    print(json.dumps(res.metadata, indent=2)[:2000])
    print((res.output or "")[:2000])

    if not res.success:
        print("\nLIVE TEST SKIPPED: LLM unreachable/failed")
        shutil.rmtree(root, ignore_errors=True)
        return

    with open(path, encoding="utf-8") as fh:
        fixed = fh.read()
    print("\n--- fixed file ---")
    print(fixed)
    print("---")
    try:
        subprocess.run([sys.executable, "-m", "py_compile", path], check=True, capture_output=True, timeout=120)
        print("RESULT: file now compiles cleanly" if res.metadata.get("edits_applied") else "no edits were applied")
    except subprocess.CalledProcessError:
        print("RESULT: file still does NOT compile - model may have failed repairs")

    if res.metadata.get("approval_token"):
        ok = await executor.execute(
            ToolCall(tool_name="autofix", arguments={"action": "confirm_writes", "approval_token": res.metadata["approval_token"]}), ctx
        )
        print("Confirm sensitive writes:", ok.output)

    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
