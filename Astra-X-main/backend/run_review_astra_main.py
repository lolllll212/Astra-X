"""Drive AutoFix 'review' against C:\\Users\\Ashut\\Astra-main using the real LM Studio LLM.

Read-only audit: the local model inspects the project and returns per-file
issues, upgrade suggestions, and database-relevance classification.
Nothing is written to disk.

Run: python run_review_astra_main.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.autofix import AutoFixTool
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall
from app.tools.registry import ToolRegistry


async def main() -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    registry.register(AutoFixTool())
    ctx = ToolContext(workspace=os.getcwd())

    res = await executor.execute(
        ToolCall(tool_name="autofix", arguments={
            "action": "review",
            "prompt": "Audit this project: find bugs, suggest upgrades, classify each file as database-related or not, and flag clear junk files unrelated to the database.",
            "path": r"C:\Users\Ashut\Astra-main",
            "model": "qwen2.5-coder-7b-instruct",
            "use_heavy_model": False,
            "max_files": 200,
        }), ctx
    )

    if not res.success or not res.output:
        print("REVIEW FAILED:", res.error or res.output)
        return

    data = json.loads(res.output)
    reviews = data.get("reviews", [])
    summary = data.get("summary", {})

    print(f"Target: {data.get('target_dir')}")
    print(f"Files analyzed: {summary.get('files_analyzed')} | With issues: {summary.get('with_issues')} | "
          f"DB-related: {summary.get('db_related')}")
    print()

    print("=== NON-DATABASE FILES (candidate cleanup) ===")
    for r in reviews:
        if not r.get("database_related"):
            action = r.get("action", "?")
            prio = r.get("priority", "?")
            issues = r.get("issues") or []
            print(f"  [{prio:^15}] {r.get('path'):<42} action={action:<7} issues={len(issues)}")

    print()
    print("=== DATABASE / KEEP FILES ===")
    for r in reviews:
        if r.get("database_related"):
            issues = r.get("issues") or []
            ups = r.get("upgrades") or []
            print(f"  {r.get('path')}  (issues={len(issues)}, upgrades={len(ups)})")
            for u in ups[:2]:
                print(f"      upgrade: {u}")

    print()
    print("=== ISSUES (all files) ===")
    for r in reviews:
        for i in (r.get("issues") or [])[:3]:
            print(f"  {r.get('path')}: {i}")

    print()
    print("=== UPGRADE SUGGESTIONS (sample) ===")
    for r in reviews:
        for u in (r.get("upgrades") or [])[:2]:
            print(f"  {r.get('path')}: {u}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astra_main_review.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nFull review saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())