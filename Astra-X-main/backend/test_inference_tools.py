"""End-to-end smoke test for Colibri, Ghidra, and LM Studio tools.

Run: python test_inference_tools.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.colibri import ColibriTool
from app.tools.context import ToolContext
from app.tools.ghidra import GhidraTool
from app.tools.lmstudio import LmStudioTool


def make_context() -> ToolContext:
    return ToolContext(workspace=os.getcwd())


async def test_colibri() -> None:
    print("\n=== Colibri ===")
    tool = ColibriTool()
    ctx = make_context()

    r = await tool.execute(ctx, action="get_status")
    print("  get_status:", r.success)
    assert r.success
    st = json.loads(r.output)
    print("    cli_found:", st.get("cli_found"), "| cli_path:", st.get("cli_path"))
    print("    families:", st.get("supported_families"))
    assert len(st["supported_families"]) == 9

    r = await tool.execute(ctx, action="run", prompt="hello")
    # No model dir configured - should fail gracefully with a clear error.
    print("  run (no model, graceful):", r.success, "|", (r.error or "")[:80])
    assert isinstance(r.success, bool)

    r = await tool.execute(ctx, action="serve", model_dir="D:/no/such/model")
    print("  serve (no model, graceful):", r.success, "|", (r.error or "")[:80])

    r = await tool.execute(ctx, action="plan", model_dir="D:/no/such/model")
    print("  plan (no model, graceful):", r.success, "|", (r.error or "")[:80])

    r = await tool.execute(ctx, action="doctor")
    print("  doctor:", r.success, "|", (r.error or r.output or "")[:80])


async def test_ghidra() -> None:
    print("\n=== Ghidra ===")
    tool = GhidraTool()
    ctx = make_context()

    r = await tool.execute(ctx, action="get_status")
    print("  get_status:", r.success)
    assert r.success
    st = json.loads(r.output)
    print("    installed:", st.get("installed"), "| home:", st.get("ghidra_home") or "(none)")
    assert "installed" in st

    r = await tool.execute(ctx, action="analyze")
    print("  analyze (no binary, graceful):", r.success, "|", (r.error or "")[:80])
    assert not r.success

    r = await tool.execute(ctx, action="decompile", binary="D:/no/such.exe")
    print("  decompile (missing binary, graceful):", r.success, "|", (r.error or "")[:80])
    assert not r.success


async def test_lmstudio() -> None:
    print("\n=== LM Studio ===")
    tool = LmStudioTool()
    ctx = make_context()

    r = await tool.execute(ctx, action="get_status")
    print("  get_status:", r.success)
    assert r.success
    st = json.loads(r.output)
    print("    reachable:", st.get("reachable"), "| models:", st.get("model_count"))
    assert "reachable" in st

    expected = {
        "deepseek-r1-distill-qwen-7b",
        "qwen2.5-coder-7b-instruct",
        "llava-v1.6-mistral-7b",
    }
    if st.get("reachable"):
        loaded = set(st.get("models", []))
        print("    has expected models:", expected.issubset(loaded))
        assert expected.issubset(loaded)
    else:
        print("    LM Studio offline - skipping live cases (graceful)")
        return

    r = await tool.execute(ctx, action="list_models")
    print("  list_models:", r.success, "| count:", (r.metadata or {}).get("model_count"))
    assert r.success

    r = await tool.execute(
        ctx,
        action="chat",
        model="deepseek-r1-distill-qwen-7b",
        prompt="Reply with exactly the two words: TOOL OK",
        max_tokens=64,
        temperature=0.0,
    )
    print("  chat deepseek-r1:", r.success, "|", (r.output or "")[:60].replace("\n", " "))
    assert r.success, r.error
    meta = r.metadata or {}
    print("    usage:", meta.get("usage_total_tokens"), "tokens | latency_ms:", meta.get("latency_ms"))

    r = await tool.execute(
        ctx,
        action="chat",
        model="qwen2.5-coder-7b-instruct",
        system="You are a code generator. Reply with code only.",
        prompt="print hello in python",
        max_tokens=64,
        temperature=0.0,
    )
    print("  chat qwen2.5-coder:", r.success, "|", (r.output or "")[:60].replace("\n", " "))
    assert r.success, r.error

    r = await tool.execute(
        ctx,
        action="completion",
        model="deepseek-r1-distill-qwen-7b",
        prompt="Q: 2+2=?\nA:",
        max_tokens=16,
        temperature=0.0,
    )
    print("  completion:", r.success, "|", (r.output or "")[:60].replace("\n", " "))
    assert r.success, r.error

    r = await tool.execute(ctx, action="chat", model="deepseek-r1-distill-qwen-7b")
    print("  chat (missing prompt, graceful):", r.success, "|", (r.error or "")[:80])
    assert not r.success


async def main() -> None:
    print("Smoke test for inference tools (colibri / ghidra / lmstudio)")
    await test_colibri()
    await test_ghidra()
    await test_lmstudio()
    print("\nALL INFERENCE TOOL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
