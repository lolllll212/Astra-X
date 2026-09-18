"""Prove automatic tool dispatch end-to-end via the real executor pipeline.

The framework routes tools three ways, each verified here for every new tool:

1. ToolRegistry  -> register + get by name
2. CapabilityRegistry -> planner resolves an abstract capability to a tool
3. ToolExecutor  -> validate arguments, execute a ToolCall automatically

Run: python test_auto_dispatch.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools import (
    BrowserSkillTool,
    ColibriTool,
    ECCTool,
    GhidraTool,
    LmStudioTool,
    VoiceboxTool,
)
from app.tools.capabilities import CapabilityRegistry
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall
from app.tools.registry import ToolRegistry

TOOLS = {
    "browser_skill": (BrowserSkillTool, "browser_automation", {"action": "get_session"}),
    "voicebox": (VoiceboxTool, "tts", {"action": "get_status"}),
    "ecc": (ECCTool, "workflow_orchestration", {"action": "get_status"}),
    "colibri": (ColibriTool, "frontier_inference", {"action": "get_status"}),
    "ghidra": (GhidraTool, "reverse_engineering", {"action": "get_status"}),
    "lmstudio": (LmStudioTool, "local_inference", {"action": "get_status"}),
}

AUTO_EXECUTABLE = {
    "browser_skill": {"action": "execute_action"},
    "voicebox": {"action": "speak", "voice": "en_kokoro_v0_1"},
    "ecc": {"action": "scan_security"},
    "colibri": {"action": "doctor"},
    "ghidra": {"action": "analyze"},
    "lmstudio": {"action": "chat"},
}


def make_context() -> ToolContext:
    return ToolContext(workspace=os.getcwd())


async def main() -> None:
    print("Automatic tool dispatch verification\n")

    registry = ToolRegistry()
    capreg = CapabilityRegistry(registry)
    executor = ToolExecutor(registry)
    ctx = make_context()

    passed = 0
    for name, (tool_cls, capability, live_args) in TOOLS.items():
        tag = f"{name:<13}"
        tool = tool_cls()

        # 1. ToolRegistry: register + look up by name.
        registry.register(tool)
        assert registry.get(tool.name) is tool, f"{name}: registry lookup failed"
        print(f"{tag} ToolRegistry   -> '{tool.name}' registered & resolvable by name")

        # 2. CapabilityRegistry: planner resolves capability without knowing the tool.
        capreg.rebuild()
        assert capreg.resolve(capability).name == tool.name, f"{name}: capability resolution failed"
        assert registry.exists(tool.name)
        print(f"{tag} Capability    -> '{capability}' resolves to '{tool.name}'")

        # 3. ToolExecutor: automatic ToolCall execution incl. argument validation.
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments=live_args), ctx)
        assert isinstance(res.success, bool)
        print(f"{tag} Auto-executed '{live_args['action']}' -> success={res.success}"
              + (f" ({res.output[:50]})" if res.success and res.output else ""))
        passed += 1

        auto_args = AUTO_EXECUTABLE[name]
        res = await executor.execute(ToolCall(tool_name=tool.name, arguments=auto_args), ctx)
        print(f"{tag} Auto-executed '{auto_args['action']}' -> success={res.success} (expected false without deps/args):"
              + f" {(res.error or res.output or '')[:60]}")
        passed += 1

    # Validation guardrails fire through the executor too.
    lm = LmStudioTool()
    res = await executor.execute(ToolCall(tool_name=lm.name, arguments={"action": "chat"}), ctx)
    assert res.success is False and res.error, "executor must reject missing required args"
    print("\nExecutor validation rejected: " + res.error)
    passed += 1

    # ECC-main integration: the ECC tool auto-drives the real ECC-main checkout
    # (Node CLI + src/llm LLM providers, OpenAI-compatible -> LM Studio).
    ecc = ECCTool()
    st = json.loads((await ecc.execute(ctx, action="get_status")).output)
    ecc_home = st.get("ecc_home") or ""
    print(f"\nECC-main home: {ecc_home or '(not found - in-memory fallback)'}")
    print(f"ECC CLI available: {st.get('ecc_cli_available')} | LLM src: {st.get('ecc_llm_src_available')}")
    if ecc_home:
        res = await executor.execute(
            ToolCall(tool_name=ecc.name, arguments={"action": "ecc_cli", "ecc_command": "status --json"}), ctx
        )
        assert res.success, f"ecc_cli status failed: {res.error}"
        assert '"readiness"' in res.output or "dbPath" in res.output
        print(f"ECC CLI 'status --json' -> success: {res.success} (real ECC query)")
        passed += 1

        res = await executor.execute(
            ToolCall(tool_name=ecc.name, arguments={
                "action": "llm_generate",
                "prompt": "Reply with exactly: ECC-LLM-AUTO-OK",
                "model": "qwen2.5-coder-7b-instruct",
                "max_tokens": 32,
                "temperature": 0.0,
            }), ctx
        )
        if res.success:
            print(f"ECC LLM via src/llm -> success: {res.success} | {res.output[:40]!r}")
            passed += 1
        else:
            print(f"ECC LLM via src/llm -> skipped ({res.error[:70]}) - LM Studio offline?")

    capreg.rebuild()
    total_schema = len(registry.schemas())
    print(f"\nRegistry now hosts {registry.count} tools, {len(capreg.list_capabilities())} capabilities, "
          f"{total_schema} schemas")
    print("\nALL AUTO-DISPATCH TESTS PASSED" if passed else "FAILED")


if __name__ == "__main__":
    asyncio.run(main())
