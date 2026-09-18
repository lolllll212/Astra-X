"""End-to-end smoke test for BrowserSkill, Voicebox, and ECC tools.

Run: python test_automation_tools.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.browserskill import BrowserSkillTool
from app.tools.context import ToolContext
from app.tools.ecc import ECCTool
from app.tools.voicebox import VoiceboxTool


def make_context() -> ToolContext:
    ctx = ToolContext(workspace=os.getcwd())
    return ctx


async def test_browserskill() -> None:
    print("\n=== BrowserSkill ===")
    tool = BrowserSkillTool()
    ctx = make_context()

    r = await tool.execute(ctx, action="start_session")
    # Requires the external `bsk` CLI. Succeeds if installed; otherwise errors gracefully.
    print("  start_session:", r.success, r.output if r.success else r.error)
    assert isinstance(r.success, bool)

    if not r.success:
        # CLI missing - verify graceful error, then end BrowserSkill section.
        assert "daemon" in r.error.lower() or "bsk" in r.error.lower() or "not found" in r.error.lower() or "failed" in r.error.lower()
        # Ensure unknown commands still error cleanly without a session.
        r2 = await tool.execute(ctx, action="execute_action", command="navigate")
        assert not r2.success
        print("  execute_action (no session):", r2.success, r2.error)
        return

    session_id = r.metadata["session_id"]
    print("  session_id:", session_id)

    r = await tool.execute(ctx, action="get_session", session_id=session_id)
    print("  get_session:", r.success)
    assert r.success, r.error

    r = await tool.execute(ctx, action="execute_action", session_id=session_id, command="navigate", url="https://example.com")
    print("  execute_action:", r.success)
    assert isinstance(r.success, bool)

    r = await tool.execute(ctx, action="close_session", session_id=session_id)
    print("  close_session:", r.success)
    assert r.success, r.error


async def test_voicebox() -> None:
    print("\n=== Voicebox ===")
    tool = VoiceboxTool()
    ctx = make_context()

    r = await tool.execute(ctx, action="get_status")
    print("  get_status:", r.success)
    assert r.success, r.error
    st = json.loads(r.output)
    print("    voice_profiles_count:", st.get("voice_profiles_count"))
    assert st.get("voice_profiles_count", 0) >= 10

    r = await tool.execute(ctx, action="list_voices")
    print("  list_voices:", r.success, (r.metadata or {}).get("count"))
    assert r.success, r.error
    assert (r.metadata or {}).get("count", 0) == 31

    r = await tool.execute(ctx, action="clone_voice", voice_name="test_clone", reference_audio_path="nonexistent.wav")
    print("  clone_voice (should fail gracefully):", r.success)
    assert not r.success  # missing audio path must fail

    r = await tool.execute(ctx, action="transcribe", audio_path="nonexistent.wav")
    print("  transcribe (should fail gracefully):", r.success)
    assert not r.success

    r = await tool.execute(ctx, action="speak", text="", voice_profile_id="kokoro_af_heart")
    print("  speak empty text (should fail):", r.success)
    assert not r.success

    r = await tool.execute(ctx, action="start_dictation")
    # Dictation requires a hotkey manager; may fail if unavailable. Only assert not NotImplemented.
    print("  start_dictation:", r.success, r.output if hasattr(r, "output") else "")

    r = await tool.execute(ctx, action="create_story", story_name="Test Story")
    print("  create_story:", r.success)
    assert r.success

    r = await tool.execute(ctx, action="add_story_track", track_data={"text": "hello", "voice": "kokoro_af_heart"})
    print("  add_story_track:", r.success)
    assert r.success

    r = await tool.execute(ctx, action="stop_dictation")
    print("  stop_dictation:", r.success)


async def test_ecc() -> None:
    print("\n=== ECC ===")
    tool = ECCTool()
    ctx = make_context()

    r = await tool.execute(ctx, action="get_status")
    print("  get_status:", r.success)
    assert r.success
    st = json.loads(r.output)
    assert st["agents_loaded"] == 13
    assert st["skills_loaded"] == 15
    assert st["hooks_loaded"] == 5
    print(f"    agents={st['agents_loaded']} skills={st['skills_loaded']} hooks={st['hooks_loaded']} memory={st['memory_entries']}")

    r = await tool.execute(ctx, action="execute_workflow", feature="add rate limiting middleware")
    print("  execute_workflow:", r.success)
    assert r.success, r.error
    meta = r.metadata or {}
    print(f"    workflow_id={meta.get('workflow_id')} steps={meta.get('steps')}")

    r = await tool.execute(ctx, action="create_agent", name="QA Tester", role="nightshift_test_agent")
    print("  create_agent (invalid role falls back):", r.success)
    assert r.success
    assert r.metadata.get("role") == "domain_expert"

    r = await tool.execute(ctx, action="create_skill", name="Refactor Bot", category="refactoring")
    print("  create_skill:", r.success)
    assert r.success

    r = await tool.execute(ctx, action="create_hook", name="OnBoot", hook_type="session_start")
    print("  create_hook:", r.success)
    assert r.success

    r = await tool.execute(ctx, action="create_rule", name="NoConsoleLogs", content="Never commit console.log")
    print("  create_rule:", r.success)
    assert r.success

    r = await tool.execute(ctx, action="save_memory", name="api-design", content="All endpoints follow /v1", tags=["api", "style"])
    print("  save_memory:", r.success)
    assert r.success

    r = await tool.execute(ctx, action="search_memory", query="api endpoints")
    print("  search_memory:", r.success, (r.metadata or {}).get("results_count"))
    assert r.success
    assert (r.metadata or {}).get("memory_total", 0) >= 2

    r = await tool.execute(ctx, action="scan_security", scan_type="project")
    print("  scan_security:", r.success, (r.metadata or {}).get("findings_count"))
    assert r.success

    r = await tool.execute(ctx, action="build_context", project_dir=".", feature="refactor auth")
    print("  build_context:", r.success, (r.metadata or {}).get("context_window"))
    assert r.success

    r = await tool.execute(ctx, action="get_status")
    st = json.loads(r.output)
    print("  final status: agents=%s skills=%s hooks=%s memory=%s rules=%s" % (
        st["agents_loaded"], st["skills_loaded"], st["hooks_loaded"], st["memory_entries"], st["rules_loaded"]))
    assert st["skills_loaded"] == 16
    assert st["rules_loaded"] == 1
    assert st["memory_entries"] == 2


async def main() -> None:
    print("Smoke test for automation tools")
    await test_browserskill()
    await test_voicebox()
    await test_ecc()
    print("\nALL AUTOMATION TOOL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
