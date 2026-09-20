"""Enhanced Interactive REPL for the ``astra`` CLI.

Opencode-style interactive session with:
- ANSI-themed prompt (astra>)
- Readline history with up/down arrows
- Tab completion for slash commands, model names, tool names
- Agent mode toggle
- Session persistence/restore
- Rich streaming renderer with tool progress
- Slash commands: /help, /tools, /models, /doctor, /clear, /exit, /quit,
  /verbose, /agent, /session, /config, /theme, /history
"""

from __future__ import annotations

import asyncio
import atexit
import sys
from pathlib import Path
from typing import Any

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

from app.domain.stream import (
    ArtifactEvent,
    PlanEvent,
    ReflectionEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamMetadataEvent,
    StreamStartEvent,
    StreamUsageEvent,
    TaskProgressEvent,
    TextDeltaEvent,
    ThinkingEvent,
    ToolProgressEvent,
    ToolResultStreamEvent,
)

from .cli_runtime import (
    AstraRuntime,
    build_runtime,
    new_conversation,
    user_message,
)

__all__ = ["run_repl"]


# ---------------------------------------------------------------------------
# ANSI styling
# ---------------------------------------------------------------------------

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_CYAN = "\x1b[36m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_MAGENTA = "\x1b[35m"
_BLUE = "\x1b[34m"
_WHITE = "\x1b[37m"
_ORANGE = "\x1b[38;5;208m"


def _supports_color() -> bool:
    if getattr(sys, "stdout", None) is None:
        return False
    platform = sys.platform
    if platform == "win32":
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if handle and handle != -1 and ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return bool(mode.value & 0x0004)
        except Exception:
            return False
    return True


_COLOR = _supports_color()


def _paint(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _COLOR else text


def _banner() -> str:
    lines = [
        _paint("  █████╗ ██╗   ██╗███████╗███████╗██╗  ██╗", _CYAN),
        _paint(" ██╔══██╗██║   ██║██╔════╝██╔════╝██║  ██║", _CYAN),
        _paint(" ███████║██║   ██║███████╗███████╗███████║", _CYAN),
        _paint(" ██╔══██║██║   ██║╚════██║╚════██║██╔══██║", _CYAN),
        _paint(" ██║  ██║╚██████╔╝███████╗███████║██║  ██║", _CYAN),
        _paint(" ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝", _CYAN),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Readline history & completion
# ---------------------------------------------------------------------------

_HISTORY_FILE = Path.home() / ".astra_history"
_MAX_HISTORY = 1000

_SLASH_COMMANDS = [
    "/help", "/tools", "/models", "/doctor", "/clear",
    "/exit", "/quit", "/verbose", "/agent", "/session",
    "/config", "/theme", "/history", "/model",
]

_THEMES = ["default", "dark", "light", "monochrome", "high-contrast"]


def _setup_readline() -> None:
    if not HAS_READLINE:
        return

    # History
    if _HISTORY_FILE.exists():
        readline.read_history_file(str(_HISTORY_FILE))
    readline.set_history_length(_MAX_HISTORY)
    atexit.register(readline.write_history_file, str(_HISTORY_FILE))

    # Tab completion
    readline.parse_and_bind("tab: complete")
    readline.set_completer(_completer)


def _completer(text: str, state: int) -> str | None:
    if not text.startswith("/"):
        return None

    matches = [cmd for cmd in _SLASH_COMMANDS if cmd.startswith(text)]
    if state < len(matches):
        return matches[state]
    return None


# ---------------------------------------------------------------------------
# Stream renderer (upgraded from cli_interactive)
# ---------------------------------------------------------------------------


class _StreamRenderer:
    """Renders :class:`StreamEvent` objects to the console with rich formatting."""

    def __init__(self, *, verbose: bool = False, theme: str = "default") -> None:
        self._verbose = verbose
        self._theme = theme
        self._usage: StreamUsageEvent | None = None
        self._text_buffer: str = ""
        self._in_tool_call: bool = False

    def _looks_like_tool_call(self, text: str) -> bool:
        """Check if text looks like a JSON tool call."""
        text = text.strip()
        if not text.startswith("{"):
            return False
        try:
            import json
            obj = json.loads(text)
            return isinstance(obj, dict) and ("name" in obj or "tool" in obj) and "arguments" in obj
        except json.JSONDecodeError:
            # Check for partial JSON that looks like tool call
            # Look for key patterns that indicate a tool call
            return ('"name"' in text and '"arguments"' in text) or ('"tool"' in text and '"arguments"' in text)

    def _flush_buffer(self) -> None:
        if self._text_buffer and not self._in_tool_call:
            print(self._text_buffer, end="", flush=True)
        self._text_buffer = ""
        self._in_tool_call = False

    def feed(self, event: object) -> None:
        from app.domain.stream import StreamEvent
        assert isinstance(event, StreamEvent)

        match event:
            case StreamStartEvent():
                self._text_buffer = ""
                self._in_tool_call = False
                pass
            case StreamMetadataEvent():
                if self._verbose:
                    print(_paint(f"  → {event.provider} · {event.model}", _DIM))
            case ThinkingEvent():
                if self._verbose:
                    print(_paint("  💭 " + event.thought, _DIM))
            case TextDeltaEvent():
                self._text_buffer += event.delta
                # DEBUG
                # print(f"[DEBUG] Buffer: {repr(self._text_buffer)}", file=sys.stderr)
                # Check if buffer contains a tool call JSON
                if self._looks_like_tool_call(self._text_buffer):
                    self._in_tool_call = True
                    # Don't print the JSON tool call
                elif not self._in_tool_call:
                    print(event.delta, end="", flush=True)
            case ToolProgressEvent():
                # Flush any buffered text (the tool call JSON) before showing tool progress
                self._flush_buffer()
                self._tool_progress(event.tool_name, event.status, event.message)
            case ToolResultStreamEvent():
                self._flush_buffer()
                self._tool_result(
                    event.tool_name,
                    event.output,
                    event.is_error,
                    event.duration_ms,
                )
            case PlanEvent():
                print()
                print(_paint(f"  ⚡ Plan ({event.iteration + 1}): {event.goal}", _YELLOW))
                for task in event.tasks:
                    print(
                        _paint(
                            f"    • [{task.status}] {task.description}"
                            + (f"  ({task.tool_name})" if task.tool_name else ""),
                            _DIM if task.status != "pending" else _BLUE,
                        )
                    )
            case TaskProgressEvent():
                marker = "✓" if event.status == "completed" else ("✗" if event.status == "failed" else "…")
                color = (
                    _GREEN if event.status == "completed"
                    else _RED if event.status == "failed"
                    else _YELLOW
                )
                print(_paint(f"  {marker} {event.description} [{event.status}]", color))
            case ReflectionEvent():
                color = _GREEN if event.decision == "accept" else _YELLOW
                print(
                    _paint(
                        f"  ↻ reflection: {event.decision} "
                        f"(confidence {event.confidence:.0%}) — {event.reason}",
                        color,
                    )
                )
            case ArtifactEvent():
                if self._verbose:
                    print(
                        _paint(
                            f"  📎 {event.label or event.artifact_type} "
                            f"from {event.task_id}",
                            _DIM,
                        )
                    )
            case StreamUsageEvent():
                self._usage = event
            case StreamErrorEvent():
                self._flush_buffer()
                print()
                print(_paint(f"  ✗ {event.message}", _RED))
            case StreamDoneEvent():
                self._flush_buffer()
                if self._usage is not None:
                    print()
                    print(
                        _paint(
                            f"  ({self._usage.prompt_tokens} in "
                            f"/ {self._usage.completion_tokens} out · "
                            f"{self._usage.total_tokens} total)",
                            _DIM,
                        )
                    )
            case _:
                if self._verbose:
                    print(_paint(f"  [event: {type(event).__name__}]", _DIM))

    @staticmethod
    def _tool_progress(tool_name: str, status: str, message: str) -> None:
        color = _YELLOW if status == "running" else _GREEN if status == "completed" else _RED
        text = f"  ⚙ {tool_name} — {status}"
        if message:
            text += f" ({message})"
        print(_paint(text, color))

    @staticmethod
    def _tool_result(tool_name: str, output: str, is_error: bool, duration_ms: int | None) -> None:
        label = f"  ⚙ {tool_name} "
        label += f"(done in {duration_ms}ms)" if duration_ms is not None else "(done)"
        print(_paint(label, _GREEN if not is_error else _RED))


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


_HELP_TEXT = f"""
{_paint("Astra X — interactive session", _BOLD)}

{_paint("How to use", _CYAN)}
  Type a message to chat with Astra. The assistant can use any of the
  registered tools (web, filesystem, LM Studio, colibri, ecc, n8n,
  ghidra, voicebox, ...) as needed.

{_paint("Slash commands", _CYAN)}
  /help        Show this help.
  /tools       List registered tools and their status.
  /models      List models available in LM Studio.
  /doctor      Run the tool health checks.
  /clear       Reset the in-memory conversation.
  /exit        Leave the session (alias: /quit).
  /verbose     Toggle verbose event rendering.
  /agent       Toggle agent mode (plan/execute/reflect).
  /session     Manage sessions (list, save, load).
  /config      View config (get <key>, set <key> <val>, list).
  /theme       Set theme ({', '.join(_THEMES)}).
  /history     Show command history.
  /model       Show or set current model.
"""


async def _cmd_tools(runtime: AstraRuntime) -> None:
    print()
    tools = list(runtime.registry.list_tools())
    print(_paint(f"  {len(tools)} tool(s) registered:\n", _BOLD))
    header = f"    {'Name':<18} {'Capabilities'}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for tool in tools:
        caps = ", ".join(tool.capabilities) if tool.capabilities else "-"
        print(f"    {tool.name:<18} {caps}")
    print()


async def _cmd_models(runtime: AstraRuntime) -> None:
    print()
    models = await runtime.list_lmstudio_models()
    if not models:
        print(_paint("  No models reachable. Is LM Studio running (port 1234)?", _YELLOW))
        return
    print(_paint(f"  {len(models)} model(s) available:\n", _BOLD))
    for model in sorted(models):
        print(f"    • {model}")
    print()


async def _cmd_doctor(runtime: AstraRuntime) -> None:
    from .cli_doctor import run_doctor
    await run_doctor(runtime)


async def _cmd_config_get(key: str) -> None:
    from .cli_config import cmd_config_get
    cmd_config_get(key)


async def _cmd_config_set(key: str, value: str) -> None:
    from .cli_config import cmd_config_set
    cmd_config_set(key, value)


async def _cmd_config_list() -> None:
    from .cli_config import cmd_config_list
    cmd_config_list()


async def _cmd_session_list() -> None:
    from .cli_session import cmd_session_list
    await cmd_session_list()


async def _cmd_session_save(conversation_id: str) -> None:
    from .cli_session import cmd_session_export
    await cmd_session_export(conversation_id)


async def _cmd_history() -> None:
    if not HAS_READLINE or not _HISTORY_FILE.exists():
        print(_paint("  No history available.", _DIM))
        return
    print(_paint("  Command history:\n", _BOLD))
    with open(_HISTORY_FILE) as f:
        lines = f.readlines()
    for i, line in enumerate(lines[-50:], 1):
        print(f"  {i:3}  {line.rstrip()}")
    print()


# ---------------------------------------------------------------------------
# Chat turn
# ---------------------------------------------------------------------------


async def _run_turn(
    runtime: AstraRuntime,
    conversation_id: str,
    history: list[object],
    text: str,
    *,
    verbose: bool,
    use_agent: bool,
    theme: str,
) -> str:
    from app.agents.base import AgentConfig
    from app.agents.executor import Executor as AgentExecutor
    from app.agents.learning_manager import LearningManager
    from app.agents.memory_manager import MemoryManager
    from app.agents.planner import Planner
    from app.agents.reflection import Reflection
    from app.domain.enums import MessageRole
    from app.domain.message import Message
    from app.services.coordinator import ChatCoordinator

    user_msg = user_message(conversation_id, text)
    prompt_messages: list[Message] = [
        m for m in history if isinstance(m, Message)
    ] + [user_msg]

    # Build coordinator with or without agent pipeline
    if use_agent:

        planner = Planner(
            llm_router=runtime.router,
            tool_registry=runtime.registry,
            config=AgentConfig(),
        )
        agent_executor = AgentExecutor(
            llm_router=runtime.router,
            tool_registry=runtime.registry,
            tool_executor=runtime.executor,
            config=AgentConfig(),
        )
        reflection = Reflection(
            llm_router=runtime.router,
            tool_registry=runtime.registry,
            config=AgentConfig(),
        )
        memory_manager = MemoryManager(
            llm_router=runtime.router,
            tool_registry=runtime.registry,
        )
        learning_manager = LearningManager(
            llm_router=runtime.router,
            tool_registry=runtime.registry,
        )
    else:
        planner = agent_executor = reflection = memory_manager = learning_manager = None

    coordinator = ChatCoordinator(
        llm_router=runtime.router,
        tool_registry=runtime.registry,
        tool_executor=runtime.executor,
        planner=planner,
        agent_executor=agent_executor,
        reflection=reflection,
        memory_manager=memory_manager,
        learning_manager=learning_manager,
        native_tools=False,
        tool_schema_role=MessageRole.USER,
    )

    conversation = new_conversation(
        model=runtime.default_model,
        provider=runtime.provider_id,
        title="CLI session",
    )
    conversation.id = conversation_id

    renderer = _StreamRenderer(verbose=verbose, theme=theme)
    collected: list[str] = []

    async for event in coordinator.run(
        conversation=conversation,
        messages=prompt_messages,
        user_message=user_msg,
        model=runtime.default_model,
        provider=runtime.provider_id,
    ):
        renderer.feed(event)
        if isinstance(event, TextDeltaEvent):
            collected.append(event.delta)
        if isinstance(event, StreamDoneEvent):
            break

    print()
    return "".join(collected)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _loop(runtime: AstraRuntime) -> int:
    _setup_readline()

    print(_banner())
    print()
    print(_paint("  Astra X — local AI assistant. Type /help for commands.", _CYAN))
    print()

    conversation = new_conversation(
        model=runtime.default_model,
        provider=runtime.provider_id,
        title="CLI session",
    )
    conversation_id = conversation.id
    history: list[object] = []

    verbose = False
    use_agent = False
    theme = "default"
    model = runtime.default_model

    while True:
        try:
            prompt = _paint("  astra> ", _GREEN if _COLOR else "")
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = raw.strip()
        if not text:
            continue

        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("/exit", "/quit"):
                print(_paint("  Bye!", _DIM))
                break
            elif cmd == "/help":
                print(_HELP_TEXT)
                continue
            elif cmd == "/tools":
                await _cmd_tools(runtime)
                continue
            elif cmd == "/models":
                await _cmd_models(runtime)
                continue
            elif cmd == "/doctor":
                await _cmd_doctor(runtime)
                continue
            elif cmd == "/clear":
                history = []
                print(_paint("  Conversation reset.", _DIM))
                continue
            elif cmd == "/verbose":
                verbose = not verbose
                print(_paint(f"  Verbose rendering: {'on' if verbose else 'off'}", _DIM))
                continue
            elif cmd == "/agent":
                use_agent = not use_agent
                print(_paint(f"  Agent mode: {'on' if use_agent else 'off'}", _DIM))
                continue
            elif cmd == "/session":
                if not args or args[0] == "list":
                    await _cmd_session_list()
                elif args[0] == "save":
                    if len(args) >= 2:
                        await _cmd_session_save(args[1])
                    else:
                        print(_paint("  Usage: /session save <session_id>", _YELLOW))
                else:
                    print(_paint(f"  Unknown session command: {args[0]}", _YELLOW))
                continue
            elif cmd == "/config":
                if not args:
                    await _cmd_config_list()
                elif args[0] == "get" and len(args) >= 2:
                    await _cmd_config_get(args[1])
                elif args[0] == "set" and len(args) >= 3:
                    await _cmd_config_set(args[1], args[2])
                elif args[0] == "list":
                    await _cmd_config_list()
                else:
                    print(_paint("  Usage: /config [get <key>|set <key> <val>|list]", _YELLOW))
                continue
            elif cmd == "/theme":
                if not args:
                    print(_paint(f"  Current theme: {theme}", _DIM))
                    print(f"  Available: {', '.join(_THEMES)}")
                elif args[0] in _THEMES:
                    theme = args[0]
                    print(_paint(f"  Theme set to: {theme}", _DIM))
                else:
                    print(_paint(f"  Unknown theme: {args[0]}. Available: {', '.join(_THEMES)}", _YELLOW))
                continue
            elif cmd == "/history":
                await _cmd_history()
                continue
            elif cmd == "/model":
                if not args:
                    print(_paint(f"  Current model: {model}", _DIM))
                    models = await runtime.list_lmstudio_models()
                    if models:
                        print("  Available models:")
                        for m in sorted(models):
                            mark = " *" if m == model else ""
                            print(f"    {m}{mark}")
                elif args[0] in await runtime.list_lmstudio_models():
                    model = args[0]
                    runtime._settings.default_llm_model = model  # type: ignore
                    print(_paint(f"  Model set to: {model}", _DIM))
                else:
                    print(_paint(f"  Unknown model: {args[0]}", _YELLOW))
                continue
            else:
                print(_paint(f"  Unknown command: {cmd} (try /help)", _YELLOW))
                continue

        try:
            await _run_turn(
                runtime,
                conversation_id,
                history,
                text,
                verbose=verbose,
                use_agent=use_agent,
                theme=theme,
            )
            history.append(user_message(conversation_id, text))
        except KeyboardInterrupt:
            print()
            print(_paint("  Interrupted.", _DIM))
        except Exception as exc:
            print(_paint(f"  ✗ {exc}", _RED))

    return 0


async def _run(arguments: dict[str, Any] | None = None) -> int:
    runtime = build_runtime()
    try:
        return await _loop(runtime)
    finally:
        await runtime.close()


def run_repl(arguments: dict[str, Any] | None = None) -> int:
    """Entry point for the enhanced interactive REPL."""
    return asyncio.run(_run(arguments))
