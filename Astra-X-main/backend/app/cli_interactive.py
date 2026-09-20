"""Interactive REPL for the ``astra`` CLI.

Provides an opencode-style interactive session: a colored prompt, live
streaming responses, tool-call progress display, and slash commands
(``/help``, ``/tools``, ``/models``, ``/doctor``, ``/clear``,
``/exit``/``/quit``).

The REPL keeps an in-memory conversation so context persists across
turns, matching the behavior of an interactive chat session.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

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
    ToolProgressEvent,
    ToolResultStreamEvent,
    ThinkingEvent,
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


def _supports_color() -> bool:
    """Best-effort check whether the current console supports ANSI colors."""
    if os_environ := getattr(sys, "stdout", None) is None:
        return False
    platform = sys.platform
    if platform == "win32":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if handle and handle != -1 and ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return bool(mode.value & 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            return False
    return True


_COLOR = _supports_color()


def _paint(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _COLOR else text


def _banner() -> str:
    lines = [
        _paint("  __  __     ____  ____  ____  _____  ____", _CYAN),
        _paint(" (  \\/  )___ / ___||  _ \\|  _ \\|_   _|/ ___|", _CYAN),
        _paint("  )    (  __ \\___ \\| |_) | |_) | | |  \\___ \\", _CYAN),
        _paint(" /  /\\  \\_/  \\ ___) |  _ <|  _ <  | |   ___) |", _CYAN),
        _paint("/__/\\_\\___/  |____/|_| \\_\\_| \\_\\ |_|  |____/", _CYAN),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stream rendering
# ---------------------------------------------------------------------------


class _StreamRenderer:
    """Renders :class:`StreamEvent` objects to the console with tool call suppression."""

    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose
        self._usage: StreamUsageEvent | None = None
        self._text_buffer: str = ""
        self._tool_call_detected: bool = False

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
            return ('"name"' in text and '"arguments"' in text) or ('"tool"' in text and '"arguments"' in text)

    def _flush_buffer(self) -> None:
        """Print buffered text only if no tool call was detected."""
        if self._text_buffer and not self._tool_call_detected:
            # Also check if buffer looks like a tool call JSON (in case detection failed)
            if not self._looks_like_tool_call(self._text_buffer):
                print(self._text_buffer, end="", flush=True)
        self._text_buffer = ""
        self._tool_call_detected = False

    def feed(self, event: object) -> None:
        from app.domain.stream import StreamEvent
        assert isinstance(event, StreamEvent)

        match event:
            case StreamStartEvent():
                self._text_buffer = ""
                self._tool_call_detected = False
                pass
            case StreamMetadataEvent():
                if self._verbose:
                    print(
                        _paint(
                            f"  → {event.provider} · {event.model}",
                            _DIM,
                        )
                    )
            case ThinkingEvent():
                if self._verbose:
                    print(_paint("  💭 " + event.thought, _DIM))
            case TextDeltaEvent():
                # Buffer ALL text - don't print yet
                self._text_buffer += event.delta
            case ToolProgressEvent():
                # Tool call confirmed - discard buffer and show tool progress
                self._text_buffer = ""
                self._tool_call_detected = False
                self._tool_progress(event.tool_name, event.status, event.message)
            case ToolResultStreamEvent():
                self._text_buffer = ""
                self._tool_call_detected = False
                self._tool_result(
                    event.tool_name,
                    event.output,
                    event.is_error,
                    event.duration_ms,
                )
            case PlanEvent():
                self._text_buffer = ""
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
                self._text_buffer = ""
                marker = "✓" if event.status == "completed" else ("✗" if event.status == "failed" else "…")
                color = (
                    _GREEN
                    if event.status == "completed"
                    else _RED
                    if event.status == "failed"
                    else _YELLOW
                )
                print(_paint(f"  {marker} {event.description} [{event.status}]", color))
            case ReflectionEvent():
                self._text_buffer = ""
                color = _GREEN if event.decision == "accept" else _YELLOW
                print(
                    _paint(
                        f"  ↻ reflection: {event.decision} "
                        f"(confidence {event.confidence:.0%}) — {event.reason}",
                        color,
                    )
                )
            case ArtifactEvent():
                self._text_buffer = ""
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
                self._text_buffer = ""
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
        text = f"  [TOOL] {tool_name} -- {status}"
        if message:
            text += f" ({message})"
        print(_paint(text, color))

    @staticmethod
    def _tool_result(tool_name: str, output: str, is_error: bool, duration_ms: int | None) -> None:
        label = f"  [TOOL] {tool_name} "
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
  /help      Show this help.
  /tools     List registered tools and their status.
  /models    List models available in LM Studio.
  /doctor    Run the tool health checks.
  /clear     Reset the in-memory conversation.
  /exit      Leave the session (alias: /quit).
  /verbose   Toggle verbose event rendering.
"""


async def _cmd_tools(runtime: AstraRuntime) -> None:
    print()
    tools = list(runtime.registry.list_tools())
    print(
        _paint(
            f"  {len(tools)} tool(s) registered:\n",
            _BOLD,
        )
    )
    header = f"    {'Name':<14} {'Capabilities'}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for tool in tools:
        caps = ", ".join(tool.capabilities) if tool.capabilities else "-"
        print(f"    {tool.name:<14} {caps}")
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
) -> str:
    """Run a single chat turn through the ChatCoordinator.

    Returns the assembled assistant message text (also streamed live).
    """
    from app.domain.enums import MessageRole
    from app.domain.message import Message
    from app.services.coordinator import ChatCoordinator

    user_msg = user_message(conversation_id, text)
    prompt_messages: list[Message] = [
        m for m in history if isinstance(m, Message)
    ] + [user_msg]

    coordinator = ChatCoordinator(
        llm_router=runtime.router,
        tool_registry=runtime.registry,
        tool_executor=runtime.executor,
        native_tools=False,
        tool_schema_role=MessageRole.USER,
    )

    # Keep the ephemeral conversation out of the coordinator: it only
    # needs id/model/provider, which are already in metadata.
    from .cli_runtime import new_conversation

    conversation = new_conversation(
        model=runtime.default_model,
        provider=runtime.provider_id,
        title="CLI session",
    )
    conversation.id = conversation_id

    renderer = _StreamRenderer(verbose=verbose)
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
    model = runtime.default_model

    while True:
        try:
            raw = input(_paint("  astra> ", _GREEN if _COLOR else ""))
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = raw.strip()
        if not text:
            continue

        if text.startswith("/"):
            cmd = text.split()[0].lower()
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
            elif cmd == "/model":
                parts = text.split()
                if len(parts) >= 2:
                    model = parts[1]
                    print(_paint(f"  Model set to: {model}", _DIM))
                else:
                    print(_paint(f"  Current model: {model}", _DIM))
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
            )
            history.append(user_message(conversation_id, text))
            # The assistant's message isn't persisted in history for the
            # next turn because the coordinator re-generates context from
            # the full prompt each time; keep history lean.
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
    """Entry point for the interactive REPL (runs the asyncio loop)."""
    return asyncio.run(_run(arguments))