"""Astra X TUI - Main application (opencode-style)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.widgets.tree import TreeNode
from textual.message import Message
from textual.reactive import reactive

from app.cli_runtime import build_runtime
from app.cli_runtime import new_conversation
from app.cli_runtime import user_message
from app.domain.stream import (
    StreamEvent,
    StreamStartEvent,
    StreamMetadataEvent,
    TextDeltaEvent,
    ToolProgressEvent,
    ToolResultStreamEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamUsageEvent,
    PlanEvent,
    TaskProgressEvent,
    ReflectionEvent,
    ArtifactEvent,
    ThinkingEvent,
)
from app.services.coordinator import ChatCoordinator
from app.domain.enums import MessageRole


# ──────────────────────────────────────────────────────────────────────────────
# ANSI styling
# ──────────────────────────────────────────────────────────────────────────────

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
_GRAY = "\x1b[90m"


def _supports_color() -> bool:
    import sys
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
        _paint(" ██║  ██║╚██████╔╝███████║███████║██║  ██║", _CYAN),
        _paint(" ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝", _CYAN),
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Readline history & completion
# ──────────────────────────────────────────────────────────────────────────────

_HISTORY_FILE = Path.home() / ".astra_history"
_MAX_HISTORY = 1000

_SLASH_COMMANDS = [
    "/help", "/tools", "/models", "/doctor", "/clear",
    "/exit", "/quit", "/verbose", "/agent", "/session",
    "/config", "/theme", "/history", "/model", "/compact",
]

_THEMES = ["default", "dark", "light", "monochrome", "high-contrast"]


def _setup_readline() -> None:
    try:
        import readline
    except ImportError:
        return

    if _HISTORY_FILE.exists():
        readline.read_history_file(str(_HISTORY_FILE))
    readline.set_history_length(_MAX_HISTORY)
    import atexit
    atexit.register(readline.write_history_file, str(_HISTORY_FILE))

    readline.parse_and_bind("tab: complete")
    readline.set_completer(_completer)


def _completer(text: str, state: int) -> str | None:
    if not text.startswith("/"):
        return None

    matches = [cmd for cmd in _SLASH_COMMANDS if cmd.startswith(text)]
    if state < len(matches):
        return matches[state]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Tool call widget (inline tool row like opencode)
# ──────────────────────────────────────────────────────────────────────────────

class ToolCallWidget(Static):
    """Widget displaying a tool call with progress (inline tool row)."""

    def __init__(self, tool_name: str, tool_id: str, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.status = "running"
        self.output = ""
        self.arguments = ""

    def compose(self) -> ComposeResult:
        yield Label(f"▸ {self.tool_name}", id="tool-name", classes="tool-name")
        yield Label("Running...", id="tool-status", classes="tool-status")
        yield Static("", id="tool-args", classes="tool-args")

    def update_status(self, status: str, message: str = "") -> None:
        self.status = status
        status_labels = {"running": "Running...", "completed": "Done", "failed": "Failed"}
        label = status_labels.get(status, status)
        if message:
            label += f" ({message})"
        self.query_one("#tool-status", Label).update(label)

    def update_arguments(self, args: str) -> None:
        self.arguments = args
        if args:
            self.query_one("#tool-args", Static).update(f"Args: {args[:200]}")

    def update_output(self, output: str, is_error: bool) -> None:
        self.output = output
        prefix = "Error: " if is_error else "Output: "
        self.query_one("#tool-status", Label).update(f"{prefix}{output[:200]}")


# ──────────────────────────────────────────────────────────────────────────────
# Message widgets
# ──────────────────────────────────────────────────────────────────────────────

class ChatMessage(Static):
    """A chat message widget."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        if self.role == "user":
            yield Label("▸ You", classes="role-label user")
        elif self.role == "assistant":
            yield Label("▸ Astra", classes="role-label assistant")
        elif self.role == "tool":
            yield Label("▸ Tool", classes="role-label tool")
        elif self.role == "system":
            yield Label("▸ System", classes="role-label system")
        else:
            yield Label(f"▸ {self.role}", classes="role-label")

        yield Markdown(self.content, classes="message-content")


class ToolCallDisplay(Static):
    """Display for tool calls in chat (opencode-style inline)."""

    def __init__(self, tool_name: str, tool_id: str, arguments: str, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.arguments = arguments

    def compose(self) -> ComposeResult:
        yield Label(f"▸ {self.tool_name}", classes="tool-call-name")
        if self.arguments:
            yield Static(f"```json\n{self.arguments}\n```", classes="tool-call-args")


# ──────────────────────────────────────────────────────────────────────────────
# File Tree with git status
# ──────────────────────────────────────────────────────────────────────────────

class FileTree(Tree):
    """File tree with git status indicators."""

    def __init__(self, root_path: Path, **kwargs):
        super().__init__("Files", **kwargs)
        self.root_path = root_path.resolve()
        self.show_root = True
        self.guide_depth = 3

    def on_mount(self) -> None:
        self.load_directory(self.root_path, self.root)

    def load_directory(self, path: Path, node: TreeNode) -> None:
        try:
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for item in items:
                if item.name.startswith(".") and item.name not in (".gitignore", ".env", ".git"):
                    continue
                
                label = self._format_item(item)
                child = node.add(label, data={"path": item})
                if item.is_dir():
                    child.allow_expand = True
        except PermissionError:
            pass

    def _format_item(self, item: Path) -> str:
        icon = "📁" if item.is_dir() else "📄"
        status = self._get_git_status(item)
        return f"{icon} {item.name}{status}"

    def _get_git_status(self, item: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--", str(item)],
                capture_output=True, text=True, cwd=self.root_path, timeout=2
            )
            if result.stdout.strip():
                status = result.stdout.strip()[0]
                if status == "M":
                    return " ●"
                elif status == "?":
                    return " ○"
                elif status == "A":
                    return " +"
                elif status == "D":
                    return " -"
        except Exception:
            pass
        return ""

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        if node.children:
            return
        data = node.data
        if data and isinstance(data, dict) and data.get("path"):
            path = data["path"]
            if path.is_dir():
                self.load_directory(path, node)


# ──────────────────────────────────────────────────────────────────────────────
# Tool Sidebar
# ──────────────────────────────────────────────────────────────────────────────

class ToolList(ListView):
    """List of available tools with status indicators."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tools = []

    def on_mount(self) -> None:
        self.refresh_tools()

    def refresh_tools(self) -> None:
        self.clear()
        for tool in self.tools:
            caps = ", ".join(tool.capabilities) if tool.capabilities else "-"
            self.append(ListItem(Label(f"{tool.name:<18} {caps}")))


class SessionList(ListView):
    """List of sessions."""

    def on_mount(self) -> None:
        self.append(ListItem(Label("Current session (new)")))


# ──────────────────────────────────────────────────────────────────────────────
# Stream Renderer (opencode-style)
# ──────────────────────────────────────────────────────────────────────────────

class StreamRenderer:
    """Renders StreamEvent objects to the console with opencode-style formatting."""

    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose
        self._usage = None
        self._text_buffer = ""
        self._tool_call_detected = False
        self._current_tool_widget = None

    def _looks_like_tool_call(self, text: str) -> bool:
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
        if self._text_buffer and not self._tool_call_detected:
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
                    print(f"  → {event.provider} · {event.model}", style="dim")
            case ThinkingEvent():
                if self._verbose:
                    print("  💭 " + event.thought, style="dim")
            case TextDeltaEvent():
                self._text_buffer += event.delta
                if self._looks_like_tool_call(self._text_buffer):
                    self._tool_call_detected = True
            case ToolProgressEvent():
                if not self._tool_call_detected:
                    self._flush_buffer()
                print(f"  [TOOL] {event.tool_name} -- {event.status}", end="")
                if event.message:
                    print(f" ({event.message})", end="")
                print()
            case ToolResultStreamEvent():
                print(f"  [TOOL] {event.tool_name} -- Output: {event.output[:100]}")
            case PlanEvent():
                self._flush_buffer()
                print()
                print(f"  ⚡ Plan ({event.iteration + 1}): {event.goal}", style="yellow")
                for task in event.tasks:
                    print(
                        f"    • [{task.status}] {task.description}"
                        + (f"  ({task.tool_name})" if task.tool_name else ""),
                        style="dim" if task.status != "pending" else "blue",
                    )
            case TaskProgressEvent():
                marker = "✓" if event.status == "completed" else ("✗" if event.status == "failed" else "…")
                color = "green" if event.status == "completed" else "red" if event.status == "failed" else "yellow"
                print(f"  {marker} {event.description} [{event.status}]", style=color)
            case ReflectionEvent():
                color = "green" if event.decision == "accept" else "yellow"
                print(
                    f"  ↻ reflection: {event.decision} "
                    f"(confidence {event.confidence:.0%}) — {event.reason}",
                    style=color,
                )
            case ArtifactEvent():
                if self._verbose:
                    print(
                        f"  📎 {event.label or event.artifact_type} "
                        f"from {event.task_id}",
                        style="dim",
                    )
            case StreamUsageEvent():
                self._usage = event
            case StreamErrorEvent():
                self._flush_buffer()
                print()
                print(f"  ✗ {event.message}", style="red")
            case StreamDoneEvent():
                self._flush_buffer()
                if self._usage is not None:
                    print()
                    print(
                        f"  ({self._usage.prompt_tokens} in "
                        f"/ {self._usage.completion_tokens} out · "
                        f"{self._usage.total_tokens} total)",
                        style="dim",
                    )
            case _:
                if self._verbose:
                    print(f"  [event: {type(event).__name__}]", style="dim")


# ──────────────────────────────────────────────────────────────────────────────
# Slash commands
# ──────────────────────────────────────────────────────────────────────────────

_HELP_TEXT = """
Astra X — interactive session

How to use
  Type a message to chat with Astra. The assistant can use any of the
  registered tools (web, filesystem, LM Studio, colibri, ecc, n8n,
  ghidra, voicebox, ...) as needed.

Slash commands
  /help        Show this help.
  /tools       List registered tools and their status.
  /models      List models available in LM Studio.
  /doctor      Run the tool health checks.
  /clear       Reset the in-memory conversation.
  /compact     Compact the conversation history.
  /exit        Leave the session (alias: /quit).
  /verbose     Toggle verbose event rendering.
  /agent       Toggle agent mode (plan/execute/reflect).
  /session     Manage sessions (list, save, load).
  /config      View config (get <key>, set <key> <val>, list).
  /theme       Set theme (default, dark, light, monochrome, high-contrast).
  /history     Show command history.
  /model       Show or set current model.
"""


async def _cmd_tools(runtime) -> None:
    print()
    tools = list(runtime.registry.list_tools())
    print(f"  {len(tools)} tool(s) registered:\n")
    header = f"    {'Name':<18} {'Capabilities'}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for tool in tools:
        caps = ", ".join(tool.capabilities) if tool.capabilities else "-"
        print(f"    {tool.name:<18} {caps}")
    print()


async def _cmd_models(runtime) -> None:
    print()
    models = await runtime.list_lmstudio_models()
    if not models:
        print("  No models reachable. Is LM Studio running (port 1234)?")
        return
    print(f"  {len(models)} model(s) available:\n")
    for model in sorted(models):
        print(f"    • {model}")
    print()


async def _cmd_doctor(runtime) -> None:
    from app.cli_doctor import run_doctor
    await run_doctor(runtime)


# ──────────────────────────────────────────────────────────────────────────────
# Chat turn
# ──────────────────────────────────────────────────────────────────────────────

async def _run_turn(
    runtime,
    conversation_id: str,
    history: list,
    text: str,
    *,
    verbose: bool,
    use_agent: bool,
) -> str:
    from app.domain.enums import MessageRole
    from app.domain.message import Message
    from app.services.coordinator import ChatCoordinator
    from app.agents.planner import Planner
    from app.agents.executor import Executor as AgentExecutor
    from app.agents.reflection import Reflection
    from app.agents.memory_manager import MemoryManager
    from app.agents.learning_manager import LearningManager
    from app.agents.base import AgentConfig

    user_msg = user_message(conversation_id, text)
    prompt_messages = [m for m in history if isinstance(m, Message)] + [user_msg]

    if use_agent:
        from app.config.settings import get_settings
        planner = Planner(llm_router=runtime.router, tool_registry=runtime.registry, config=AgentConfig())
        agent_executor = AgentExecutor(
            llm_router=runtime.router, tool_registry=runtime.registry, tool_executor=runtime.executor, config=AgentConfig()
        )
        reflection = Reflection(llm_router=runtime.router, tool_registry=runtime.registry, config=AgentConfig())
        memory_manager = MemoryManager(llm_router=runtime.router, tool_registry=runtime.registry)
        learning_manager = LearningManager(llm_router=runtime.router, tool_registry=runtime.registry)
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

    renderer = StreamRenderer(verbose=verbose)
    collected = []

    async for event in coordinator.run(
        conversation=conversation,
        messages=prompt_messages,
        user_message=user_msg,
        model=runtime.default_model,
        provider=runtime.provider_id,
    ):
        renderer.feed(event)
        if hasattr(event, 'delta'):
            collected.append(event.delta)
        if hasattr(event, 'finish_reason'):
            break

    print()
    return "".join(collected)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

async def _loop(runtime) -> int:
    _setup_readline()

    print(_banner())
    print()
    print("  Astra X — local AI assistant. Type /help for commands.")
    print()

    conversation = new_conversation(
        model=runtime.default_model,
        provider=runtime.provider_id,
        title="CLI session",
    )
    conversation_id = conversation.id
    history = []

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
            elif cmd == "/compact":
                history = history[-10:]  # Keep last 10 messages
                print(_paint("  Conversation compacted.", _DIM))
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
                    from app.cli_session import cmd_session_list
                    await cmd_session_list()
                elif args[0] == "save":
                    if len(args) >= 2:
                        from app.cli_session import cmd_session_export
                        await cmd_session_export(args[1])
                    else:
                        print("  Usage: /session save <session_id>")
                else:
                    print(f"  Unknown session command: {args[0]}")
                continue
            elif cmd == "/config":
                if not args:
                    from app.cli_config import cmd_config_list
                    cmd_config_list()
                elif args[0] == "get" and len(args) >= 2:
                    from app.cli_config import cmd_config_get
                    cmd_config_get(args[1])
                elif args[0] == "set" and len(args) >= 3:
                    from app.cli_config import cmd_config_set
                    cmd_config_set(args[1], args[2])
                elif args[0] == "list":
                    from app.cli_config import cmd_config_list
                    cmd_config_list()
                else:
                    print("  Usage: /config [get <key>|set <key> <val>|list]")
                continue
            elif cmd == "/theme":
                if not args:
                    print(f"  Current theme: {theme}")
                    print(f"  Available: {', '.join(_THEMES)}")
                elif args[0] in _THEMES:
                    theme = args[0]
                    print(_paint(f"  Theme set to: {theme}", _DIM))
                else:
                    print(f"  Unknown theme: {args[0]}. Available: {', '.join(_THEMES)}")
                continue
            elif cmd == "/history":
                if not hasattr(__builtins__, 'readline') or not _HISTORY_FILE.exists():
                    print("  No history available.")
                    continue
                print("  Command history:\n")
                with open(_HISTORY_FILE, "r") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines[-50:], 1):
                    print(f"  {i:3}  {line.rstrip()}")
                print()
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
                    runtime._settings.default_llm_model = model
                    print(_paint(f"  Model set to: {model}", _DIM))
                else:
                    print(f"  Unknown model: {args[0]}")
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
    """Entry point for the interactive REPL."""
    return asyncio.run(_run(arguments))


# ──────────────────────────────────────────────────────────────────────────────
# TUI Application (opencode-style)
# ──────────────────────────────────────────────────────────────────────────────

class ToolCallDisplay(Static):
    """Display for tool calls (opencode-style inline)."""

    def __init__(self, tool_name: str, tool_id: str, arguments: str = "", **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.arguments = arguments

    def compose(self) -> ComposeResult:
        yield Label(f"▸ {self.tool_name}", classes="tool-call-name")
        if self.arguments:
            yield Static(f"```json\n{self.arguments}\n```", classes="tool-call-args")


class ToolProgressWidget(Static):
    """Tool progress indicator (opencode-style)."""

    def __init__(self, tool_name: str, tool_id: str, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.status = "running"

    def compose(self) -> ComposeResult:
        yield Label(f"▸ {self.tool_name}", classes="tool-progress-name")
        yield Label("Running...", id="tool-progress-status", classes="tool-progress-status")

    def update_status(self, status: str, message: str = "") -> None:
        self.status = status
        label = self.query_one("#tool-progress-status", Label)
        status_text = {"running": "Running...", "completed": "Done", "failed": "Failed"}.get(status, status)
        if message:
            status_text += f" ({message})"
        label.update(status_text)


class ToolResultWidget(Static):
    """Tool result display."""

    def __init__(self, tool_name: str, output: str, is_error: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.output = output
        self.is_error = is_error

    def compose(self) -> ComposeResult:
        prefix = "Error: " if self.is_error else "Output: "
        yield Label(f"▸ {self.tool_name} -- {prefix}{self.output[:200]}", classes="tool-result")


class ToolListWidget(ListView):
    """List of available tools with status (opencode-style)."""

    def __init__(self, runtime=None, **kwargs):
        super().__init__(**kwargs)
        self.runtime = runtime
        self._tools_populated = False

    def set_runtime(self, runtime):
        self.runtime = runtime
        if not self._tools_populated and runtime:
            self._populate_tools()

    def on_mount(self) -> None:
        if self.runtime and not self._tools_populated:
            self._populate_tools()

    def _populate_tools(self) -> None:
        if not self.runtime:
            return
        self.clear()
        for tool in self.runtime.registry.list_tools():
            caps = ", ".join(tool.capabilities) if tool.capabilities else "-"
            self.append(ListItem(Label(f"{tool.name:<18} {caps}")))
        self._tools_populated = True


class SessionListWidget(ListView):
    """Session list widget."""

    def on_mount(self) -> None:
        self.append(ListItem(Label("Current session (new)")))


class FileTreeWidget(Tree):
    """File tree with git status (opencode-style)."""

    def __init__(self, root_path: Path, **kwargs):
        super().__init__("Files", **kwargs)
        self.root_path = root_path.resolve()
        self.show_root = True
        self.guide_depth = 3

    def on_mount(self) -> None:
        self.load_directory(self.root_path, self.root)

    def load_directory(self, path: Path, node: TreeNode) -> None:
        try:
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for item in items:
                if item.name.startswith(".") and item.name not in (".gitignore", ".env", ".git"):
                    continue
                label = self._format_item(item)
                child = node.add(label, data={"path": item})
                if item.is_dir():
                    child.allow_expand = True
        except PermissionError:
            pass

    def _format_item(self, item: Path) -> str:
        icon = "📁" if item.is_dir() else "📄"
        status = self._get_git_status(item)
        return f"{icon} {item.name}{status}"

    def _get_git_status(self, item: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--", str(item)],
                capture_output=True, text=True, cwd=self.root_path, timeout=2
            )
            if result.stdout.strip():
                status = result.stdout.strip()[0]
                if status == "M":
                    return " ●"
                elif status == "?":
                    return " ○"
                elif status == "A":
                    return " +"
                elif status == "D":
                    return " -"
        except Exception:
            pass
        return ""

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        if node.children:
            return
        data = node.data
        if data and isinstance(data, dict) and data.get("path"):
            path = data["path"]
            if path.is_dir():
                self.load_directory(path, node)


class MessageWidget(Static):
    """Chat message widget (opencode-style)."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        role_labels = {
            "user": ("▸ You", "user"),
            "assistant": ("▸ Astra", "assistant"),
            "tool": ("▸ Tool", "tool"),
            "system": ("▸ System", "system"),
        }
        label, role_class = role_labels.get(self.role, (f"▸ {self.role}", "system"))
        yield Label(label, classes=f"role-label {role_class}")
        yield Markdown(self.content, classes="message-content")


class ToolCallWidget(Static):
    """Tool call display (opencode-style inline)."""

    def __init__(self, tool_name: str, arguments: str = "", **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.arguments = arguments

    def compose(self) -> ComposeResult:
        yield Label(f"▸ {self.tool_name}", classes="tool-call-name")
        if self.arguments:
            yield Static(f"```json\n{self.arguments}\n```", classes="tool-call-args")


class ToolProgressWidget(Static):
    """Tool progress indicator (opencode-style)."""

    def __init__(self, tool_name: str, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name

    def compose(self) -> ComposeResult:
        yield Label(f"▸ {self.tool_name}", classes="tool-progress-name")
        yield Label("Running...", id="tool-progress-status", classes="tool-progress-status")

    def update_status(self, status: str, message: str = "") -> None:
        self.status = status
        label = self.query_one("#tool-progress-status", Label)
        status_text = {"running": "Running...", "completed": "Done", "failed": "Failed"}.get(status, status)
        if message:
            status_text += f" ({message})"
        label.update(status_text)


class ToolResultWidget(Static):
    """Tool result display."""

    def __init__(self, tool_name: str, output: str, is_error: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.output = output
        self.is_error = is_error

    def compose(self) -> ComposeResult:
        prefix = "Error: " if self.is_error else "Output: "
        yield Label(f"▸ {self.tool_name} -- {prefix}{self.output[:200]}", classes="tool-result")


class SessionListWidget(ListView):
    """Session list widget."""

    def on_mount(self) -> None:
        self.append(ListItem(Label("Current session (new)")))


class FileTreeWidget(Tree):
    """File tree with git status (opencode-style)."""

    def __init__(self, root_path: Path, **kwargs):
        super().__init__("Files", **kwargs)
        self.root_path = root_path.resolve()
        self.show_root = True
        self.guide_depth = 3

    def on_mount(self) -> None:
        self.load_directory(self.root_path, self.root)

    def load_directory(self, path: Path, node: TreeNode) -> None:
        try:
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for item in items:
                if item.name.startswith(".") and item.name not in (".gitignore", ".env", ".git"):
                    continue
                label = self._format_item(item)
                child = node.add(label, data={"path": item})
                if item.is_dir():
                    child.allow_expand = True
        except PermissionError:
            pass

    def _format_item(self, item: Path) -> str:
        icon = "📁" if item.is_dir() else "📄"
        status = self._get_git_status(item)
        return f"{icon} {item.name}{status}"

    def _get_git_status(self, item: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--", str(item)],
                capture_output=True, text=True, cwd=self.root_path, timeout=2
            )
            if result.stdout.strip():
                status = result.stdout.strip()[0]
                if status == "M":
                    return " ●"
                elif status == "?":
                    return " ○"
                elif status == "A":
                    return " +"
                elif status == "D":
                    return " -"
        except Exception:
            pass
        return ""

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        if node.children:
            return
        data = node.data
        if data and isinstance(data, dict) and data.get("path"):
            path = data["path"]
            if path.is_dir():
                self.load_directory(path, node)


class AstraTUI(App):
    """Main Astra X TUI application (opencode-style)."""

    CSS = """
    Screen {
        background: $surface;
    }

    #sidebar {
        width: 42;
        border-right: solid $primary;
        height: 100%;
        background: $surface;
    }

    #main {
        width: 1fr;
        height: 100%;
    }

    #chat-area {
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }

    #input-area {
        height: auto;
        min-height: 3;
        border-top: solid $primary;
        padding: 0 1;
    }

    #input {
        width: 100%;
    }

    .role-label {
        text-style: bold;
        padding-bottom: 0;
    }

    .role-label.user {
        color: $accent;
    }

    .role-label.assistant {
        color: $success;
    }

    .role-label.tool {
        color: $warning;
    }

    .message-content {
        padding-left: 2;
    }

    .tool-call-name {
        text-style: bold;
        color: $warning;
    }

    .tool-call-args {
        color: $text-muted;
        padding-left: 2;
    }

    .tool-progress-name {
        text-style: bold;
        color: $warning;
    }

    .tool-progress-status {
        color: $warning;
    }

    .tool-result {
        color: $text-muted;
    }

    .tool-result.error {
        color: $error;
    }

    #status-bar {
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
        dock: bottom;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1;
    }

    ListView {
        border: none;
    }

    ListItem {
        padding: 0 1;
    }

    ListItem:hover {
        background: $boost;
    }

    ToolProgressWidget {
        border: solid $warning;
        margin: 1;
        padding: 1;
    }

    ToolCallWidget {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }

    ToolResultWidget {
        border: solid $success;
        margin: 1;
        padding: 1;
    }

    ToolCallWidget {
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+f", "focus_files", "Files"),
        Binding("ctrl+t", "focus_tools", "Tools"),
        Binding("ctrl+s", "focus_sessions", "Sessions"),
        Binding("ctrl+i", "focus_input", "Input"),
        Binding("ctrl+p", "command_palette", "Command Palette"),
        Binding("ctrl+up", "history_up", "History Up"),
        Binding("ctrl+down", "history_down", "History Down"),
        Binding("tab", "complete", "Complete"),
        Binding("ctrl+b", "toggle_sidebar", "Toggle Sidebar"),
    ]

    def __init__(self):
        super().__init__()
        self.runtime = None
        self.coordinator = None
        self.conversation = None
        self.streaming = False
        self.current_assistant_msg = ""
        self.history = []
        self._tool_widgets = {}
        self._tool_call_detected = False

    async def on_mount(self) -> None:
        self.runtime = build_runtime()
        self.current_model = self.runtime.default_model
        self.current_provider = self.runtime.provider_id

        self.conversation = new_conversation(
            model=self.current_model,
            provider=self.current_provider,
            title="TUI Session",
        )

        self.coordinator = ChatCoordinator(
            llm_router=self.runtime.router,
            tool_registry=self.runtime.registry,
            tool_executor=self.runtime.executor,
            native_tools=False,
            tool_schema_role=MessageRole.USER,
        )

        # Load tools into sidebar
        tool_list = self.query_one("#tool-list", ToolListWidget)
        tool_list.set_runtime(self.runtime)

        self.update_status()

    def compose(self) -> ComposeResult:
        yield Header(name="Astra X", show_clock=True)

        with Horizontal():
            # Sidebar (opencode-style: 42 cols)
            with Container(id="sidebar"):
                with TabbedContent(initial="tab-files"):
                    with TabPane("Files", id="tab-files"):
                        yield FileTreeWidget(Path.cwd(), id="file-tree")
                    with TabPane("Tools", id="tab-tools"):
                        yield ToolListWidget(None, id="tool-list")
                    with TabPane("Sessions", id="tab-sessions"):
                        yield SessionListWidget(id="session-list")

            # Main area
            with Container(id="main"):
                with VerticalScroll(id="chat-area"):
                    yield Static(id="messages")

                with Container(id="input-area"):
                    yield Input(placeholder="Type a message... (Ctrl+I to focus)", id="input")

        yield Static(id="status-bar")
        yield Footer()

    def update_status(self) -> None:
        status = self.query_one("#status-bar", Static)
        status.update(f"Model: {self.current_model} | Provider: {self.current_provider} | Tools: {len(self.runtime.registry.list_tools())}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        # Add to history
        self.history.append(("user", text))
        self.history_index = len(self.history)

        # Clear input
        event.input.value = ""

        # Add user message
        self.add_message("user", text)

        # Stream response
        self.stream_response(text)

    def add_message(self, role: str, content: str) -> None:
        messages = self.query_one("#messages", Static)
        messages.mount(MessageWidget(role, content))
        messages.scroll_end()

    def add_tool_widget(self, tool_name: str, tool_id: str) -> Static:
        messages = self.query_one("#messages", Static)
        widget = ToolProgressWidget(tool_name, tool_id)
        messages.mount(widget)
        messages.scroll_end()
        return widget

    def stream_response(self, text: str) -> None:
        self.streaming = True
        self.current_assistant_msg = ""
        self._run_stream(text)

    @work(exclusive=True, thread=True)
    def _run_stream(self, text: str) -> None:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_stream(text))
        finally:
            loop.close()

    async def _async_stream(self, text: str) -> None:
        user_msg = user_message(self.conversation.id, text)

        # Build messages with history
        messages = []
        for role, content in self.history:
            if role == "user":
                messages.append(user_message(self.conversation.id, content))
            elif role == "assistant":
                from app.domain.message import Message, TextBlock
                from app.domain.enums import MessageRole
                from datetime import UTC, datetime
                from uuid import uuid4
                messages.append(Message(
                    id=str(uuid4()),
                    conversation_id=self.conversation.id,
                    role=MessageRole.ASSISTANT,
                    content=[TextBlock(text=content)],
                    created_at=datetime.now(UTC),
                ))

        messages.append(user_message(self.conversation.id, text))

        tool_widgets = {}

        async for event in self.coordinator.run(
            conversation=self.conversation,
            messages=messages,
            user_message=user_message(self.conversation.id, text),
            model=self.current_model,
            provider=self.current_provider,
        ):
            self.call_from_thread(self._handle_event, event)

        self.call_from_thread(self._stream_done)

    def _handle_event(self, event) -> None:
        from app.domain.stream import (
            TextDeltaEvent, ToolProgressEvent, ToolResultStreamEvent,
            StreamDoneEvent, StreamErrorEvent
        )
        import json

        if isinstance(event, TextDeltaEvent):
            self.current_assistant_msg += event.delta
            
            # Check if buffer looks like a tool call JSON
            if self._looks_like_tool_call(self.current_assistant_msg):
                self._tool_call_detected = True
            
            # Only display if not a tool call
            if not self._tool_call_detected:
                messages = self.query_one("#messages", Static)
                children = list(messages.children)
                if children and isinstance(children[-1], MessageWidget) and children[-1].role == "assistant":
                    children[-1].content = self.current_assistant_msg
                    children[-1].refresh()
                else:
                    self.add_message("assistant", self.current_assistant_msg)

        elif isinstance(event, ToolProgressEvent):
            # Tool call confirmed - discard buffer
            self.current_assistant_msg = ""
            self._tool_call_detected = False
            if event.tool_name not in self._tool_widgets:
                widget = self.add_tool_widget(event.tool_name, event.tool_call_id or "unknown")
                self._tool_widgets[event.tool_name] = widget
            else:
                widget = self._tool_widgets[event.tool_name]
            widget.update_status(event.status, event.message or "")

        elif isinstance(event, ToolResultStreamEvent):
            self.current_assistant_msg = ""
            self._tool_call_detected = False
            if event.tool_name in self._tool_widgets:
                widget = self._tool_widgets[event.tool_name]
                widget.remove()
                del self._tool_widgets[event.tool_name]
            self.add_message("tool", f"**{event.tool_name}**: {event.output}")

        elif isinstance(event, StreamErrorEvent):
            self.add_message("assistant", f"Error: {event.message}")

    def _looks_like_tool_call(self, text: str) -> bool:
        """Check if text looks like a JSON tool call."""
        text = text.strip()
        if not text.startswith("{"):
            return False
        try:
            obj = json.loads(text)
            return isinstance(obj, dict) and ("name" in obj or "tool" in obj) and "arguments" in obj
        except json.JSONDecodeError:
            return ('"name"' in text and '"arguments"' in text) or ('"tool"' in text and '"arguments"' in text)

    def _stream_done(self) -> None:
        self.streaming = False
        self.history.append(("assistant", self.current_assistant_msg))

    def add_tool_widget(self, tool_name: str, tool_id: str) -> Static:
        messages = self.query_one("#messages", Static)
        widget = ToolProgressWidget(tool_name, tool_id)
        messages.mount(widget)
        messages.scroll_end()
        return widget

    def action_quit(self) -> None:
        self.exit()

    def action_clear_chat(self) -> None:
        messages = self.query_one("#messages", Static)
        messages.remove_children()
        self.add_message("system", "Chat cleared.")

    def action_focus_files(self) -> None:
        self.query_one("#tab-files", TabPane).focus()

    def action_focus_tools(self) -> None:
        self.query_one("#tab-tools", TabPane).focus()

    def action_focus_sessions(self) -> None:
        self.query_one("#tab-sessions", TabPane).focus()

    def action_focus_input(self) -> None:
        self.query_one("#input", Input).focus()

    def action_command_palette(self) -> None:
        # Simple command palette
        commands = [
            ("/help", "Show help"),
            ("/tools", "List tools"),
            ("/models", "List models"),
            ("/doctor", "Run doctor"),
            ("/clear", "Clear chat"),
            ("/agent", "Toggle agent mode"),
            ("/compact", "Compact history"),
            ("/model", "Switch model"),
        ]
        print("\n  Commands:")
        for cmd, desc in commands:
            print(f"  {cmd:<15} {desc}")

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", Container)
        sidebar.display = not sidebar.display

    def action_history_up(self) -> None:
        if not self.history or self.history_index <= 0:
            return
        self.history_index -= 1
        self.query_one("#input", Input).value = self.history[self.history_index][1]

    def action_history_down(self) -> None:
        if self.history_index >= len(self.history) - 1:
            self.history_index = len(self.history)
            self.query_one("#input", Input).value = ""
            return
        self.history_index += 1
        self.query_one("#input", Input).value = self.history[self.history_index][1]

    def action_complete(self) -> None:
        input_widget = self.query_one("#input", Input)
        text = input_widget.value
        if text.startswith("/"):
            commands = ["/help", "/tools", "/models", "/doctor", "/clear", "/exit", "/agent", "/compact", "/model"]
            for cmd in commands:
                if cmd.startswith(text):
                    input_widget.value = cmd
                    input_widget.cursor_position = len(cmd)
                    break


def run_tui() -> None:
    """Entry point for the TUI."""
    app = AstraTUI()
    app.run()