"""One-shot CLI commands: chat, run, tool calls, model listing, server.

Each function here is an ``async`` coroutine that takes parsed command
arguments and runs against an :class:`AstraRuntime`.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from app.tools.models import ToolCall

if TYPE_CHECKING:
    from .cli_runtime import AstraRuntime


async def cmd_chat(
    runtime: "AstraRuntime",
    text: str,
    *,
    model: str | None = None,
    verbose: bool = False,
) -> int:
    """One-shot chat: send a message and print the assistant's reply."""
    from app.domain.enums import MessageRole
    from app.domain.stream import TextDeltaEvent, StreamDoneEvent
    from app.services.coordinator import ChatCoordinator

    from .cli_interactive import _StreamRenderer, new_conversation, user_message

    conversation = new_conversation(
        model=model or runtime.default_model,
        provider=runtime.provider_id,
        title="CLI chat",
    )
    user_msg = user_message(conversation.id, text)

    coordinator = ChatCoordinator(
        llm_router=runtime.router,
        tool_registry=runtime.registry,
        tool_executor=runtime.executor,
        native_tools=False,
        tool_schema_role=MessageRole.USER,
    )

    renderer = _StreamRenderer(verbose=verbose)
    async for event in coordinator.run(
        conversation=conversation,
        messages=[user_msg],
        user_message=user_msg,
        model=model or runtime.default_model,
        provider=runtime.provider_id,
    ):
        renderer.feed(event)
    print()
    return 0


async def cmd_run(
    runtime: "AstraRuntime",
    text: str,
    *,
    model: str | None = None,
    verbose: bool = False,
) -> int:
    """One-shot agent run: plan → execute → reflect → answer (streamed)."""
    from app.domain.enums import MessageRole
    from app.services.coordinator import ChatCoordinator

    from .cli_interactive import _StreamRenderer, new_conversation, user_message

    conversation = new_conversation(
        model=model or runtime.default_model,
        provider=runtime.provider_id,
        title="CLI run",
    )
    msgs = [user_message(conversation.id, text)]
    user_msg = msgs[-1]

    coordinator = ChatCoordinator(
        llm_router=runtime.router,
        tool_registry=runtime.registry,
        tool_executor=runtime.executor,
        native_tools=False,
        tool_schema_role=MessageRole.USER,
    )

    renderer = _StreamRenderer(verbose=verbose)
    async for event in coordinator.run(
        conversation=conversation,
        messages=msgs,
        user_message=user_msg,
        model=model or runtime.default_model,
        provider=runtime.provider_id,
    ):
        renderer.feed(event)
    print()
    return 0


async def cmd_tools_list(runtime: "AstraRuntime") -> int:
    """List registered tools and their capabilities."""
    print()
    tools = sorted(runtime.registry.list_tools(), key=lambda t: t.name)
    if not tools:
        print("  No tools registered.")
        return 0
    print(f"  {len(tools)} tool(s) registered:\n")
    for tool in tools:
        caps = ", ".join(tool.capabilities) if tool.capabilities else "-"
        print(f"    {tool.name:<18} {caps}")
    print()
    return 0


async def cmd_tools_call(runtime: "AstraRuntime", tool_name: str, args: dict[str, Any]) -> int:
    """Call a single tool with provided key=value arguments."""
    if not runtime.registry.exists(tool_name):
        print(f"  Unknown tool '{tool_name}'. Use 'astra tools list' to see available tools.")
        return 1

    result = await runtime.executor.execute(
        ToolCall(tool_name=tool_name, arguments=args),
        runtime.tool_context(),
    )
    print()
    if result.output:
        print(result.output)
    if result.error:
        print(f"\n  ✗ {result.error}")
    if result.metadata:
        summary = {k: v for k, v in result.metadata.items() if k not in {"models", "model"}}
        print("\n  metadata:", json.dumps(summary, default=str))
    print(f"\n  {'PASS' if result.success else 'FAIL'} "
          f"({result.execution_time_ms or 0}ms)")
    return 0 if result.success else 1


async def cmd_models(runtime: "AstraRuntime") -> int:
    """List models available via the configured provider (LM Studio)."""
    print()
    models = await runtime.list_lmstudio_models()
    if not models:
        print("  No models reachable. Start LM Studio (localhost:1234) first.")
        return 1
    print(f"  {len(models)} model(s) available:\n")
    for model in sorted(models):
        print(f"    • {model}")
    print()
    return 0


def cmd_serve(*, host: str, port: int, reload: bool) -> int:
    """Start the FastAPI backend server."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def parse_keyval(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` CLI arguments into a dict.

    Values starting with ``{`` or ``[`` are parsed as JSON; ``true`` /
    ``false`` / ``null`` literals are coerced to Python booleans / None.
    """
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            key, value = pair, "true"
        else:
            key, value = pair.split("=", 1)
        key = key.strip()
        value = _coerce(value)
        result[key] = value
    return result


def _coerce(value: str) -> Any:
    text = value.strip()
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "null":
        return None
    return text