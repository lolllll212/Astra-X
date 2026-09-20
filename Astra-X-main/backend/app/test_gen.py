import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx

from app.cli_runtime import build_runtime, new_conversation, user_message
from app.domain.enums import MessageRole
from app.domain.message import Message, TextBlock, ToolCallBlock, ToolResultBlock
from app.llm.providers.adapters.chat_completions import domain_to_openai_messages
from app.llm.types import CompletionRequest, Conversation, GenerationParams

MODEL = "qwen2.5-coder-7b-instruct"


def msg(role, blocks):
    return Message(
        id=str(uuid4()), conversation_id="c", role=role,
        content=blocks, created_at=datetime.now(UTC),
    )


def build_payload(messages):
    dto = domain_to_openai_messages(messages)
    print("SERIALIZED:")
    for m in dto:
        print("  ", m)
    return {
        "model": MODEL,
        "messages": dto,
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
        "stop": None,
    }


async def post(client, label, messages):
    r = await client.post(
        "http://localhost:1234/v1/chat/completions", json=build_payload(messages),
        timeout=180,
    )
    print(f"--- {label} -> {r.status_code}")
    print("  ", (r.text[:180].replace(chr(10), " ")))
    return r


async def main():
    runtime = build_runtime()
    conv = new_conversation(model=MODEL, provider="lm_studio", title="t")
    um = user_message(conv.id, "Use the calculator to compute 1234 * 5678.")

    sys_schema = msg(MessageRole.SYSTEM, [TextBlock(text="You have a calculator tool.")])

    assistant_call_dict = msg(MessageRole.ASSISTANT, [ToolCallBlock(
        tool_call_id="tc-1", tool_name="calculator",
        arguments={"expression": "1234 * 5678"},
    )])
    assistant_call_str = msg(MessageRole.ASSISTANT, [ToolCallBlock(
        tool_call_id="tc-1", tool_name="calculator",
        arguments='{"expression": "1234 * 5678"}',
    )])
    tool_result = msg(MessageRole.TOOL, [ToolResultBlock(
        tool_call_id="tc-1", tool_name="calculator", output="7006652",
        is_error=False,
    )])

    async with httpx.AsyncClient() as client:
        await post(client, "A_user_only", [um])
        await post(client, "B_sys_user", [sys_schema, um])
        await post(client, "C_user_assistant_dict_tooltag", [um, assistant_call_dict, tool_result])
        await post(client, "D_user_assistant_str_tooltag", [um, assistant_call_str, tool_result])

    await runtime.close()


asyncio.run(main())