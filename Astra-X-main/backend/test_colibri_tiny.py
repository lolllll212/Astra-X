"""Test Colibri with tiny models."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.colibri import ColibriTool
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.models import ToolCall


async def test_colibri_tiny() -> None:
    ctx = ToolContext(workspace=".")
    registry = __import__("app.tools.registry", fromlist=["ToolRegistry"]).ToolRegistry()
    registry.register(ColibriTool())
    executor = ToolExecutor(registry)

    # Test with deepseek_v4_tiny
    model_dir = r"C:\Users\Ashut\Downloads\colibri-main\c\deepseek_v4_tiny"
    
    print("=" * 60)
    print("Testing Colibri with deepseek_v4_tiny")
    print("=" * 60)
    
    # Test run
    res = await executor.execute(
        ToolCall(
            tool_name="colibri",
            arguments={
                "action": "run",
                "prompt": "Hello, what is 2+2?",
                "model_dir": model_dir,
                "ram": 8,
                "ngen": 100,
            },
        ),
        ctx,
    )
    print(f"Run success: {res.success}")
    print(f"Output: {res.output[:500] if res.output else '(none)'}")
    print(f"Error: {res.error}")
    print()

    # Test plan
    res = await executor.execute(
        ToolCall(
            tool_name="colibri",
            arguments={
                "action": "plan",
                "model_dir": model_dir,
            },
        ),
        ctx,
    )
    print(f"Plan success: {res.success}")
    print(f"Output: {res.output[:500] if res.output else '(none)'}")
    print(f"Error: {res.error}")
    print()

    # Test doctor
    res = await executor.execute(
        ToolCall(
            tool_name="colibri",
            arguments={
                "action": "doctor",
                "model_dir": model_dir,
            },
        ),
        ctx,
    )
    print(f"Doctor success: {res.success}")
    print(f"Output: {res.output[:500] if res.output else '(none)'}")
    print(f"Error: {res.error}")
    print()

    # Test with qwen36_tiny
    model_dir = r"C:\Users\Ashut\Downloads\colibri-main\c\qwen36_tiny"
    
    print("=" * 60)
    print("Testing Colibri with qwen36_tiny")
    print("=" * 60)
    
    res = await executor.execute(
        ToolCall(
            tool_name="colibri",
            arguments={
                "action": "run",
                "prompt": "Write a Python hello world function",
                "model_dir": model_dir,
                "ram": 8,
                "ngen": 100,
            },
        ),
        ctx,
    )
    print(f"Run success: {res.success}")
    print(f"Output: {res.output[:500] if res.output else '(none)'}")
    print(f"Error: {res.error}")
    print()

    # Test get_status
    res = await executor.execute(
        ToolCall(tool_name="colibri", arguments={"action": "get_status"}),
        ctx,
    )
    print("Colibri status:")
    print(f"  success: {res.success}")
    print(f"  output: {res.output[:800] if res.output else '(none)'}")
    print(f"  error: {res.error}")


if __name__ == "__main__":
    asyncio.run(test_colibri_tiny())