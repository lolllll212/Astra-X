"""Benchmark and stress-test harness for Astra-X core components.

Usage:
    py -3 scripts/benchmark.py              # run all benchmarks
    py -3 scripts/benchmark.py --quick      # quick smoke-test
    py -3 scripts/benchmark.py --profile    # with cProfile output

Measures throughput and latency for:
  - CapabilityRegistry lookups
  - Tracer span creation and nesting
  - ToolRegistry operations
  - TaskGraph operations
  - PlannedCoordinator flow (mocked)
  - ChatCoordinator flow (mocked)

Exit code is 0 if all benchmarks pass within acceptable thresholds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import os
os.environ.setdefault("ASTRA_DATABASE_URL", "sqlite+aiosqlite://")
os.environ.setdefault("ASTRA_ENVIRONMENT", "testing")
os.environ.setdefault("ASTRA_SECRET_KEY", "bench-secret-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start

    def report(self) -> str:
        return f"  {self.label}: {self.elapsed*1000:.1f} ms"


def bench(name: str, fn, iterations: int = 1000):
    """Run *fn* *iterations* times and report."""
    timer = Timer(name)
    with timer:
        for _ in range(iterations):
            fn()
    print(timer.report())
    return timer.elapsed


async def bench_async(name: str, fn, iterations: int = 100):
    """Run async *fn* *iterations* times and report."""
    timer = Timer(name)
    with timer:
        for _ in range(iterations):
            await fn()
    print(timer.report())
    return timer.elapsed


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

ALL_OK: bool = True
MAX_ACCEPTABLE_MS: dict[str, float] = {
    "capability_resolve": 10.0,
    "tracer_flat": 1500.0,
    "tracer_nested": 1500.0,
    "tool_executor": 2000.0,
    "task_graph_ops": 5000.0,
    "coordinator_run": 500.0,
}


def _check(label: str, elapsed: float) -> None:
    global ALL_OK
    key = label.split("(")[0].strip()
    threshold = MAX_ACCEPTABLE_MS.get(key, 10.0)
    ms = elapsed * 1000
    if ms > threshold:
        print(f"    (!) {ms:.1f} ms exceeds threshold {threshold:.1f} ms")
        ALL_OK = False


def benchmark_registry() -> None:
    from app.tools.base import Tool
    from app.tools.capabilities import CapabilityRegistry
    from app.tools.registry import ToolRegistry
    from app.tools.result import ToolResult

    class _FastTool(Tool):
        def __init__(self, name: str) -> None:
            self._name = name
        @property
        def name(self) -> str:
            return self._name
        @property
        def description(self) -> str:
            return "fast"
        @property
        def schema(self) -> object:
            return object()
        async def _execute(self, context: object, **kwargs: object) -> ToolResult:
            return ToolResult(success=True, output="ok")

    tr = ToolRegistry()
    for i in range(10):
        tr.register(_FastTool(f"tool_{i}"))
    cr = CapabilityRegistry(tr)

    def resolve():
        cr.resolve("tool_0")

    elapsed = bench("capability_resolve (10k)", resolve, 10_000)
    _check("capability_resolve", elapsed)


def benchmark_tracer() -> None:
    from app.core.tracing import Tracer

    t = Tracer()
    def flat():
        for _ in range(10):
            s = t.start("op")
            t.end(s)

    t2 = Tracer()
    def nested():
        root = t2.start("root")
        for _ in range(5):
            c = t2.start("child")
            t2.end(c)
        t2.end(root)

    e1 = bench("tracer_flat (10k)", flat, 10_000)
    _check("tracer_flat", e1)
    e2 = bench("tracer_nested (10k)", nested, 10_000)
    _check("tracer_nested", e2)


def benchmark_task_graph() -> None:
    from app.agents.models.task import Task, TaskStatus
    from app.agents.task_graph import TaskGraph

    def ops():
        g = TaskGraph()
        for i in range(100):
            g.add_task(Task(id=f"t{i}", description=f"task {i}"))
            if i > 0:
                g.update_status(f"t{i-1}", TaskStatus.COMPLETED)
            g.get_ready()

    elapsed = bench("task_graph_ops (1k)", ops, 1_000)
    _check("task_graph_ops", elapsed)


async def benchmark_tools() -> None:
    from app.tools.executor import ToolExecutor
    from app.tools.models import ToolCall, ToolParameter, ToolSchema
    from app.tools.context import ToolContext
    from app.tools.registry import ToolRegistry
    from app.tools.base import Tool
    from app.tools.result import ToolResult

    class _QuickTool(Tool):
        def __init__(self) -> None:
            super().__init__()
        @property
        def name(self) -> str:
            return "quick"
        @property
        def description(self) -> str:
            return "quick tool"
        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(name="quick", description="quick", parameters=[])
        async def _execute(self, context: object, **kwargs: object) -> ToolResult:
            return ToolResult(success=True, output="ok")

    tr = ToolRegistry()
    tr.register(_QuickTool())
    ex = ToolExecutor(tr)
    call = ToolCall(tool_name="quick")

    async def run():
        await ex.execute(call, ToolContext())

    elapsed = await bench_async("tool_executor (1k)", run, 1_000)
    _check("tool_executor", elapsed)


async def benchmark_coordinator() -> None:
    """Benchmark coordinator with full mocked pipeline (100 runs)."""
    from unittest.mock import AsyncMock, MagicMock
    from app.agents.coordinator import Coordinator
    from app.agents.executor import Executor
    from app.agents.memory_manager import MemoryManager
    from app.agents.models.execution import ExecutionResult, ReflectionResult
    from app.agents.models.plan import Plan
    from app.agents.models.task import Task, TaskStatus
    from app.agents.planner import Planner
    from app.agents.reflection import Reflection
    from app.domain.message import Message, TextBlock
    from app.llm.models import CompletionResponse
    from app.llm.router import LLMRouter

    planner = MagicMock(spec=Planner)
    planner.plan = AsyncMock(return_value=Plan(
        goal="bench",
        tasks=[Task(id="t1", description="bench task")],
    ))

    executor = MagicMock(spec=Executor)
    executor.execute = AsyncMock(return_value=ExecutionResult(
        task_id="t1", status=TaskStatus.COMPLETED, output="bench result",
    ))

    from app.agents.models.execution import ReflectionDecision as RD
    reflection = MagicMock(spec=Reflection)
    reflection.reflect = AsyncMock(return_value=ReflectionResult(
        decision=RD.ACCEPT,
        reason="ok", confidence=1.0,
    ))

    mm = MagicMock(spec=MemoryManager)
    mm.get_context = AsyncMock(return_value="")
    mm.store_result = AsyncMock()

    llm_router = MagicMock(spec=LLMRouter)
    llm_router.generate = AsyncMock(return_value=CompletionResponse(
        message=Message(
            id="r1", conversation_id="", role="assistant",
            content=[TextBlock(text="no, done")],
        ),
    ))

    coord = Coordinator(
        planner=planner, executor=executor, reflection=reflection,
        memory_manager=mm, llm_router=llm_router,
    )

    async def run():
        await coord.run(conversation_id="bench", goal="bench goal")

    elapsed = await bench_async("coordinator_run (100)", run, 100)
    _check("coordinator_run", elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Astra-X benchmark harness")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke-test only")
    parser.add_argument("--profile", action="store_true", help="Run under cProfile")
    args = parser.parse_args()

    benchmarks: list[tuple[str, object]] = [
        ("CapabilityRegistry", benchmark_registry),
        ("Tracer", benchmark_tracer),
        ("TaskGraph", benchmark_task_graph),
        ("ToolExecutor", benchmark_tools),
    ]
    if not args.quick:
        benchmarks.append(("Coordinator", benchmark_coordinator))

    print(f"Astra-X Benchmark ({'quick' if args.quick else 'full'})")
    print("=" * 50)

    for name, fn in benchmarks:
        print(f"\n[{name}]")
        if args.profile:
            import cProfile
            import pstats
            prof = cProfile.Profile()
            prof.enable()
            result = fn() if not hasattr(fn, '__call__') else await fn() if hasattr(fn, '__call__') and fn.__name__ in ('benchmark_tools', 'benchmark_coordinator') else fn()
            prof.disable()
            stats = pstats.Stats(prof).sort_stats("cumtime")
            stats.print_stats(15)
        else:
            if name in ("ToolExecutor", "Coordinator"):
                await fn()
            else:
                fn()

    print(f"\n{'=' * 50}")
    if ALL_OK:
        print("All benchmarks passed.")
        return 0
    else:
        print("Some benchmarks exceeded thresholds.")
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
