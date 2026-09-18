"""Unit tests for the planner agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.planner import Planner
from app.domain.message import Message, TextBlock
from app.llm.exceptions import GenerationError
from app.llm.models import CompletionResponse
from app.llm.router import LLMRouter


def _mock_router(json_response: str) -> MagicMock:
    router = MagicMock(spec=LLMRouter)
    router.generate = AsyncMock(
        return_value=CompletionResponse(
            message=Message(
                id="r1",
                conversation_id="",
                role="assistant",
                content=[TextBlock(text=json_response)],
            ),
        ),
    )
    return router


class TestPlanner:
    @pytest.mark.asyncio
    async def test_plan_returns_valid_plan(self) -> None:
        router = _mock_router("""[
            {"id": "t1", "description": "Search web", "capability": "search_web", "dependencies": []},
            {"id": "t2", "description": "Summarize", "capability": null, "dependencies": ["t1"]}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Research topic")
        assert plan.goal == "Research topic"
        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "t1"
        assert plan.tasks[0].capability == "search_web"
        assert plan.tasks[1].dependencies == ["t1"]

    @pytest.mark.asyncio
    async def test_plan_with_memory_context(self) -> None:
        """Memory context is prepended when provided."""
        router = _mock_router("""[
            {"id": "t1", "description": "Use prior result", "capability": null, "dependencies": []}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Continue work", memory_context="Previous result: X")
        assert plan.goal == "Continue work"
        assert len(plan.tasks) == 1

    @pytest.mark.asyncio
    async def test_plan_empty_memory_context(self) -> None:
        """Empty memory context is handled gracefully."""
        router = _mock_router("""[
            {"id": "t1", "description": "Do something", "capability": null, "dependencies": []}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Do work", memory_context="")
        assert len(plan.tasks) == 1

    @pytest.mark.asyncio
    async def test_plan_parses_json_from_code_fence(self) -> None:
        """LLM may wrap JSON in markdown code fences."""
        router = _mock_router("""```json
[
    {"id": "t1", "description": "Search", "capability": "search_web", "dependencies": []}
]
```""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Search")
        assert len(plan.tasks) == 1
        assert plan.tasks[0].capability == "search_web"

    @pytest.mark.asyncio
    async def test_plan_parses_json_with_surrounding_text(self) -> None:
        """LLM may include explanatory text before/after JSON."""
        router = _mock_router("""Here is the plan:
[
    {"id": "t1", "description": "Search", "capability": "search_web", "dependencies": []}
]
That should work.""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Search")
        assert len(plan.tasks) == 1

    @pytest.mark.asyncio
    async def test_plan_raises_on_non_json_output(self) -> None:
        router = _mock_router("I cannot plan this request.")
        planner = Planner(llm_router=router)
        with pytest.raises(GenerationError, match="could not be parsed"):
            await planner.plan(goal="Impossible")

    @pytest.mark.asyncio
    async def test_plan_raises_on_empty_task_list(self) -> None:
        router = _mock_router("""[]""")
        planner = Planner(llm_router=router)
        with pytest.raises(GenerationError, match="empty task list"):
            await planner.plan(goal="Nothing")

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self) -> None:
        router = _mock_router("""{bad json]""")
        planner = Planner(llm_router=router)
        with pytest.raises(GenerationError, match="could not be parsed"):
            await planner.plan(goal="Bad")

    @pytest.mark.asyncio
    async def test_plan_task_without_capability(self) -> None:
        """Capability can be null/omitted for thinking tasks."""
        router = _mock_router("""[
            {"id": "t1", "description": "Think", "dependencies": []}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Think")
        assert plan.tasks[0].capability is None

    @pytest.mark.asyncio
    async def test_plan_task_with_profile(self) -> None:
        """Task profile is parsed when present in the planner output."""
        router = _mock_router("""[
            {
                "id": "t1",
                "description": "Write code",
                "capability": null,
                "dependencies": [],
                "profile": {"requires_coding": true, "reasoning": "medium"}
            }
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Code")
        task = plan.tasks[0]
        assert task.profile is not None
        assert task.profile["requires_coding"] is True
        assert task.profile["reasoning"] == "medium"

    @pytest.mark.asyncio
    async def test_plan_task_without_profile(self) -> None:
        """Task profile is None when omitted."""
        router = _mock_router("""[
            {"id": "t1", "description": "Simple task", "dependencies": []}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Simple")
        assert plan.tasks[0].profile is None

    @pytest.mark.asyncio
    async def test_plan_task_with_non_dict_profile_ignored(self) -> None:
        """Non-dict profile values are silently ignored."""
        router = _mock_router("""[
            {"id": "t1", "description": "Task", "profile": "invalid", "dependencies": []}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Test")
        assert plan.tasks[0].profile is None

    @pytest.mark.asyncio
    async def test_plan_single_task(self) -> None:
        router = _mock_router("""[
            {"id": "t1", "description": "Do one thing", "capability": null, "dependencies": []}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="One thing")
        assert plan.task_count == 1

    @pytest.mark.asyncio
    async def test_plan_multiple_independent_tasks(self) -> None:
        router = _mock_router("""[
            {"id": "t1", "description": "Task A", "capability": null, "dependencies": []},
            {"id": "t2", "description": "Task B", "capability": null, "dependencies": []},
            {"id": "t3", "description": "Task C", "capability": null, "dependencies": []}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Multiple")
        assert plan.task_count == 3

    @pytest.mark.asyncio
    async def test_plan_chained_dependencies(self) -> None:
        router = _mock_router("""[
            {"id": "t1", "description": "First", "capability": null, "dependencies": []},
            {"id": "t2", "description": "Second", "capability": null, "dependencies": ["t1"]},
            {"id": "t3", "description": "Third", "capability": null, "dependencies": ["t2"]}
        ]""")
        planner = Planner(llm_router=router)
        plan = await planner.plan(goal="Chain")
        assert plan.tasks[2].dependencies == ["t2"]

    @pytest.mark.asyncio
    async def test_plan_execute_not_implemented(self) -> None:
        planner = Planner(llm_router=MagicMock(spec=LLMRouter))
        with pytest.raises(NotImplementedError):
            await planner.execute(MagicMock())

    @pytest.mark.asyncio
    async def test_plan_reflect_not_implemented(self) -> None:
        planner = Planner(llm_router=MagicMock(spec=LLMRouter))
        with pytest.raises(NotImplementedError):
            await planner.reflect(MagicMock())

    def test_extract_json_object(self) -> None:
        """_extract_json handles JSON objects (not just arrays)."""
        result = Planner._extract_json('{"key": "value"}')
        assert result is not None
        assert '"key"' in result

    def test_extract_json_no_json(self) -> None:
        assert Planner._extract_json("No brackets here") is None

    def test_extract_json_unmatched_brackets(self) -> None:
        result = Planner._extract_json("[incomplete")
        assert result is None

    def test_extract_json_nested(self) -> None:
        result = Planner._extract_json('{"outer": {"inner": "value"}}')
        assert result is not None
        assert '"inner"' in result

    def test_extract_json_with_string_braces(self) -> None:
        """Braces inside JSON strings should not affect bracket matching.
        _extract_json finds the first '[' or '{' — use an object-first string."""
        text = '{"key": "a{b}c", "nested": {"inner": true}}'
        result = Planner._extract_json(text)
        assert result is not None
        assert '"nested"' in result

    def test_extract_json_array_with_string_braces(self) -> None:
        """String containing braces inside an array element."""
        text = '[{"key": "a{b}c"}]'
        result = Planner._extract_json(text)
        assert result is not None
        assert '"key"' in result
