from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from app.agents.learning_manager import LearningManager
from app.agents.learning_store import LearningStore
from app.agents.models.execution import ExecutionResult, ReflectionDecision, ReflectionResult
from app.agents.models.pattern import ExecutionPattern
from app.agents.models.plan import Plan
from app.agents.models.task import Task, TaskStatus
from app.database.converters.pattern_mapper import pattern_from_model, pattern_to_model
from app.database.models.execution_pattern import ExecutionPatternModel

# =========================================================================
# ExecutionPattern domain model
# =========================================================================


class TestExecutionPattern:
    def test_basic_construction(self) -> None:
        p = ExecutionPattern(goal_pattern="research {topic}")
        assert p.goal_pattern == "research {topic}"
        assert p.success_count == 1
        assert p.total_count == 1
        assert p.capability is None
        assert isinstance(p.id, str)

    def test_frozen(self) -> None:
        p = ExecutionPattern(goal_pattern="test")
        with pytest.raises(ValueError, match="frozen"):
            p.goal_pattern = "changed"

    def test_with_all_fields(self) -> None:
        now = datetime.now()
        p = ExecutionPattern(
            goal_pattern="write code for {task}",
            capability="execute_python",
            strategy_summary="Start with search, then implement",
            plan_template="- Search for examples\n- Write code\n- Test",
            tags=["code", "python", "implementation"],
            success_count=5,
            total_count=7,
            avg_confidence=0.85,
            avg_importance=0.72,
            last_success_at=now,
            created_at=now,
            updated_at=now,
        )
        assert p.goal_pattern == "write code for {task}"
        assert p.capability == "execute_python"
        assert p.tags == ["code", "python", "implementation"]
        assert p.success_count == 5
        assert p.total_count == 7


# =========================================================================
# Pattern mapper
# =========================================================================


class TestPatternMapper:
    def test_pattern_to_model(self) -> None:
        now = datetime.now()
        domain = ExecutionPattern(
            id=str(uuid4()),
            goal_pattern="research {topic}",
            capability="search_web",
            strategy_summary="Search first, then scrape",
            plan_template="- Search\n- Scrape\n- Summarize",
            tags=["research", "web"],
            success_count=3,
            total_count=4,
            avg_confidence=0.9,
            avg_importance=0.7,
            last_success_at=now,
            created_at=now,
            updated_at=now,
        )
        model = pattern_to_model(domain)
        assert model.id == domain.id
        assert model.goal_pattern == "research {topic}"
        assert model.tags == "research,web"
        assert model.success_count == 3

    def test_model_to_domain(self) -> None:
        now = datetime.now()
        model = ExecutionPatternModel(
            id=str(uuid4()),
            goal_pattern="research {topic}",
            capability="search_web",
            strategy_summary="Search first",
            plan_template="- Search\n- Summarize",
            tags="research,web",
            success_count=3,
            total_count=4,
            avg_confidence=0.9,
            avg_importance=0.7,
            last_success_at=now,
            created_at=now,
            updated_at=now,
        )
        domain = pattern_from_model(model)
        assert domain.goal_pattern == "research {topic}"
        assert domain.tags == ["research", "web"]
        assert domain.success_count == 3

    def test_model_to_domain_empty_tags(self) -> None:
        from uuid import uuid4
        model = ExecutionPatternModel(id=str(uuid4()), goal_pattern="test", tags="")
        domain = pattern_from_model(model)
        assert domain.tags == []


# =========================================================================
# LearningStore (in-memory)
# =========================================================================


class TestLearningStore:
    def test_store_and_retrieve(self) -> None:
        store = LearningStore()
        p = ExecutionPattern(goal_pattern="research {topic}")
        store._cache[p.goal_pattern] = p
        assert store.get_by_goal_pattern("research {topic}") is p
        assert store.get_by_goal_pattern("nonexistent") is None

    def test_extract_keywords(self) -> None:
        store = LearningStore()
        keywords = store._extract_keywords("Research the best Python web frameworks")
        assert "research" in keywords
        assert "python" in keywords
        assert "web" in keywords
        assert "frameworks" in keywords
        assert "the" not in keywords

    def test_match_score(self) -> None:
        p = ExecutionPattern(
            goal_pattern="research {topic}",
            tags=["research", "web"],
            strategy_summary="Search first, then scrape top results",
        )
        weights = {"similarity": 1.0, "success_rate": 0.0, "recency": 0.0, "cost": 0.0, "confidence": 0.0}
        score = LearningStore._rank_score(p, ["research", "python", "web"], weights)
        assert score > 0

    def test_match_score_zero(self) -> None:
        p = ExecutionPattern(
            goal_pattern="write code",
            tags=["code"],
            strategy_summary="Write Python code",
        )
        weights = {"similarity": 1.0, "success_rate": 0.0, "recency": 0.0, "cost": 0.0, "confidence": 0.0}
        score = LearningStore._rank_score(p, ["cooking", "recipes"], weights)
        assert score == 0.0

    def test_search_by_keywords(self) -> None:
        store = LearningStore()
        p1 = ExecutionPattern(goal_pattern="research {topic}", tags=["research", "web"])
        p2 = ExecutionPattern(goal_pattern="write code", tags=["code", "python"])
        p3 = ExecutionPattern(goal_pattern="cook food", tags=["cooking", "recipes"])
        for p in (p1, p2, p3):
            store._cache[p.goal_pattern] = p

        results = store.search("Research Python web frameworks", limit=2)
        assert len(results) >= 1
        assert results[0].goal_pattern == "research {topic}"

    def test_search_no_match(self) -> None:
        store = LearningStore()
        p = ExecutionPattern(goal_pattern="research {topic}", tags=["research"])
        store._cache[p.goal_pattern] = p
        results = store.search("completely unrelated query about cooking", limit=5)
        assert len(results) == 0

    def test_save_updates_cache(self) -> None:
        store = LearningStore()
        p = ExecutionPattern(goal_pattern="test pattern")
        import asyncio
        saved = asyncio.run(store.save(p))
        assert store.get_by_goal_pattern("test pattern") is saved


# =========================================================================
# LearningManager
# =========================================================================


class TestLearningManager:
    def _make_result(self, task_id: str, status: TaskStatus = TaskStatus.COMPLETED, tool: str | None = None) -> ExecutionResult:
        from datetime import UTC
        return ExecutionResult(
            task_id=task_id,
            status=status,
            output="done" if status == TaskStatus.COMPLETED else None,
            tool_name=tool,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            metadata={},
        )

    def _make_reflection(self, decision: ReflectionDecision = ReflectionDecision.ACCEPT, confidence: float = 0.9) -> ReflectionResult:
        return ReflectionResult(
            decision=decision,
            reason="good",
            confidence=confidence,
        )

    def _make_plan(self, *task_descs: str) -> Plan:
        tasks = [
            Task(id=f"t{i}", description=desc)
            for i, desc in enumerate(task_descs)
        ]
        return Plan(goal=task_descs[0] if task_descs else "goal", tasks=tasks)

    @pytest.mark.asyncio
    async def test_extract_pattern_successful(self) -> None:
        lm = LearningManager()
        plan = self._make_plan("Research Python", "Write summary")
        results = [
            self._make_result("t0", tool="search_web"),
            self._make_result("t1"),
        ]
        reflections = [self._make_reflection(), self._make_reflection(confidence=0.85)]

        pattern = await lm.extract_pattern(
            goal="Research Python web frameworks",
            plan=plan,
            results=results,
            reflections=reflections,
        )
        assert pattern is not None
        assert pattern.success_count == 1
        assert pattern.capability == "search_web"
        assert len(pattern.tags) > 0

    @pytest.mark.asyncio
    async def test_extract_pattern_skipped_on_failure(self) -> None:
        lm = LearningManager()
        plan = self._make_plan("Do something")
        results = [self._make_result("t0", status=TaskStatus.FAILED)]

        pattern = await lm.extract_pattern(
            goal="Do something",
            plan=plan,
            results=results,
            reflections=[],
        )
        assert pattern is None

    @pytest.mark.asyncio
    async def test_extract_pattern_skipped_low_confidence(self) -> None:
        lm = LearningManager()
        plan = self._make_plan("Do something")
        results = [self._make_result("t0")]
        reflections = [self._make_reflection(confidence=0.3)]

        pattern = await lm.extract_pattern(
            goal="Do something",
            plan=plan,
            results=results,
            reflections=reflections,
        )
        assert pattern is None

    @pytest.mark.asyncio
    async def test_extract_pattern_empty_results(self) -> None:
        lm = LearningManager()
        plan = self._make_plan("Do something")
        pattern = await lm.extract_pattern(
            goal="test",
            plan=plan,
            results=[],
            reflections=[],
        )
        assert pattern is None

    @pytest.mark.asyncio
    async def test_extract_pattern_updates_existing(self) -> None:
        lm = LearningManager()
        plan = self._make_plan("Research Python")

        # First extraction
        results = [self._make_result("t0", tool="search_web")]
        reflections = [self._make_reflection(confidence=0.9)]
        p1 = await lm.extract_pattern(
            goal="Research Python web frameworks",
            plan=plan,
            results=results,
            reflections=reflections,
        )
        assert p1 is not None
        assert p1.success_count == 1

        # Second extraction (same normalized goal)
        p2 = await lm.extract_pattern(
            goal="Research Python web frameworks",
            plan=plan,
            results=results,
            reflections=reflections,
        )
        assert p2 is not None
        assert p2.success_count == 2
        assert p2.total_count == 2

    def test_normalize_goal(self) -> None:
        normal = LearningManager._normalize_goal("Research the top 5 Python web frameworks")
        assert "{n}" in normal
        assert "research" in normal
        assert "the" not in normal

    def test_build_strategy_summary(self) -> None:
        plan = self._make_plan("Research Python", "Write code")
        results = [
            self._make_result("t0", tool="search_web"),
            self._make_result("t1", tool="execute_python"),
        ]
        summary = LearningManager._build_strategy_summary(plan, results)
        assert "search_web" in summary or "execute_python" in summary

    def test_build_plan_template(self) -> None:
        plan = self._make_plan("Research Python", "Write code")
        tpl = LearningManager._build_plan_template(plan)
        assert "- Research Python" in tpl
        assert "- Write code" in tpl

    @pytest.mark.asyncio
    async def test_get_lessons_no_patterns(self) -> None:
        lm = LearningManager()
        lessons = await lm.get_lessons("Research something")
        assert lessons == ""

    @pytest.mark.asyncio
    async def test_get_lessons_with_patterns(self) -> None:
        lm = LearningManager()
        # Pre-populate a pattern
        p = ExecutionPattern(
            goal_pattern="research {topic}",
            tags=["research", "python"],
            strategy_summary="Search first",
        )
        await lm._store.save(p)

        lessons = await lm.get_lessons("Research Python web frameworks")
        assert "Lessons learned" in lessons
        assert "research {topic}" in lessons
        assert "Search first" in lessons


class TestLearningManagerStatic:
    @staticmethod
    def _make_plan(*task_descs: str) -> Plan:
        tasks = [
            Task(id=f"t{i}", description=desc)
            for i, desc in enumerate(task_descs)
        ]
        return Plan(goal=task_descs[0] if task_descs else "goal", tasks=tasks)

    @staticmethod
    def _make_result(task_id: str, status: TaskStatus = TaskStatus.COMPLETED, tool: str | None = None) -> ExecutionResult:
        from datetime import UTC
        return ExecutionResult(
            task_id=task_id,
            status=status,
            output="done" if status == TaskStatus.COMPLETED else None,
            tool_name=tool,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            metadata={},
        )

    def test_build_strategy_summary(self) -> None:
        plan = self._make_plan("Research Python", "Write code")
        results = [
            self._make_result("t0", tool="search_web"),
            self._make_result("t1"),
        ]
        summary = LearningManager._build_strategy_summary(plan, results)
        assert "search_web" in summary

    def test_build_plan_template(self) -> None:
        plan = self._make_plan("Research Python", "Write code")
        tpl = LearningManager._build_plan_template(plan)
        assert "- Research Python" in tpl
        assert "- Write code" in tpl
