"""Goal Manager — persists and tracks the Mission → Objective → Goal → Action hierarchy.

The coordinator calls into the GoalManager to create or resume goals,
log actions, and update status so that work persists across sessions
and enables long-running autonomous missions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.agents.models.goal import (
    Action,
    GoalNode,
    GoalStatus,
    Mission,
    MissionStatus,
    Objective,
    ObjectiveStatus,
)
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.database.repositories.goal_repository import GoalRepository

logger = get_logger(__name__)


class GoalManager:
    """High-level API for the goal hierarchy.

    Usage inside the coordinator::

        goal = await goal_manager.start_or_resume_goal(mission_id, "Deploy to prod")
        ...
        await goal_manager.complete_goal(goal.id, result_summary="Deployed v2.1.0")
        await goal_manager.record_action(goal.id, task_id="t1", tool_name="kubectl", ...)
    """

    def __init__(
        self,
        repository: GoalRepository,
    ) -> None:
        self._repo = repository

    # ------------------------------------------------------------------
    # Mission lifecycle
    # ------------------------------------------------------------------

    async def create_mission(
        self,
        title: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Mission:
        """Start a new long-running mission."""
        mission = Mission(
            title=title,
            description=description,
            metadata=metadata or {},
        )
        result = await self._repo.create_mission(mission)
        logger.info("goal.mission_created", mission_id=result.id, title=title)
        return result

    async def get_mission(self, mission_id: str) -> Mission | None:
        return await self._repo.get_mission(mission_id)

    async def list_active_missions(self, limit: int = 10) -> list[Mission]:
        return await self._repo.list_missions(status="active", limit=limit)

    async def complete_mission(self, mission_id: str) -> None:
        await self._repo.update_mission(mission_id, status="completed")
        logger.info("goal.mission_completed", mission_id=mission_id)

    async def pause_mission(self, mission_id: str) -> None:
        await self._repo.update_mission(mission_id, status="paused")
        logger.info("goal.mission_paused", mission_id=mission_id)

    async def resume_mission(self, mission_id: str) -> Mission | None:
        mission = await self._repo.update_mission(mission_id, status="active")
        if mission:
            logger.info("goal.mission_resumed", mission_id=mission_id)
        return mission

    # ------------------------------------------------------------------
    # Objective lifecycle
    # ------------------------------------------------------------------

    async def create_objective(
        self,
        mission_id: str,
        title: str,
        description: str = "",
        order: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Objective:
        obj = Objective(
            mission_id=mission_id,
            title=title,
            description=description,
            order=order,
            metadata=metadata or {},
        )
        result = await self._repo.create_objective(obj)
        logger.info("goal.objective_created", objective_id=result.id, mission_id=mission_id)
        return result

    async def list_objectives(self, mission_id: str) -> list[Objective]:
        return await self._repo.list_objectives(mission_id)

    async def complete_objective(self, objective_id: str) -> None:
        await self._repo.update_objective(objective_id, status="completed")
        logger.info("goal.objective_completed", objective_id=objective_id)

    async def get_next_incomplete_objective(
        self,
        mission_id: str,
    ) -> Objective | None:
        """Return the first incomplete objective for a mission."""
        objectives = await self._repo.list_objectives(mission_id)
        for obj in objectives:
            if obj.status in (ObjectiveStatus.PENDING, ObjectiveStatus.IN_PROGRESS):
                return obj
        return None

    # ------------------------------------------------------------------
    # Goal lifecycle (one per coordinator run)
    # ------------------------------------------------------------------

    async def create_goal(
        self,
        objective_id: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> GoalNode:
        """Create a new goal within an objective."""
        goal = GoalNode(
            objective_id=objective_id,
            description=description,
            metadata=metadata or {},
        )
        result = await self._repo.create_goal(goal)

        # Mark the objective as in-progress if it was pending.
        await self._repo.update_objective(
            objective_id,
            status=ObjectiveStatus.IN_PROGRESS.value,
        )

        logger.info(
            "goal.goal_created",
            goal_id=result.id,
            objective_id=objective_id,
        )
        return result

    async def start_or_resume_goal(
        self,
        objective_id: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> GoalNode:
        """Find the last incomplete goal for an objective, or create a new one."""
        goals = await self._repo.list_goals(objective_id, limit=5)
        for g in goals:
            if g.status in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS):
                logger.info("goal.resumed", goal_id=g.id, description=g.description)
                return g
        return await self.create_goal(objective_id, description, metadata)

    async def complete_goal(
        self,
        goal_id: str,
        result_summary: str = "",
        plan_id: str | None = None,
    ) -> None:
        """Mark a goal as completed."""
        updates: dict[str, Any] = {
            "status": GoalStatus.COMPLETED.value,
            "result_summary": result_summary,
            "completed_at": datetime.now(timezone.utc),
        }
        if plan_id is not None:
            updates["plan_id"] = plan_id
        await self._repo.update_goal(goal_id, **updates)
        logger.info("goal.goal_completed", goal_id=goal_id)

    async def fail_goal(
        self,
        goal_id: str,
        error: str,
    ) -> None:
        """Mark a goal as failed."""
        await self._repo.update_goal(
            goal_id,
            status=GoalStatus.FAILED.value,
            error=error,
        )
        logger.info("goal.goal_failed", goal_id=goal_id, error=error[:200])

    async def get_goal(self, goal_id: str) -> GoalNode | None:
        return await self._repo.get_goal(goal_id)

    # ------------------------------------------------------------------
    # Action logging
    # ------------------------------------------------------------------

    async def record_action(
        self,
        goal_id: str,
        task_id: str = "",
        tool_name: str | None = None,
        description: str = "",
        input_summary: str = "",
        output_summary: str = "",
        status: str = "completed",
        latency_ms: float = 0.0,
        reflection_confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Action:
        action = Action(
            goal_id=goal_id,
            task_id=task_id,
            tool_name=tool_name,
            description=description,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status,
            latency_ms=latency_ms,
            reflection_confidence=reflection_confidence,
            metadata=metadata or {},
        )
        return await self._repo.record_action(action)

    async def list_actions(self, goal_id: str, limit: int = 100) -> list[Action]:
        return await self._repo.list_actions(goal_id, limit=limit)
