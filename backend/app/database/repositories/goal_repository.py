"""Repository for the goal hierarchy — Mission, Objective, Goal, Action."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.models.goal import (
    Action,
    GoalNode,
    GoalStatus,
    Mission,
    MissionStatus,
    Objective,
    ObjectiveStatus,
    Priority,
)
from app.database.models.goal import ActionModel, GoalModel, MissionModel, ObjectiveModel
from app.database.repositories.base import BaseRepository


class GoalRepository:
    """CRUD for the goal hierarchy.

     Operations are intentionally flat (per-level) rather than nested
     to keep the API simple and testable.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Mission
    # ------------------------------------------------------------------

    async def create_mission(self, mission: Mission) -> Mission:
        model = MissionModel(
            id=mission.id,
            title=mission.title,
            description=mission.description,
            status=mission.status.value,
            metadata_=mission.metadata,
            created_at=mission.created_at,
            updated_at=mission.updated_at,
        )
        self._session.add(model)
        await self._session.flush()
        return mission

    async def get_mission(self, mission_id: str) -> Mission | None:
        model = await self._session.get(MissionModel, mission_id)
        return self._mission_to_domain(model) if model else None

    async def list_missions(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> list[Mission]:
        stmt = select(MissionModel)
        if status:
            stmt = stmt.where(MissionModel.status == status)
        stmt = stmt.order_by(MissionModel.updated_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return [self._mission_to_domain(m) for m in result.scalars().all()]

    async def update_mission(
        self,
        mission_id: str,
        **updates: Any,
    ) -> Mission | None:
        model = await self._session.get(MissionModel, mission_id)
        if model is None:
            return None
        for key, val in updates.items():
            if key == "metadata_":
                model.metadata_ = val
            elif hasattr(model, key):
                setattr(model, key, val)
        model.updated_at = datetime.now(UTC)
        await self._session.flush()
        return self._mission_to_domain(model)

    async def delete_mission(self, mission_id: str) -> bool:
        model = await self._session.get(MissionModel, mission_id)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    async def create_objective(self, obj: Objective) -> Objective:
        model = ObjectiveModel(
            id=obj.id,
            mission_id=obj.mission_id,
            title=obj.title,
            description=obj.description,
            status=obj.status.value,
            order=obj.order,
            metadata_=obj.metadata,
            created_at=obj.created_at,
        )
        self._session.add(model)
        await self._session.flush()
        return obj

    async def get_objective(self, objective_id: str) -> Objective | None:
        model = await self._session.get(ObjectiveModel, objective_id)
        return self._objective_to_domain(model) if model else None

    async def list_objectives(
        self,
        mission_id: str,
        limit: int = 50,
    ) -> list[Objective]:
        stmt = (
            select(ObjectiveModel)
            .where(ObjectiveModel.mission_id == mission_id)
            .order_by(ObjectiveModel.order)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [self._objective_to_domain(m) for m in result.scalars().all()]

    async def update_objective(
        self,
        objective_id: str,
        **updates: Any,
    ) -> Objective | None:
        model = await self._session.get(ObjectiveModel, objective_id)
        if model is None:
            return None
        for key, val in updates.items():
            if hasattr(model, key):
                setattr(model, key, val)
        await self._session.flush()
        return self._objective_to_domain(model)

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    async def create_goal(self, goal: GoalNode) -> GoalNode:
        model = GoalModel(
            id=goal.id,
            objective_id=goal.objective_id,
            description=goal.description,
            status=goal.status.value,
            priority=goal.priority.value,
            urgency=goal.urgency,
            estimated_cost_ms=goal.estimated_cost_ms,
            success_probability=goal.success_probability,
            dependencies=goal.dependencies or None,
            plan_id=goal.plan_id,
            result_summary=goal.result_summary,
            error=goal.error,
            metadata_=goal.metadata,
            created_at=goal.created_at,
            completed_at=goal.completed_at,
        )
        self._session.add(model)
        await self._session.flush()
        return goal

    async def get_goal(self, goal_id: str) -> GoalNode | None:
        model = await self._session.get(GoalModel, goal_id)
        return self._goal_to_domain(model) if model else None

    async def list_goals(
        self,
        objective_id: str,
        limit: int = 50,
    ) -> list[GoalNode]:
        stmt = (
            select(GoalModel)
            .where(GoalModel.objective_id == objective_id)
            .order_by(GoalModel.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [self._goal_to_domain(m) for m in result.scalars().all()]

    async def list_goals_by_priority(
        self,
        objective_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[GoalNode]:
        """Return goals ordered by priority (critical first), then by urgency."""
        order_col = GoalModel.priority.desc()
        stmt = select(GoalModel).where(GoalModel.objective_id == objective_id)
        if status:
            stmt = stmt.where(GoalModel.status == status)
        stmt = stmt.order_by(order_col, GoalModel.urgency.asc().nullslast()).limit(limit)
        result = await self._session.execute(stmt)
        return [self._goal_to_domain(m) for m in result.scalars().all()]

    async def list_ready_goals(
        self,
        objective_id: str,
        limit: int = 50,
    ) -> list[GoalNode]:
        """Return pending/in-progress goals whose dependencies are met."""
        stmt = (
            select(GoalModel)
            .where(GoalModel.objective_id == objective_id)
            .where(GoalModel.status.in_(["pending", "in_progress"]))
            .order_by(GoalModel.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        goals = [self._goal_to_domain(m) for m in result.scalars().all()]

        # Fetch completed goal IDs for dependency checking.
        completed = await self._session.execute(
            select(GoalModel.id).where(
                GoalModel.objective_id == objective_id,
                GoalModel.status == "completed",
            )
        )
        completed_ids = {row[0] for row in completed.all()}

        ready = []
        for g in goals:
            if g.dependencies and not completed_ids.issuperset(g.dependencies):
                continue
            ready.append(g)
        return ready

    async def update_goal(
        self,
        goal_id: str,
        **updates: Any,
    ) -> GoalNode | None:
        model = await self._session.get(GoalModel, goal_id)
        if model is None:
            return None
        for key, val in updates.items():
            if hasattr(model, key):
                setattr(model, key, val)
        await self._session.flush()
        return self._goal_to_domain(model)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    async def record_action(self, action: Action) -> Action:
        model = ActionModel(
            id=action.id,
            goal_id=action.goal_id,
            task_id=action.task_id,
            tool_name=action.tool_name,
            description=action.description,
            input_summary=action.input_summary,
            output_summary=action.output_summary,
            status=action.status,
            latency_ms=action.latency_ms,
            reflection_confidence=action.reflection_confidence,
            metadata_=action.metadata,
            timestamp=action.timestamp,
        )
        self._session.add(model)
        await self._session.flush()
        return action

    async def list_actions(
        self,
        goal_id: str,
        limit: int = 100,
    ) -> list[Action]:
        stmt = (
            select(ActionModel)
            .where(ActionModel.goal_id == goal_id)
            .order_by(ActionModel.timestamp)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [self._action_to_domain(m) for m in result.scalars().all()]

    # ------------------------------------------------------------------
    # Converters
    # ------------------------------------------------------------------

    @staticmethod
    def _mission_to_domain(model: MissionModel) -> Mission:
        return Mission(
            id=model.id,
            title=model.title,
            description=model.description,
            status=MissionStatus(model.status),
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata_ or {},
        )

    @staticmethod
    def _objective_to_domain(model: ObjectiveModel) -> Objective:
        return Objective(
            id=model.id,
            mission_id=model.mission_id,
            title=model.title,
            description=model.description,
            status=ObjectiveStatus(model.status),
            order=model.order,
            created_at=model.created_at,
            metadata=model.metadata_ or {},
        )

    @staticmethod
    def _goal_to_domain(model: GoalModel) -> GoalNode:
        return GoalNode(
            id=model.id,
            objective_id=model.objective_id,
            description=model.description,
            status=GoalStatus(model.status),
            priority=Priority(model.priority) if model.priority else Priority.MEDIUM,
            urgency=model.urgency,
            estimated_cost_ms=model.estimated_cost_ms or 0.0,
            success_probability=model.success_probability or 0.0,
            dependencies=list(model.dependencies) if model.dependencies else [],
            plan_id=model.plan_id,
            result_summary=model.result_summary or "",
            error=model.error,
            created_at=model.created_at,
            completed_at=model.completed_at,
            metadata=model.metadata_ or {},
        )

    @staticmethod
    def _action_to_domain(model: ActionModel) -> Action:
        return Action(
            id=model.id,
            goal_id=model.goal_id,
            task_id=model.task_id,
            tool_name=model.tool_name,
            description=model.description,
            input_summary=model.input_summary,
            output_summary=model.output_summary,
            status=model.status,
            latency_ms=model.latency_ms,
            reflection_confidence=model.reflection_confidence,
            metadata=model.metadata_ or {},
            timestamp=model.timestamp,
        )
