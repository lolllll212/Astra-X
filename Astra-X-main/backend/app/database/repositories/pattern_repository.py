from __future__ import annotations

import re
from typing import Any

from sqlalchemy import or_, select

from app.agents.models.pattern import ExecutionPattern
from app.database.converters.pattern_mapper import pattern_from_model, pattern_to_model
from app.database.models.execution_pattern import ExecutionPatternModel
from app.database.repositories.base import BaseRepository


class PatternRepository(BaseRepository[ExecutionPattern, ExecutionPatternModel]):
    @property
    def _model_cls(self) -> type[ExecutionPatternModel]:
        return ExecutionPatternModel

    def _to_domain(self, model: ExecutionPatternModel) -> ExecutionPattern:
        return pattern_from_model(model)

    def _to_model(self, domain: ExecutionPattern) -> ExecutionPatternModel:
        return pattern_to_model(domain)

    async def find_by_goal_pattern(self, goal_pattern: str) -> ExecutionPattern | None:
        stmt = select(self._model_cls).where(
            self._model_cls.goal_pattern == goal_pattern
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_domain(model)

    async def search_by_tags(self, keywords: list[str], limit: int = 10) -> list[ExecutionPattern]:
        if not keywords:
            return []
        conditions: list[Any] = []
        for kw in keywords:
            escaped = re.escape(kw)
            conditions.append(self._model_cls.tags.ilike(f"%{escaped}%"))
            conditions.append(self._model_cls.goal_pattern.ilike(f"%{escaped}%"))
        stmt = (
            select(self._model_cls)
            .where(or_(*conditions))
            .order_by(
                self._model_cls.success_count.desc(),
                self._model_cls.total_count.desc(),
            )
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_domain(m) for m in models]

    async def upsert(self, domain: ExecutionPattern) -> ExecutionPattern:
        existing = await self.find_by_goal_pattern(domain.goal_pattern)
        if existing is not None:
            merged = existing.model_copy(
                update=domain.model_dump(exclude={"id", "created_at"}),
            )
            return await self.update(merged)
        return await self.add(domain)

    async def list_all(self) -> list[ExecutionPattern]:
        models = await super().list_all()
        return models
