from __future__ import annotations

from datetime import datetime

from app.agents.models.pattern import ExecutionPattern
from app.database.models.execution_pattern import ExecutionPatternModel


def pattern_to_model(domain: ExecutionPattern) -> ExecutionPatternModel:
    return ExecutionPatternModel(
        id=domain.id,
        goal_pattern=domain.goal_pattern,
        capability=domain.capability,
        strategy_summary=domain.strategy_summary,
        plan_template=domain.plan_template,
        preferred_provider=domain.preferred_provider,
        preferred_model=domain.preferred_model,
        tool_sequence=",".join(domain.tool_sequence) if domain.tool_sequence else None,
        tags=",".join(domain.tags),
        success_count=domain.success_count,
        total_count=domain.total_count,
        avg_confidence=domain.avg_confidence,
        avg_importance=domain.avg_importance,
        avg_execution_cost_ms=domain.avg_execution_cost_ms,
        last_reflection_confidence=domain.last_reflection_confidence,
        last_success_at=domain.last_success_at,
        created_at=domain.created_at or datetime.now(),
        updated_at=domain.updated_at or datetime.now(),
    )


def pattern_from_model(model: ExecutionPatternModel) -> ExecutionPattern:
    tags = [t.strip() for t in (model.tags or "").split(",") if t.strip()]
    return ExecutionPattern(
        id=model.id,
        goal_pattern=model.goal_pattern,
        capability=model.capability,
        strategy_summary=model.strategy_summary or "",
        plan_template=model.plan_template,
        preferred_provider=model.preferred_provider,
        preferred_model=model.preferred_model,
        tool_sequence=[t.strip() for t in (model.tool_sequence or "").split(",") if t.strip()],
        tags=tags,
        success_count=model.success_count or 1,
        total_count=model.total_count or 1,
        avg_confidence=model.avg_confidence or 0.0,
        avg_importance=model.avg_importance or 0.0,
        avg_execution_cost_ms=model.avg_execution_cost_ms or 0.0,
        last_reflection_confidence=model.last_reflection_confidence or 0.0,
        last_success_at=model.last_success_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
