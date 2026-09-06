"""Tool domain-to-ORM mapper."""

from __future__ import annotations

from typing import Any

from app.database.models.tool import ToolModel
from app.domain.enums import ToolType
from app.domain.tool import ToolParameter, ToolSpec

__all__ = [
    "tool_from_model",
    "tool_to_model",
]


def _parameter_to_dict(param: ToolParameter) -> dict[str, Any]:
    return param.model_dump(mode="json", by_alias=True, exclude_none=True)


def _parameter_from_dict(data: dict[str, Any]) -> ToolParameter:
    return ToolParameter.model_validate(data)


def tool_to_model(domain: ToolSpec) -> ToolModel:
    params_data: list[dict[str, Any]] = [_parameter_to_dict(p) for p in domain.parameters]
    return ToolModel(
        name=domain.name,
        description=domain.description,
        tool_type=domain.tool_type.value,
        parameters=params_data,
    )


def tool_from_model(model: ToolModel) -> ToolSpec:
    params: list[ToolParameter] = [_parameter_from_dict(p) for p in (model.parameters or [])]
    return ToolSpec(
        name=model.name,
        description=model.description,
        tool_type=ToolType(model.tool_type),
        parameters=params,
    )
