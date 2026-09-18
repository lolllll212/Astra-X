"""Tool management service.

Handles registration, discovery, and invocation of tools available to
the LLM.
"""

from __future__ import annotations

from app.core.exceptions import ResourceNotFoundError
from app.core.logging import get_logger
from app.database.repositories.tool_repository import ToolRepository
from app.domain.enums import ToolType
from app.domain.tool import ToolParameter, ToolSpec

logger = get_logger(__name__)


class ToolService:
    """Manages tool specifications and discovery."""

    def __init__(self, repository: ToolRepository) -> None:
        self._repo = repository

    async def register(
        self,
        name: str,
        description: str,
        tool_type: ToolType = ToolType.FUNCTION,
        parameters: list[ToolParameter] | None = None,
    ) -> ToolSpec:
        """Register a new tool.

        Args:
            name: Unique tool name.
            description: Human-readable description.
            tool_type: Category of tool.
            parameters: Accepted parameters.

        Returns:
            The registered tool spec.
        """
        spec = ToolSpec(
            name=name,
            description=description,
            tool_type=tool_type,
            parameters=parameters or [],
        )
        result = await self._repo.add(spec)
        logger.info("tool.registered", tool_name=name)
        return result

    async def get(self, name: str) -> ToolSpec:
        """Retrieve a tool by name.

        Args:
            name: The tool name.
        """
        spec = await self._repo.get(name)
        if spec is None:
            raise ResourceNotFoundError(
                message=f"Tool '{name}' not found.",
            )
        return spec

    async def list_available(self) -> list[ToolSpec]:
        """List all available tools.

        Returns:
            All registered tool specs.
        """
        return await self._repo.list_all(is_enabled=True)

    async def remove(self, name: str) -> None:
        """Remove a tool registration.

        Args:
            name: The tool name.
        """
        deleted = await self._repo.delete(name)
        if not deleted:
            raise ResourceNotFoundError(
                message=f"Tool '{name}' not found.",
            )
        logger.info("tool.removed", tool_name=name)
