"""Example Provider Plugin — demonstrates provider registration lifecycle.

Registers a mock ``ProviderSpec`` with the ``ProviderService`` on load
and removes it on unload.
"""

from __future__ import annotations

from app.domain.enums import ProviderType
from app.plugins.base import Plugin, PluginContext


class ExampleProviderPlugin(Plugin):
    """Plugin that registers an example provider on load and removes on unload."""

    def __init__(self) -> None:
        self._provider_id: str | None = None
        self._context: PluginContext | None = None

    @property
    def name(self) -> str:
        return "example_provider"

    async def on_load(self, context: PluginContext) -> None:
        self._context = context
        if context.provider_service is None:
            return

        result = await context.provider_service.register(
            provider_id="example",
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            display_name="Example Plugin Provider",
            base_url="https://api.example.com/v1",
            models=["example-model-1", "example-model-2"],
            is_enabled=True,
        )
        self._provider_id = result.id

    async def on_unload(self) -> None:
        if self._provider_id is not None and self._context is not None:
            ps = self._context.provider_service
            if ps is not None:
                try:
                    await ps.remove(self._provider_id)
                except Exception:
                    pass
        self._provider_id = None
        self._context = None
