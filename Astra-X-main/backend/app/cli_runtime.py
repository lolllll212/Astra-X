"""CLI runtime wiring: tool registry, LLM router, and coordinator wiring.

This module mirrors the composition root in ``app.core.lifecycle`` but is
kept lightweight so the ``astra`` CLI can run without booting the full
FastAPI application. It builds:

* A :class:`ToolRegistry` with every built-in tool registered.
* A :class:`ToolExecutor` bound to that registry.
* An :class:`LLMRouter` with multiple providers registered
  (LM Studio, NVIDIA NIM, OpenRouter, Ollama, OpenAI Compatible).
* A :class:`ChatCoordinator` used by the interactive REPL and one-shot
  ``astra run`` / ``astra chat`` commands.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.domain.conversation import Conversation
from app.domain.enums import ConversationStatus, MessageRole, ModelCapability, ProviderType
from app.domain.message import Message, TextBlock
from app.domain.provider import ProviderID, ProviderSpec
from app.llm.factory import create_provider_adapter
from app.llm.router import LLMRouter
from app.tools.context import ToolContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry

__all__ = [
    "AstraRuntime",
    "build_runtime",
    "new_conversation",
    "user_message",
]


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register every built-in tool that can be imported successfully.

    Tools that fail to import (e.g. because of a missing optional
    dependency or external binary) are skipped with a warning rather than
    crashing startup.
    """
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    from app.tools.autofix import AutoFixTool
    from app.tools.browserskill import BrowserSkillTool
    from app.tools.builtin import (
        CalculatorTool,
        DateTimeTool,
        JsonTool,
        TextTool,
        UuidTool,
    )
    from app.tools.capabilities import CapabilityRegistry  # noqa: F401
    from app.tools.colibri import ColibriTool
    from app.tools.ecc import ECCTool
    from app.tools.filesystem import (
        ListDirectoryTool,
        ReadFileTool,
        SearchFilesTool,
        WriteFileTool,
    )
    from app.tools.ghidra import GhidraTool
    from app.tools.github import (
        GitHubCommitsTool,
        GitHubIssuesTool,
        GitHubPullRequestsTool,
        GitHubRepositoryTool,
    )
    from app.tools.lmstudio import LmStudioTool
    from app.tools.n8n import N8NTool
    from app.tools.voicebox import VoiceboxTool
    from app.tools.web import (
        DownloadTool,
        FetchTool,
        ScrapeTool,
        SearchTool,
    )

    tool_classes = (
        CalculatorTool,
        DateTimeTool,
        JsonTool,
        TextTool,
        UuidTool,
        SearchTool,
        FetchTool,
        ScrapeTool,
        DownloadTool,
        ReadFileTool,
        WriteFileTool,
        ListDirectoryTool,
        SearchFilesTool,
        GitHubRepositoryTool,
        GitHubIssuesTool,
        GitHubPullRequestsTool,
        GitHubCommitsTool,
        BrowserSkillTool,
        VoiceboxTool,
        ECCTool,
        AutoFixTool,
        ColibriTool,
        GhidraTool,
        LmStudioTool,
        N8NTool,
    )

    for tool_cls in tool_classes:
        try:
            registry.register(tool_cls())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "cli.tool_register_failed",
                tool=tool_cls.__name__,
                error=str(exc),
            )


def _lmstudio_provider_spec() -> ProviderSpec:
    """Build a ProviderSpec for the LM Studio local server."""
    from app.domain.enums import ModelCapability, ProviderType
    from app.domain.provider import ProviderID

    settings = get_settings()
    return ProviderSpec(
        id=ProviderID("lm_studio"),
        provider_type=ProviderType.LM_STUDIO,
        display_name="LM Studio",
        base_url=settings.lm_studio_base_url,
        models=[],
        supported_capabilities=frozenset({ModelCapability.CHAT}),
    )


class AstraRuntime:
    """A self-contained runtime for the ``astra`` CLI.

    Attributes:
        registry: The registered tool registry.
        executor: The tool executor bound to :attr:`registry`.
        router: The LLM router with registered provider(s).
        provider_id: The default provider identifier.
    """

    def __init__(
        self,
        *,
        provider_id: str = "lm_studio",
        provider_spec: ProviderSpec | None = None,
    ) -> None:
        from app.core.logging import get_logger

        self._logger = get_logger(__name__)
        self._cached_models: dict[str, list[str]] = {}
        self._known_models: dict[str, list[str]] = {}
        settings = get_settings()

        self.registry = ToolRegistry()
        _register_builtin_tools(self.registry)
        self.executor = ToolExecutor(self.registry)

        self.router = LLMRouter(
            settings=settings,
            default_provider_id=provider_id,
            max_retries=1,
        )

        # Register multiple providers
        self._register_providers(settings)

        self.provider_id = provider_id
        self._settings = settings

    def _register_providers(self, settings) -> None:
        """Register all available providers."""
        providers_to_register = [
            ("lm_studio", ProviderType.LM_STUDIO, settings.lm_studio_base_url, "local-model"),
            ("nvidia_nim", ProviderType.NVIDIA_NIM, settings.nvidia_nim_base_url, "nvidia/nemotron-3-ultra"),
            ("openrouter", ProviderType.OPENROUTER, settings.openrouter_base_url, "meta-llama/llama-3.1-8b-instruct:free"),
            ("ollama", ProviderType.OLLAMA, settings.ollama_base_url, None),
            ("openai_compatible", ProviderType.OPENAI_COMPATIBLE, settings.openai_compatible_base_url, None),
        ]

        for provider_id, provider_type, base_url, default_model in providers_to_register:
            if not base_url:
                continue
            try:
                spec = ProviderSpec(
                    id=ProviderID(provider_id),
                    provider_type=provider_type,
                    display_name=provider_id.replace("_", " ").title(),
                    base_url=base_url,
                    models=[],
                    supported_capabilities=frozenset({ModelCapability.CHAT}),
                )
                self.router.register_adapter(spec)
            except Exception as exc:
                self._logger.warning(
                    "cli.provider_register_failed",
                    provider=provider_id,
                    error=str(exc),
                )

    @property
    def default_model(self) -> str:
        model = self._settings.default_llm_model
        if model and model != "local-model":
            return model
        return self._preferred_model

    @property
    def _preferred_model(self) -> str:
        """A model id known to be served by any provider, if resolvable."""
        # Try to find a preferred model from any provider
        for provider_id in self._cached_models:
            models = self._cached_models.get(provider_id, []) or self._known_models.get(provider_id, [])
            for preferred in (
                "qwen/qwen3.5-9b",
                "deepseek-r1-distill-qwen-14b",
                "deepseek-r1-distill-qwen-7b",
                "qwen2.5-coder-7b-instruct",
                "mistral-7b-instruct-v0.2",
                "nvidia/nemotron-3-ultra",
                "meta-llama/llama-3.1-8b-instruct:free",
            ):
                if preferred in models:
                    return preferred
            if models:
                return sorted(models)[0]
        return ""

    @property
    def model_hint(self) -> str:
        # Return the current provider as hint
        return self.provider_id

    def tool_context(self, *, conversation_id: str = "") -> ToolContext:
        return ToolContext(conversation_id=conversation_id)

    async def list_lmstudio_models(self) -> list[str]:
        """Return model IDs served by LM Studio (empty list if unreachable)."""
        return await self.list_provider_models("lm_studio")

    async def list_provider_models(self, provider_id: str) -> list[str]:
        """Return model IDs for a specific provider."""
        if provider_id in self._cached_models:
            return self._cached_models[provider_id]
        if provider_id not in self._known_models:
            self._known_models[provider_id] = await self._discover_provider_models(provider_id)
        self._cached_models[provider_id] = self._known_models[provider_id]
        return self._cached_models[provider_id]

    async def _discover_provider_models(self, provider_id: str) -> list[str]:
        """Discover models from a specific provider."""
        provider = self.router.get_provider(provider_id)
        if not provider:
            return []
        try:
            return await provider.list_models()
        except Exception:
            return []

    def _discover_lmstudio_models(self) -> list[str]:
        """Synchronously query LM Studio for available models.

        Kept blocking (rather than async) so model resolution works from
        both sync and async callers without nesting event loops.
        """
        import json
        from urllib.request import Request, urlopen

        base_url = self._settings.lm_studio_base_url.rstrip("/")
        try:
            with urlopen(Request(base_url + "/models"), timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []

    async def close(self) -> None:
        for provider in self.router.list_providers():
            try:
                await provider.close()
            except Exception:  # pragma: no cover - defensive
                pass


def build_runtime(*, provider_id: str = "lm_studio") -> AstraRuntime:
    """Convenience factory for :class:`AstraRuntime`."""
    return AstraRuntime(provider_id=provider_id)


def new_conversation(
    *,
    model: str | None = None,
    provider: str | None = None,
    title: str | None = None,
) -> Conversation:
    """Create a lightweight in-memory conversation for CLI use."""
    from uuid import uuid4

    from app.domain.conversation import ConversationMetadata

    conversation = Conversation(
        id=str(uuid4()),
        title=title,
        status=ConversationStatus.ACTIVE,
        metadata=ConversationMetadata(
            model=model,
            provider=provider,
            title=title,
        ),
    )
    return conversation


def user_message(conversation_id: str, text: str) -> Message:
    """Create a user Message containing a single text block."""
    from datetime import UTC, datetime
    from uuid import uuid4

    return Message(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


def system_message(conversation_id: str, text: str) -> Message:
    """Create a system Message containing a single text block."""
    from datetime import UTC, datetime
    from uuid import uuid4

    return Message(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role=MessageRole.SYSTEM,
        content=[TextBlock(text=text)],
        created_at=datetime.now(UTC),
    )


__all__ = [
    "AstraRuntime",
    "build_runtime",
    "new_conversation",
    "system_message",
    "user_message",
]