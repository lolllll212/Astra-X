"""Adaptive model selection based on task capability profiles.

The :class:`ModelSelector` matches a :class:`CapabilityProfile` (what a task
needs) to the best available provider/model (what the system has) by scoring
each registered provider against the profile and returning the highest-score
match.

Usage::

    selector = ModelSelector.from_router(router)
    model, provider = await selector.select(
        CapabilityProfile(requires_coding=True, reasoning="deep"),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.llm.router import LLMRouter

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model profiles — describe what each model is good at
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PROFILES: dict[str, dict[str, object]] = {
    "deepseek-coder": {"strengths": ["coding"], "context_length": 32768, "reasoning": True},
    "deepseek-r1": {"strengths": ["reasoning", "coding"], "context_length": 32768},
    "qwen2.5": {"strengths": ["reasoning"], "context_length": 32768},
    "qwen2.5-coder": {"strengths": ["coding", "reasoning"], "context_length": 32768},
    "llama3.1": {"strengths": ["balanced", "fast"], "context_length": 8192},
    "llama3.2": {"strengths": ["balanced", "fast"], "context_length": 8192},
    "mistral-nemo": {"strengths": ["large_context", "reasoning"], "context_length": 128000},
    "mistral-small": {"strengths": ["balanced", "fast"], "context_length": 32768},
    "codellama": {"strengths": ["coding"], "context_length": 16384},
    "phi3": {"strengths": ["fast", "small"], "context_length": 4096},
    "gemma2": {"strengths": ["balanced", "fast"], "context_length": 8192},
    "llava": {"strengths": ["vision"], "context_length": 4096, "vision": True},
}


@dataclass(frozen=True)
class CapabilityProfile:
    """Describes what a task needs from the model.

    Attributes:
        requires_coding: Task involves code generation or analysis.
        reasoning: Depth of reasoning needed (``"none"``, ``"low"``,
            ``"medium"``, ``"deep"``).
        prefers_speed: Task benefits from fast token generation (e.g.
            simple text transformation).
        prefers_large_context: Task needs a large context window.
        requires_vision: Task involves image understanding.
    """

    requires_coding: bool = False
    reasoning: str = "none"
    prefers_speed: bool = False
    prefers_large_context: bool = False
    requires_vision: bool = False


@dataclass
class ModelProfile:
    """Describes a model's capabilities.

    Attributes:
        model_id: The model identifier.
        strengths: List of strength labels (``"coding"``, ``"reasoning"``,
            ``"fast"``, ``"balanced"``, ``"large_context"``, ``"vision"``).
        context_length: Maximum context window in tokens.
    """

    model_id: str
    strengths: list[str] = field(default_factory=list)
    context_length: int = 4096


# ---------------------------------------------------------------------------
# ModelSelector
# ---------------------------------------------------------------------------


class ModelSelector:
    """Selects the best model/provider for a given capability profile.

    Usage::

        selector = ModelSelector(
            router=router,
            model_profiles={
                "deepseek-coder": {"strengths": ["coding"], "context_length": 32768},
            },
        )
        model_id, provider_id = await selector.select(
            CapabilityProfile(requires_coding=True),
        )
    """

    def __init__(
        self,
        router: LLMRouter,
        model_profiles: dict[str, dict[str, object]] | None = None,
        default_model: str = "llama3.1",
        default_provider: str | None = None,
    ) -> None:
        self._router = router
        self._model_profiles: dict[str, ModelProfile] = self._build_profiles(
            model_profiles or DEFAULT_MODEL_PROFILES,
        )
        self._default_model = default_model
        self._default_provider = default_provider

    @classmethod
    def from_router(
        cls,
        router: LLMRouter,
        default_model: str = "llama3.1",
        default_provider: str | None = None,
    ) -> ModelSelector:
        """Create a selector linked to a router with default model profiles.

        Args:
            router: The LLM router to query for registered providers.
            default_model: Fallback model when no provider matches.
            default_provider: Fallback provider when no match is found.

        Returns:
            A new ModelSelector instance.
        """
        return cls(
            router=router,
            model_profiles=None,
            default_model=default_model,
            default_provider=default_provider,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select(
        self,
        profile: CapabilityProfile,
    ) -> tuple[str, str | None]:
        """Select the best (model_id, provider_id) for a capability profile.

        Providers with open circuit breakers are excluded. If multiple
        providers serve the same model, the first healthy one wins.

        Args:
            profile: What the task needs from the model.

        Returns:
            A ``(model_id, provider_id)`` tuple. The provider may be
            ``None`` if no registered provider matches, in which case
            the default model is returned.
        """
        candidates = self._score_providers(profile)
        if candidates:
            model_id, provider_id = candidates[0]
            logger.debug(
                "model_selector.selected",
                model=model_id,
                provider=provider_id,
                profile=str(profile),
            )
            return model_id, provider_id

        logger.info(
            "model_selector.fallback",
            model=self._default_model,
            provider=self._default_provider,
        )
        return self._default_model, self._default_provider

    async def select_for_planning(self) -> tuple[str, str | None]:
        """Select a model optimised for planning (reasoning-heavy).

        Returns:
            A ``(model_id, provider_id)`` tuple.
        """
        return await self.select(
            CapabilityProfile(reasoning="deep"),
        )

    async def select_for_reflection(self) -> tuple[str, str | None]:
        """Select a model optimised for reflection (fast, balanced).

        Returns:
            A ``(model_id, provider_id)`` tuple.
        """
        return await self.select(
            CapabilityProfile(prefers_speed=True),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_providers(
        self,
        profile: CapabilityProfile,
    ) -> list[tuple[str, str | None]]:
        """Score all registered providers against a profile, sorted best first.

        Args:
            profile: The capability profile to match against.

        Returns:
            A list of ``(model_id, provider_id)`` tuples sorted by
            descending score.
        """
        scored: list[tuple[float, str, str | None]] = []
        seen_models: set[str] = set()

        for provider in self._router.list_providers():
            mid = provider.model_id
            pid = provider.provider_id

            if self._router.is_provider_available(pid):
                mp = self._model_profiles.get(mid)
                score = self._score(profile, mp, mid)
                scored.append((score, mid, pid))
                seen_models.add(mid)

        # Sort by score descending, then by model name for stability.
        scored.sort(key=lambda x: (-x[0], x[1]))

        return [(mid, pid) for _, mid, pid in scored]

    @staticmethod
    def _score(
        profile: CapabilityProfile,
        mp: ModelProfile | None,
        model_id: str,
    ) -> float:
        """Score a single model/profile combination.

        Args:
            profile: What the task needs.
            mp: The model's profile (``None`` if unknown).
            model_id: The model identifier (used for fallback scoring).

        Returns:
            A score where higher is better. Returns ``-float("inf")``
            for unknown models that don't satisfy basic constraints.
        """
        score = 0.0

        if mp is None:
            # Unknown model — low baseline, but still usable.
            return -10.0

        strengths = set(mp.strengths)

        if profile.requires_coding:
            score += 3.0 if "coding" in strengths else -1.0

        if profile.reasoning in ("medium", "deep"):
            score += 3.0 if "reasoning" in strengths else -1.0
        elif profile.reasoning == "low":
            score += 1.0 if "reasoning" in strengths else 0.5

        if profile.prefers_speed:
            score += 2.0 if "fast" in strengths else -1.0

        if profile.prefers_large_context:
            if "large_context" in strengths:
                score += 2.0
            else:
                score += min(mp.context_length / 128000.0, 1.0) * 1.5

        if profile.requires_vision:
            score += 3.0 if "vision" in strengths else -10.0

        # Balanced models get a small baseline for general tasks.
        if "balanced" in strengths:
            score += 0.5

        return score

    @staticmethod
    def _build_profiles(
        raw: dict[str, dict[str, object]],
    ) -> dict[str, ModelProfile]:
        """Convert raw config dicts into ModelProfile instances.

        Args:
            raw: Mapping of model_id → attributes.

        Returns:
            A mapping of model_id → ModelProfile.
        """
        profiles: dict[str, ModelProfile] = {}
        for model_id, attrs in raw.items():
            strengths = attrs.get("strengths", [])
            if isinstance(strengths, list):
                strengths = [s.lower().strip() for s in strengths if isinstance(s, str)]
            profiles[model_id] = ModelProfile(
                model_id=model_id,
                strengths=strengths,
                context_length=int(attrs.get("context_length", 4096)),
            )
        return profiles
