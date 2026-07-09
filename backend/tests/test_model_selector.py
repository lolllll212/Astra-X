"""Unit tests for the adaptive model selector."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.llm.base import LLMProvider
from app.llm.model_selector import (
    DEFAULT_MODEL_PROFILES,
    CapabilityProfile,
    ModelProfile,
    ModelSelector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    provider_id: str,
    model_id: str,
    *,
    is_available: bool = True,
) -> MagicMock:
    provider = MagicMock(spec=LLMProvider)
    provider.provider_id = provider_id
    provider.model_id = model_id
    return provider


def _make_router(
    providers: list[MagicMock] | None = None,
    available: dict[str, bool] | None = None,
) -> MagicMock:
    router = MagicMock()
    router.list_providers.return_value = providers or []
    available = available or {}

    def is_available(pid: str) -> bool:
        return available.get(pid, True)

    router.is_provider_available.side_effect = is_available
    return router


# ---------------------------------------------------------------------------
# CapabilityProfile
# ---------------------------------------------------------------------------


class TestCapabilityProfile:
    def test_default_profile(self) -> None:
        p = CapabilityProfile()
        assert p.requires_coding is False
        assert p.reasoning == "none"
        assert p.prefers_speed is False
        assert p.prefers_large_context is False
        assert p.requires_vision is False

    def test_coding_profile(self) -> None:
        p = CapabilityProfile(requires_coding=True, reasoning="medium")
        assert p.requires_coding is True
        assert p.reasoning == "medium"


# ---------------------------------------------------------------------------
# ModelProfile
# ---------------------------------------------------------------------------


class TestModelProfile:
    def test_default_profile(self) -> None:
        mp = ModelProfile(model_id="test-model")
        assert mp.model_id == "test-model"
        assert mp.strengths == []
        assert mp.context_length == 4096

    def test_custom_profile(self) -> None:
        mp = ModelProfile(
            model_id="deepseek-coder",
            strengths=["coding", "reasoning"],
            context_length=32768,
        )
        assert "coding" in mp.strengths
        assert mp.context_length == 32768


# ---------------------------------------------------------------------------
# ModelSelector — scoring
# ---------------------------------------------------------------------------


class TestModelSelectorScoring:
    def test_select_coding_model(self) -> None:
        """Coding profile should prefer a model with coding strength."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "deepseek-coder"),
                _make_provider("ollama", "llama3.1"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(requires_coding=True, reasoning="medium"),
        )
        assert candidates[0][0] == "deepseek-coder"

    def test_select_reasoning_model(self) -> None:
        """Deep reasoning should prefer a reasoning model."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "qwen2.5"),
                _make_provider("ollama", "phi3"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(reasoning="deep"),
        )
        assert candidates[0][0] == "qwen2.5"

    def test_select_fast_model(self) -> None:
        """Speed preference should prefer a fast model."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "phi3"),
                _make_provider("ollama", "deepseek-r1"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(prefers_speed=True),
        )
        assert candidates[0][0] == "phi3"

    def test_select_large_context_model(self) -> None:
        """Large context preference should prefer mistral-nemo."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "mistral-nemo"),
                _make_provider("ollama", "phi3"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(prefers_large_context=True),
        )
        assert candidates[0][0] == "mistral-nemo"

    def test_vision_required(self) -> None:
        """Vision tasks should prefer a model with vision capability."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "llava"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(requires_vision=True),
        )
        assert candidates[0][0] == "llava"

    def test_vision_not_supported_penalises(self) -> None:
        """Vision tasks should not match non-vision models."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "llama3.1"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(requires_vision=True),
        )
        assert len(candidates) == 1
        assert candidates[0][0] == "llama3.1"

    def test_unknown_model_gets_low_score(self) -> None:
        """Unknown models should receive a very low score."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "unknown-model-xyz"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(requires_coding=True),
        )
        assert len(candidates) == 1
        assert candidates[0][0] == "unknown-model-xyz"

    def test_no_registered_providers_returns_empty(self) -> None:
        """No providers should result in an empty candidate list."""
        router = _make_router(providers=[], available={})
        selector = ModelSelector(router=router)
        assert selector._score_providers(CapabilityProfile()) == []


# ---------------------------------------------------------------------------
# ModelSelector — provider filtering
# ---------------------------------------------------------------------------


class TestModelSelectorProviderFiltering:
    def test_unavailable_provider_excluded(self) -> None:
        """Providers with open circuits should be excluded."""
        router = _make_router(
            providers=[
                _make_provider("ollama", "llama3.1"),
                _make_provider("lm_studio", "deepseek-coder"),
            ],
            available={"ollama": False, "lm_studio": True},
        )
        selector = ModelSelector(router=router)
        candidates = selector._score_providers(
            CapabilityProfile(requires_coding=True),
        )
        # deepseek-coder should be the top result (only lm_studio is available)
        assert candidates[0][0] == "deepseek-coder"
        assert candidates[0][1] == "lm_studio"


# ---------------------------------------------------------------------------
# ModelSelector — select()
# ---------------------------------------------------------------------------


class TestModelSelectorSelect:
    async def test_select_returns_top_candidate(self) -> None:
        router = _make_router(
            providers=[
                _make_provider("ollama", "deepseek-coder"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        model, provider = await selector.select(
            CapabilityProfile(requires_coding=True),
        )
        assert model == "deepseek-coder"
        assert provider == "ollama"

    async def test_select_fallback_when_no_providers(self) -> None:
        router = _make_router(providers=[], available={})
        selector = ModelSelector(
            router=router,
            default_model="fallback-model",
            default_provider="fallback-provider",
        )
        model, provider = await selector.select(CapabilityProfile())
        assert model == "fallback-model"
        assert provider == "fallback-provider"

    async def test_select_for_planning_uses_reasoning(self) -> None:
        router = _make_router(
            providers=[
                _make_provider("ollama", "qwen2.5"),
                _make_provider("ollama", "phi3"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        model, provider = await selector.select_for_planning()
        assert model == "qwen2.5"

    async def test_select_for_reflection_uses_fast(self) -> None:
        router = _make_router(
            providers=[
                _make_provider("ollama", "phi3"),
                _make_provider("ollama", "deepseek-r1"),
            ],
            available={"ollama": True},
        )
        selector = ModelSelector(router=router)
        model, provider = await selector.select_for_reflection()
        assert model == "phi3"


# ---------------------------------------------------------------------------
# ModelSelector — default model profiles
# ---------------------------------------------------------------------------


class TestDefaultModelProfiles:
    def test_all_defaults_have_required_keys(self) -> None:
        for model_id, attrs in DEFAULT_MODEL_PROFILES.items():
            assert "strengths" in attrs, f"{model_id} missing 'strengths'"
            assert "context_length" in attrs, f"{model_id} missing 'context_length'"
            assert isinstance(attrs["strengths"], list), f"{model_id} strengths not a list"
            assert isinstance(attrs["context_length"], int), f"{model_id} context_length not int"

    def test_default_profiles_are_reasonable(self) -> None:
        """Spot-check known models."""
        assert "coding" in DEFAULT_MODEL_PROFILES["deepseek-coder"]["strengths"]
        assert "reasoning" in DEFAULT_MODEL_PROFILES["deepseek-r1"]["strengths"]
        assert "vision" in DEFAULT_MODEL_PROFILES["llava"]["strengths"]
        assert DEFAULT_MODEL_PROFILES["mistral-nemo"]["context_length"] == 128000


# ---------------------------------------------------------------------------
# Build profiles from config
# ---------------------------------------------------------------------------


class TestBuildProfiles:
    def test_build_from_raw(self) -> None:
        raw = {
            "my-model": {
                "strengths": ["coding", "fast"],
                "context_length": 16384,
            },
        }
        profiles = ModelSelector._build_profiles(raw)
        assert "my-model" in profiles
        mp = profiles["my-model"]
        assert "coding" in mp.strengths
        assert "fast" in mp.strengths
        assert mp.context_length == 16384

    def test_build_empty(self) -> None:
        profiles = ModelSelector._build_profiles({})
        assert profiles == {}

    def test_build_normalizes_strengths(self) -> None:
        raw = {
            "test": {
                "strengths": ["  Coding ", "FAST"],
                "context_length": 8192,
            },
        }
        profiles = ModelSelector._build_profiles(raw)
        mp = profiles["test"]
        assert "coding" in mp.strengths
        assert "fast" in mp.strengths

    def test_build_missing_context_length_defaults(self) -> None:
        raw = {
            "test": {
                "strengths": ["balanced"],
            },
        }
        profiles = ModelSelector._build_profiles(raw)
        assert profiles["test"].context_length == 4096
