"""Domain models for Voicebox TTS/STT framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class TTSEngine(str, Enum):
    QWEN3_TTS = "qwen3_tts"
    QWEN_CUSTOMVOICE = "qwen_customvoice"
    LUX_TTS = "lux_tts"
    CHATTERBOX_MULTILINGUAL = "chatterbox_multilingual"
    CHATTERBOX_TURBO = "chatterbox_turbo"
    HUME_TADA = "hume_tada"
    KOKORO = "kokoro"


class STTEngine(str, Enum):
    WHISPER = "whisper"
    WHISPER_CPP = "whisper_cpp"
    FASTER_WHISPER = "faster_whisper"


class VoiceProfileType(str, Enum):
    CLONED = "cloned"
    PRESET = "preset"
    CUSTOM = "custom"


class DictationMode(str, Enum):
    PUSH_TO_TALK = "push_to_talk"
    TOGGLE = "toggle"
    CONTINUOUS = "continuous"


@dataclass
class VoiceProfile:
    id: str = field(default_factory=lambda: f"voice-{uuid4().hex[:8]}")
    name: str = ""
    type: VoiceProfileType = VoiceProfileType.PRESET
    engine: TTSEngine = TTSEngine.KOKORO
    language: str = "en"
    reference_audio_path: str | None = None
    reference_text: str | None = None
    persona: str | None = None
    preset_voice_id: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TTSRequest:
    id: str = field(default_factory=lambda: f"tts-{uuid4().hex[:8]}")
    text: str = ""
    voice_profile_id: str = ""
    engine: TTSEngine | None = None
    language: str | None = None
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    effects: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    chunk_size: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TTSResult:
    request_id: str
    success: bool
    audio_path: str | None = None
    audio_data: bytes | None = None
    duration_seconds: float = 0.0
    sample_rate: int = 48000
    channels: int = 1
    error: str | None = None
    engine_used: TTSEngine | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class STTRequest:
    id: str = field(default_factory=lambda: f"stt-{uuid4().hex[:8]}")
    audio_path: str | None = None
    audio_data: bytes | None = None
    engine: STTEngine = STTEngine.FASTER_WHISPER
    language: str | None = None
    task: str = "transcribe"
    vad_filter: bool = True
    word_timestamps: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class STTResult:
    request_id: str
    success: bool
    text: str = ""
    language: str | None = None
    language_probability: float = 0.0
    segments: list[dict[str, Any]] = field(default_factory=list)
    words: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None
    engine_used: STTEngine | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DictationSession:
    id: str = field(default_factory=lambda: f"dict-{uuid4().hex[:8]}")
    mode: DictationMode = DictationMode.PUSH_TO_TALK
    hotkey: str = "ctrl+shift+space"
    auto_paste: bool = True
    target_window: str | None = None
    stt_engine: STTEngine = STTEngine.FASTER_WHISPER
    language: str | None = None
    is_active: bool = False
    started_at: datetime | None = None
    transcriptions: list[STTResult] = field(default_factory=list)


@dataclass
class VoiceboxConfig:
    models_dir: str = ""
    cache_dir: str = ""
    default_tts_engine: TTSEngine = TTSEngine.KOKORO
    default_stt_engine: STTEngine = STTEngine.FASTER_WHISPER
    device: str = "auto"
    gpu_layers: int = 0
    max_concurrent_tts: int = 2
    max_concurrent_stt: int = 1
    enable_mcp: bool = True
    mcp_port: int = 8080
    api_port: int = 8081


@dataclass
class StoriesProject:
    id: str = field(default_factory=lambda: f"story-{uuid4().hex[:8]}")
    name: str = ""
    tracks: list[dict[str, Any]] = field(default_factory=list)
    total_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VoiceboxStatus:
    tts_engines_loaded: list[TTSEngine] = field(default_factory=list)
    stt_engines_loaded: list[STTEngine] = field(default_factory=list)
    voice_profiles: list[VoiceProfile] = field(default_factory=list)
    active_dictation: DictationSession | None = None
    mcp_server_running: bool = False
    api_server_running: bool = False
    gpu_memory_used_mb: float = 0.0
    cpu_memory_used_mb: float = 0.0
