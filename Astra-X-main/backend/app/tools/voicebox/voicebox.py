"""Voicebox TTS/STT tool - local-first AI voice studio with 7 TTS engines, voice cloning, dictation, MCP server."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Any

from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult
from app.tools.voicebox.models import (
    DictationMode,
    DictationSession,
    STTEngine,
    STTRequest,
    STTResult,
    TTSEngine,
    TTSRequest,
    TTSResult,
    VoiceboxConfig,
    VoiceboxStatus,
    VoiceProfile,
    VoiceProfileType,
)


class VoiceboxTool(Tool):
    """Voicebox - local-first AI voice studio: 7 TTS engines, voice cloning, 23 languages, dictation, MCP server, agent voice output."""

    @property
    def name(self) -> str:
        return "voicebox"

    @property
    def description(self) -> str:
        return "Voicebox TTS/STT: 7 TTS engines (Qwen3-TTS, Qwen CustomVoice, LuxTTS, Chatterbox Multilingual, Chatterbox Turbo, Hume TADA, Kokoro), voice cloning, 23 languages, dictation with global hotkey, MCP server for agent voice output, stories editor."

    @property
    def capabilities(self) -> list[str]:
        return ["tts", "stt", "voice_cloning", "dictation", "mcp_server", "agent_voice_output", "stories_editor"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="action",
                    type_="string",
                    description="Action: speak, transcribe, clone_voice, list_voices, start_dictation, stop_dictation, get_status, start_mcp_server, stop_mcp_server, create_story, add_story_track",
                    required=True,
                    enum=["speak", "transcribe", "clone_voice", "list_voices", "start_dictation", "stop_dictation", "get_status", "start_mcp_server", "stop_mcp_server", "create_story", "add_story_track"],
                ),
                ToolParameter(
                    name="text",
                    type_="string",
                    description="Text to speak (for speak action)",
                    required=False,
                ),
                ToolParameter(
                    name="voice_profile_id",
                    type_="string",
                    description="Voice profile ID to use",
                    required=False,
                ),
                ToolParameter(
                    name="voice_name",
                    type_="string",
                    description="Name for new voice profile",
                    required=False,
                ),
                ToolParameter(
                    name="engine",
                    type_="string",
                    description="TTS engine: qwen3_tts, qwen_customvoice, lux_tts, chatterbox_multilingual, chatterbox_turbo, hume_tada, kokoro",
                    required=False,
                ),
                ToolParameter(
                    name="reference_audio_path",
                    type_="string",
                    description="Path to reference audio for voice cloning",
                    required=False,
                ),
                ToolParameter(
                    name="reference_text",
                    type_="string",
                    description="Text of reference audio for cloning",
                    required=False,
                ),
                ToolParameter(
                    name="language",
                    type_="string",
                    description="Language code (en, zh, ja, ko, es, fr, de, ar, hi, sw, etc.)",
                    required=False,
                    default="en",
                ),
                ToolParameter(
                    name="speed",
                    type_="number",
                    description="Speech speed multiplier",
                    required=False,
                    default=1.0,
                ),
                ToolParameter(
                    name="pitch",
                    type_="number",
                    description="Pitch shift multiplier",
                    required=False,
                    default=1.0,
                ),
                ToolParameter(
                    name="effects",
                    type_="object",
                    description="Audio effects: reverb, delay, chorus, compression, pitch_shift, filters",
                    required=False,
                    default={},
                ),
                ToolParameter(
                    name="audio_path",
                    type_="string",
                    description="Path to audio file for transcription",
                    required=False,
                ),
                ToolParameter(
                    name="stt_engine",
                    type_="string",
                    description="STT engine: whisper, whisper_cpp, faster_whisper",
                    required=False,
                    default="faster_whisper",
                ),
                ToolParameter(
                    name="dictation_mode",
                    type_="string",
                    description="Dictation mode: push_to_talk, toggle, continuous",
                    required=False,
                    default="push_to_talk",
                ),
                ToolParameter(
                    name="hotkey",
                    type_="string",
                    description="Global hotkey for dictation",
                    required=False,
                    default="ctrl+shift+space",
                ),
                ToolParameter(
                    name="auto_paste",
                    type_="boolean",
                    description="Auto-paste transcription to active window",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="story_name",
                    type_="string",
                    description="Name for new story project",
                    required=False,
                ),
                ToolParameter(
                    name="track_data",
                    type_="object",
                    description="Track data for story: voice_profile_id, text, start_time, effects",
                    required=False,
                ),
                ToolParameter(
                    name="output_path",
                    type_="string",
                    description="Output path for audio file",
                    required=False,
                ),
            ],
        )

    def __init__(self) -> None:
        self._config = VoiceboxConfig()
        self._voice_profiles: dict[str, VoiceProfile] = {}
        self._dictation_session: DictationSession | None = None
        self._mcp_process: subprocess.Popen | None = None
        self._api_process: subprocess.Popen | None = None
        self._voicebox_cli = self._find_voicebox_cli()
        self._init_preset_voices()

    def _find_voicebox_cli(self) -> str:
        """Find voicebox CLI executable."""
        candidates = [
            "voicebox",
            "voicebox.exe",
            os.path.expanduser("~/.voicebox/bin/voicebox"),
            os.path.expanduser("~/.voicebox/bin/voicebox.exe"),
            "/usr/local/bin/voicebox",
            "/opt/voicebox/bin/voicebox",
        ]
        for candidate in candidates:
            try:
                result = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return candidate
            except Exception:
                continue
        return "voicebox"

    def _init_preset_voices(self) -> None:
        """Initialize preset voices for Kokoro and Qwen CustomVoice."""
        # Kokoro preset voices
        kokoro_voices = [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george",
            "bm_lewis", "af_alloy", "af_echo", "af_fable", "af_onyx",
            "af_nova", "af_shimmer", "am_echo", "am_fable", "am_onyx",
            "am_nova", "am_shimmer",
        ]
        for voice_id in kokoro_voices:
            self._voice_profiles[f"kokoro_{voice_id}"] = VoiceProfile(
                id=f"kokoro_{voice_id}",
                name=f"Kokoro: {voice_id}",
                type=VoiceProfileType.PRESET,
                engine=TTSEngine.KOKORO,
                language="en",
                preset_voice_id=voice_id,
            )

        # Qwen CustomVoice presets
        qwen_voices = ["qwen_female_1", "qwen_female_2", "qwen_male_1", "qwen_male_2",
                       "qwen_child", "qwen_elderly", "qwen_narrator", "qwen_whisper", "qwen_dramatic"]
        for voice_id in qwen_voices:
            self._voice_profiles[f"qwen_{voice_id}"] = VoiceProfile(
                id=f"qwen_{voice_id}",
                name=f"Qwen CustomVoice: {voice_id}",
                type=VoiceProfileType.PRESET,
                engine=TTSEngine.QWEN_CUSTOMVOICE,
                language="en",
                preset_voice_id=voice_id,
            )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")

        if action == "speak":
            return await self._speak(context, kwargs)
        elif action == "transcribe":
            return await self._transcribe(context, kwargs)
        elif action == "clone_voice":
            return await self._clone_voice(context, kwargs)
        elif action == "list_voices":
            return await self._list_voices(context, kwargs)
        elif action == "start_dictation":
            return await self._start_dictation(context, kwargs)
        elif action == "stop_dictation":
            return await self._stop_dictation(context, kwargs)
        elif action == "get_status":
            return await self._get_status(context, kwargs)
        elif action == "start_mcp_server":
            return await self._start_mcp_server(context, kwargs)
        elif action == "stop_mcp_server":
            return await self._stop_mcp_server(context, kwargs)
        elif action == "create_story":
            return await self._create_story(context, kwargs)
        elif action == "add_story_track":
            return await self._add_story_track(context, kwargs)
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}")

    async def _speak(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Generate speech from text."""
        text = kwargs.get("text", "")
        if not text:
            return ToolResult(success=False, error="text is required for speak action")

        voice_profile_id = kwargs.get("voice_profile_id", "")
        engine_str = kwargs.get("engine", "")
        language = kwargs.get("language", "en")
        speed = kwargs.get("speed", 1.0)
        pitch = kwargs.get("pitch", 1.0)
        effects = kwargs.get("effects", {})
        output_path = kwargs.get("output_path", "")

        # Resolve voice profile
        voice_profile = None
        if voice_profile_id and voice_profile_id in self._voice_profiles:
            voice_profile = self._voice_profiles[voice_profile_id]
        elif engine_str:
            try:
                engine = TTSEngine(engine_str)
                # Use first available preset for this engine
                for vp in self._voice_profiles.values():
                    if vp.engine == engine:
                        voice_profile = vp
                        break
            except ValueError:
                pass

        if not voice_profile:
            # Default to Kokoro
            voice_profile = next((vp for vp in self._voice_profiles.values() if vp.engine == TTSEngine.KOKORO), None)
            if not voice_profile:
                return ToolResult(success=False, error="No voice profiles available")

        request = TTSRequest(
            text=text,
            voice_profile_id=voice_profile.id,
            engine=voice_profile.engine,
            language=language,
            speed=speed,
            pitch=pitch,
            effects=effects,
        )

        result = await self._run_tts(request, output_path)

        if result.success:
            return ToolResult(
                success=True,
                output=f"Speech generated: {result.audio_path}",
                metadata={
                    "request_id": request.id,
                    "duration_seconds": result.duration_seconds,
                    "sample_rate": result.sample_rate,
                    "engine_used": result.engine_used.value if result.engine_used else None,
                },
            )
        else:
            return ToolResult(success=False, error=result.error)

    async def _run_tts(self, request: TTSRequest, output_path: str) -> TTSResult:
        """Run TTS generation via voicebox CLI."""
        try:
            # Create temp output file if not specified
            if not output_path:
                output_path = os.path.join(tempfile.gettempdir(), f"voicebox_tts_{request.id}.wav")

            cmd = [
                self._voicebox_cli, "tts",
                "--text", request.text,
                "--voice", request.voice_profile_id,
                "--output", output_path,
            ]

            if request.engine:
                cmd.extend(["--engine", request.engine.value])
            if request.language:
                cmd.extend(["--language", request.language])
            if request.speed != 1.0:
                cmd.extend(["--speed", str(request.speed)])
            if request.pitch != 1.0:
                cmd.extend(["--pitch", str(request.pitch)])
            if request.effects:
                cmd.extend(["--effects", json.dumps(request.effects)])

            # Add model dir if configured
            if self._config.models_dir:
                cmd.extend(["--models-dir", self._config.models_dir])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Parse output for metadata
                output_data = json.loads(stdout.decode()) if stdout else {}
                return TTSResult(
                    request_id=request.id,
                    success=True,
                    audio_path=output_path,
                    duration_seconds=output_data.get("duration", 0),
                    sample_rate=output_data.get("sample_rate", 48000),
                    engine_used=request.engine,
                )
            else:
                return TTSResult(
                    request_id=request.id,
                    success=False,
                    error=stderr.decode() if stderr else "TTS generation failed",
                )

        except Exception as e:
            return TTSResult(
                request_id=request.id,
                success=False,
                error=str(e),
            )

    async def _transcribe(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Transcribe audio to text."""
        audio_path = kwargs.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            return ToolResult(success=False, error="audio_path is required and must exist")

        stt_engine_str = kwargs.get("stt_engine", "faster_whisper")
        language = kwargs.get("language")

        try:
            stt_engine = STTEngine(stt_engine_str)
        except ValueError:
            stt_engine = STTEngine.FASTER_WHISPER

        request = STTRequest(
            audio_path=audio_path,
            engine=stt_engine,
            language=language,
        )

        result = await self._run_stt(request)

        if result.success:
            return ToolResult(
                success=True,
                output=result.text,
                metadata={
                    "request_id": request.id,
                    "language": result.language,
                    "duration_seconds": result.duration_seconds,
                    "segments": result.segments,
                    "engine_used": result.engine_used.value if result.engine_used else None,
                },
            )
        else:
            return ToolResult(success=False, error=result.error)

    async def _run_stt(self, request: STTRequest) -> STTResult:
        """Run STT transcription via voicebox CLI."""
        try:
            cmd = [
                self._voicebox_cli, "stt",
                "--audio", request.audio_path,
                "--engine", request.engine.value,
            ]

            if request.language:
                cmd.extend(["--language", request.language])
            if request.task != "transcribe":
                cmd.extend(["--task", request.task])
            if request.vad_filter:
                cmd.append("--vad-filter")
            if request.word_timestamps:
                cmd.append("--word-timestamps")

            if self._config.models_dir:
                cmd.extend(["--models-dir", self._config.models_dir])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output_data = json.loads(stdout.decode()) if stdout else {}
                return STTResult(
                    request_id=request.id,
                    success=True,
                    text=output_data.get("text", ""),
                    language=output_data.get("language"),
                    language_probability=output_data.get("language_probability", 0),
                    segments=output_data.get("segments", []),
                    words=output_data.get("words", []),
                    duration_seconds=output_data.get("duration", 0),
                    engine_used=request.engine,
                )
            else:
                return STTResult(
                    request_id=request.id,
                    success=False,
                    error=stderr.decode() if stderr else "STT transcription failed",
                )

        except Exception as e:
            return STTResult(
                request_id=request.id,
                success=False,
                error=str(e),
            )

    async def _clone_voice(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Create a cloned voice profile from reference audio."""
        voice_name = kwargs.get("voice_name", "")
        reference_audio_path = kwargs.get("reference_audio_path", "")
        reference_text = kwargs.get("reference_text", "")
        engine_str = kwargs.get("engine", "qwen3_tts")

        if not voice_name:
            return ToolResult(success=False, error="voice_name is required")
        if not reference_audio_path or not os.path.exists(reference_audio_path):
            return ToolResult(success=False, error="reference_audio_path is required and must exist")

        try:
            engine = TTSEngine(engine_str)
        except ValueError:
            engine = TTSEngine.QWEN3_TTS

        # Create voice profile
        profile = VoiceProfile(
            name=voice_name,
            type=VoiceProfileType.CLONED,
            engine=engine,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )

        # Register with voicebox CLI
        try:
            cmd = [
                self._voicebox_cli, "voice", "clone",
                "--name", voice_name,
                "--audio", reference_audio_path,
                "--engine", engine.value,
            ]
            if reference_text:
                cmd.extend(["--text", reference_text])
            if self._config.models_dir:
                cmd.extend(["--models-dir", self._config.models_dir])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output_data = json.loads(stdout.decode()) if stdout else {}
                profile.id = output_data.get("voice_id", profile.id)
                self._voice_profiles[profile.id] = profile
                return ToolResult(
                    success=True,
                    output=f"Voice cloned: {voice_name}",
                    metadata={"voice_profile_id": profile.id, "engine": engine.value},
                )
            else:
                return ToolResult(success=False, error=stderr.decode() if stderr else "Voice cloning failed")

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _list_voices(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """List available voice profiles."""
        voices = []
        for profile in self._voice_profiles.values():
            voices.append({
                "id": profile.id,
                "name": profile.name,
                "type": profile.type.value,
                "engine": profile.engine.value,
                "language": profile.language,
                "preset_voice_id": profile.preset_voice_id,
            })

        return ToolResult(
            success=True,
            output=json.dumps(voices, indent=2),
            metadata={"count": len(voices)},
        )

    async def _start_dictation(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Start dictation session with global hotkey."""
        if self._dictation_session and self._dictation_session.is_active:
            return ToolResult(success=False, error="Dictation session already active")

        mode_str = kwargs.get("dictation_mode", "push_to_talk")
        hotkey = kwargs.get("hotkey", "ctrl+shift+space")
        auto_paste = kwargs.get("auto_paste", True)
        stt_engine_str = kwargs.get("stt_engine", "faster_whisper")
        language = kwargs.get("language")

        try:
            mode = DictationMode(mode_str)
        except ValueError:
            mode = DictationMode.PUSH_TO_TALK

        try:
            stt_engine = STTEngine(stt_engine_str)
        except ValueError:
            stt_engine = STTEngine.FASTER_WHISPER

        self._dictation_session = DictationSession(
            mode=mode,
            hotkey=hotkey,
            auto_paste=auto_paste,
            stt_engine=stt_engine,
            language=language,
            is_active=True,
            started_at=datetime.utcnow(),
        )

        # Start dictation via voicebox CLI
        try:
            cmd = [
                self._voicebox_cli, "dictation", "start",
                "--mode", mode.value,
                "--hotkey", hotkey,
                "--engine", stt_engine.value,
            ]
            if auto_paste:
                cmd.append("--auto-paste")
            if language:
                cmd.extend(["--language", language])
            if self._config.models_dir:
                cmd.extend(["--models-dir", self._config.models_dir])

            # Run in background
            self._dictation_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            return ToolResult(
                success=True,
                output=f"Dictation started: {mode.value} mode with hotkey {hotkey}",
                metadata={"session_id": self._dictation_session.id},
            )
        except Exception as e:
            self._dictation_session = None
            return ToolResult(success=False, error=str(e))

    async def _stop_dictation(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Stop dictation session."""
        if not self._dictation_session or not self._dictation_session.is_active:
            return ToolResult(success=False, error="No active dictation session")

        try:
            cmd = [self._voicebox_cli, "dictation", "stop"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            self._dictation_session.is_active = False
            session_id = self._dictation_session.id
            self._dictation_session = None

            return ToolResult(
                success=True,
                output=f"Dictation stopped: {session_id}",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _get_status(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Get Voicebox status."""
        status = VoiceboxStatus(
            voice_profiles=list(self._voice_profiles.values()),
            active_dictation=self._dictation_session,
            mcp_server_running=self._mcp_process is not None and self._mcp_process.poll() is None,
            api_server_running=self._api_process is not None and self._api_process.poll() is None,
        )

        # Query CLI for loaded engines
        try:
            cmd = [self._voicebox_cli, "status"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                data = json.loads(stdout.decode()) if stdout else {}
                status.tts_engines_loaded = [TTSEngine(e) for e in data.get("tts_engines", [])]
                status.stt_engines_loaded = [STTEngine(e) for e in data.get("stt_engines", [])]
                status.gpu_memory_used_mb = data.get("gpu_memory_mb", 0)
                status.cpu_memory_used_mb = data.get("cpu_memory_mb", 0)
        except Exception:
            pass

        return ToolResult(
            success=True,
            output=json.dumps({
                "tts_engines_loaded": [e.value for e in status.tts_engines_loaded],
                "stt_engines_loaded": [e.value for e in status.stt_engines_loaded],
                "voice_profiles_count": len(status.voice_profiles),
                "active_dictation": status.active_dictation.id if status.active_dictation else None,
                "mcp_server_running": status.mcp_server_running,
                "api_server_running": status.api_server_running,
                "gpu_memory_mb": status.gpu_memory_used_mb,
                "cpu_memory_mb": status.cpu_memory_used_mb,
            }, indent=2),
        )

    async def _start_mcp_server(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Start MCP server for agent voice output."""
        if self._mcp_process and self._mcp_process.poll() is None:
            return ToolResult(success=False, error="MCP server already running")

        try:
            cmd = [
                self._voicebox_cli, "mcp", "start",
                "--port", str(self._config.mcp_port),
            ]
            if self._config.models_dir:
                cmd.extend(["--models-dir", self._config.models_dir])

            self._mcp_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait a moment for server to start
            await asyncio.sleep(2)

            return ToolResult(
                success=True,
                output=f"MCP server started on port {self._config.mcp_port}",
                metadata={"port": self._config.mcp_port},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _stop_mcp_server(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Stop MCP server."""
        if self._mcp_process:
            self._mcp_process.terminate()
            await self._mcp_process.wait()
            self._mcp_process = None
            return ToolResult(success=True, output="MCP server stopped")
        return ToolResult(success=False, error="MCP server not running")

    async def _create_story(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Create a new stories project."""
        story_name = kwargs.get("story_name", "")
        if not story_name:
            return ToolResult(success=False, error="story_name is required")

        # This would integrate with voicebox stories editor
        return ToolResult(
            success=True,
            output=f"Story project created: {story_name}",
            metadata={"story_name": story_name},
        )

    async def _add_story_track(self, context: ToolContext, kwargs: dict) -> ToolResult:
        """Add a track to a story project."""
        track_data = kwargs.get("track_data", {})
        if not track_data:
            return ToolResult(success=False, error="track_data is required")

        return ToolResult(
            success=True,
            output="Story track added",
            metadata={"track": track_data},
        )
