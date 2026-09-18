"""Domain models for LM Studio local API tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class LmStudioModel(str, Enum):
    DEEPSEEK_R1_DISTILL_QWEN_7B = "deepseek-r1-distill-qwen-7b"
    QWEN25_CODER_7B_INSTRUCT = "qwen2.5-coder-7b-instruct"
    LLAVA_V16_MISTRAL_7B = "llava-v1.6-mistral-7b"
    MISTRAL_7B_INSTRUCT_V02 = "mistral-7b-instruct-v0.2"
    DEEPSEEK_CODER_6_7B_INSTRUCT = "deepseek-coder-6.7b-instruct"
    GEMMA_3_4B = "google/gemma-3-4b"


@dataclass
class LmStudioModelInfo:
    id: str = ""
    object: str = "model"
    created: int = 0
    owned_by: str = "lm-studio"
    local_path: str = ""


@dataclass
class LmStudioChatResult:
    request_id: str = field(default_factory=lambda: f"lm-{uuid4().hex[:8]}")
    ok: bool = False
    content: str = ""
    model: str = ""
    finish_reason: str = ""
    usage_prompt_tokens: int = 0
    usage_completion_tokens: int = 0
    usage_total_tokens: int = 0
    reasoning_content: str = ""
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LmStudioCompletionResult:
    request_id: str = field(default_factory=lambda: f"lm-{uuid4().hex[:8]}")
    ok: bool = False
    text: str = ""
    model: str = ""
    finish_reason: str = ""
    usage_prompt_tokens: int = 0
    usage_completion_tokens: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LmStudioEmbedding:
    vector: list[float] = field(default_factory=list)
    nbytes: int = 0
    index: int = 0


@dataclass
class LmStudioStatus:
    base_url: str = ""
    reachable: bool = False
    models: list[LmStudioModelInfo] = field(default_factory=list)
    server_version: str = ""
    latencies_ms: float = 0.0
