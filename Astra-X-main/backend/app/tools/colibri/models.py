"""Domain models for Colibri inference engine tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class ColibriCommand(str, Enum):
    CHAT = "chat"
    SERVE = "serve"
    WEB = "web"
    RUN = "run"
    INFO = "info"
    PLAN = "plan"
    MIRROR = "mirror"
    DOCTOR = "doctor"
    BENCH = "bench"
    BUILD = "build"


class ColibriBackend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    AUTO = "auto"


class ExpertTier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    STORAGE = "storage"


@dataclass
class ColibriConfig:
    model_dir: str = ""
    mirror_dir: str = ""
    ram_budget_gb: float = 0.0
    repin_interval_tokens: int = 0
    topp: float = 0.0
    topk: int = 0
    ngen: int = 0
    cap: int = 0
    backend: ColibriBackend = ColibriBackend.AUTO
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ColibriModel:
    family: str = ""
    params_b: float = 0.0
    quant: str = ""
    resident_gb: float = 0.0
    disk_gb: float = 0.0
    path: str = ""
    ready: bool = False


@dataclass
class ColibriRunResult:
    request_id: str = field(default_factory=lambda: f"coli-{uuid4().hex[:8]}")
    ok: bool = False
    output: str = ""
    error: str | None = None
    tokens: int = 0
    tps: float = 0.0
    ttfs_ms: float = 0.0
    engine_stdout: str = ""
    engine_stderr: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ColibriServeSession:
    id: str = field(default_factory=lambda: f"serve-{uuid4().hex[:8]}")
    host: str = "127.0.0.1"
    port: int = 8000
    pid: int | None = None
    model_dir: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    process: Any = None


@dataclass
class ColibriTierSnapshot:
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    storage_gb: float = 0.0
    experts_in_vram: int = 0
    experts_in_ram: int = 0
    experts_in_storage: int = 0
    total_experts: int = 0


@dataclass
class ColibriStatus:
    cli_found: bool = False
    cli_path: str = ""
    version: str = ""
    built: bool = False
    supported_families: list[str] = field(default_factory=list)
    detected_backends: list[str] = field(default_factory=list)
    active_serve: ColibriServeSession | None = None
    ram_budget_gb: float = 0.0
    vram_budget_gb: float = 0.0


@dataclass
class ColibriPlan:
    model_dir: str = ""
    family: str = ""
    params_b: float = 0.0
    ram_needed_gb: float = 0.0
    disk_needed_gb: float = 0.0
    resident_plan_gb: float = 0.0
    recommend_ram_gb: float = 0.0
    tiers: list[dict[str, Any]] = field(default_factory=list)
    recommended_command: str = ""


@dataclass
class ColibriBenchResult:
    tasks: list[str] = field(default_factory=list)
    individual: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_output: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
