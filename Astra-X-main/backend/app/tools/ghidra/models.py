"""Domain models for Ghidra Software Reverse Engineering Framework tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class GhidraAnalysisMode(str, Enum):
    HEADLESS = "headless"
    SCRIPTED = "scripted"
    BATCH = "batch"


class GhidraScriptLang(str, Enum):
    JAVA = "java"
    PYTHON = "python"
    PYGHIDRA = "pyghidra"


@dataclass
class GhidraConfig:
    ghidra_home: str = ""
    jdk_home: str = ""
    analyzer_timeout_sec: int = 0
    max_memory_mb: int = 0
    extra_args: list[str] = field(default_factory=list)
    java_args: list[str] = field(default_factory=list)


@dataclass
class GhidraProject:
    id: str = field(default_factory=lambda: f"ghproject-{uuid4().hex[:8]}")
    name: str = ""
    dir: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GhidraFunction:
    name: str = ""
    entry: str = ""
    size: int = 0
    calling_convention: str = ""
    signature: str = ""
    is_thunk: bool = False
    references: int = 0


@dataclass
class GhidraDecompiled:
    function_name: str = ""
    signature: str = ""
    decompiled_code: str = ""
    c_code: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class GhidraAnalysisResult:
    request_id: str = field(default_factory=lambda: f"ghida-{uuid4().hex[:8]}")
    ok: bool = False
    output: str = ""
    error: str | None = None
    project_path: str = ""
    functions_found: int = 0
    decompiled_functions: int = 0
    duration_sec: float = 0.0
    stdout: str = ""
    stderr: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GhidraScriptResult:
    request_id: str = field(default_factory=lambda: f"ghscr-{uuid4().hex[:8]}")
    ok: bool = False
    output: str = ""
    error: str | None = None
    stdout: str = ""
    stderr: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GhidraStatus:
    installed: bool = False
    ghidra_home: str = ""
    jdk_home: str = ""
    version: str = ""
    analyze_headless_path: str = ""
    pyghidra_path: str = ""
    java_version: str = ""
    projects: list[GhidraProject] = field(default_factory=list)


@dataclass
class GhidraFindings:
    functions: list[GhidraFunction] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
