"""Sandbox infrastructure for secure Python execution.

Current state — **scaffold**.  Concrete sandbox implementations
(container-based, WASM, subprocess jail) are added on demand.

The :class:`PythonRunnerTool` uses a basic subprocess sandbox by default.
Replace the ``_run_in_sandbox`` method with a tighter implementation when
needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for the Python execution sandbox."""

    enabled: bool = False
    timeout_seconds: int = 10
    max_memory_mb: int = 256
    max_output_bytes: int = 1_048_576  # 1 MiB
    allowed_modules: tuple[str, ...] = ()
    blocked_modules: tuple[str, ...] = (
        "os",
        "subprocess",
        "shutil",
        "signal",
        "ctypes",
        "socket",
        "http",
        "urllib",
        "requests",
        "httpx",
        "pathlib",
        "tempfile",
    )
    network_access: bool = False
    filesystem_access: bool = False
