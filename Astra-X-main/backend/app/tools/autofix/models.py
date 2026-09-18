"""Domain models for the AutoFix autonomous repair agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AutoFixCandidate:
    """A file the agent proposes to inspect and possibly repair."""

    path: str
    reason: str = ""
    language: str = ""
    sensitive: bool = False


@dataclass
class FileEdit:
    """A single concrete file change produced by the repair agent."""

    path: str
    kind: str = "write"  # write | create
    summary: str = ""
    before: str = ""
    after: str = ""
    applied: bool = False
    needs_approval: bool = False
    verified: bool = False
    backup_path: str = ""


@dataclass
class AutoFixRun:
    """A full repair session (one prompt -> one manifest)."""

    id: str = ""
    prompt: str = ""
    target_dir: str = ""
    scans: list[str] = field(default_factory=list)
    candidates: list[AutoFixCandidate] = field(default_factory=list)
    edits: list[FileEdit] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    status: str = "running"  # running | awaiting_approval | completed | failed
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    approval_token: str = ""
    model: str = ""


@dataclass
class AutoFixStatus:
    """Overall AutoFix tool status."""

    enabled: bool = True
    base_dir: str = ""
    last_run_id: str = ""
    last_run_status: str = ""
    runs_count: int = 0
    edits_applied: int = 0
    edits_pending_approval: int = 0
    sensitive_detected: int = 0
