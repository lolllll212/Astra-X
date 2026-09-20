"""Domain models for the n8n workflow automation tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class N8NCommand(str, Enum):
    STATUS = "get_status"
    LIST_WORKFLOWS = "list_workflows"
    GET_WORKFLOW = "get_workflow"
    IMPORT_WORKFLOW = "import_workflow"
    EXPORT_WORKFLOW = "export_workflow"
    EXECUTE = "execute"
    LIST_EXECUTIONS = "list_executions"
    GET_EXECUTION = "get_execution"
    TRIGGER_WEBHOOK = "trigger_webhook"
    API_REQUEST = "api_request"
    RUN_CLI = "run_cli"


@dataclass
class N8NConfig:
    """Connection + runtime config for an n8n instance."""

    base_url: str = "http://127.0.0.1:5678"
    api_key: str = ""
    cli_path: str = ""
    home: str = ""
    timeout_sec: float = 60.0


@dataclass
class N8NWorkflow:
    id: str = ""
    name: str = ""
    active: bool = False
    tags: list[str] = field(default_factory=list)
    updated_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class N8NExecution:
    id: str = ""
    workflow_id: str = ""
    status: str = ""
    mode: str = ""
    finished: bool = False
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class N8NStatus:
    cli_found: bool = False
    cli_path: str = ""
    version: str = ""
    node_found: bool = False
    node_bin: str = ""
    server_running: bool = False
    base_url: str = ""
    api_key_set: bool = False
    home: str = ""
    workflow_count: int = 0
    error: str = ""
