"""Domain models for ECC (Engineering Control Center) framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class AgentRole(str, Enum):
    PLANNER = "planner"
    REVIEWER = "reviewer"
    BUILD_REPAIR = "build_repair"
    SECURITY = "security"
    ARCHITECT = "architect"
    DOMAIN_EXPERT = "domain_expert"
    TDD_SPECIALIST = "tdd_specialist"
    RESEARCHER = "researcher"
    DOCS_WRITER = "docs_writer"
    FRONTEND_DEV = "frontend_dev"
    DATA_ENGINEER = "data_engineer"
    ML_ENGINEER = "ml_engineer"
    OPS_ENGINEER = "ops_engineer"


class SkillCategory(str, Enum):
    TDD = "tdd"
    RESEARCH = "research"
    SECURITY = "security"
    DOCS = "docs"
    FRONTEND = "frontend"
    DATA = "data"
    ML = "ml"
    OPERATIONS = "operations"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    REVIEW = "review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"


class HookType(str, Enum):
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    PRE_AGENT = "pre_agent"
    POST_AGENT = "post_agent"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"
    LEARNING = "learning"


class MemoryType(str, Enum):
    SESSION_SUMMARY = "session_summary"
    INSTINCT = "instinct"
    CONTEXT_CONTROL = "context_control"
    CONTINUOUS_LEARNING = "continuous_learning"
    PROJECT_RULE = "project_rule"
    LANGUAGE_STANDARD = "language_standard"


class SecurityScanType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    HOOK_ANALYSIS = "hook_analysis"
    MCP_CONFIG = "mcp_config"
    PERMISSIONS = "permissions"
    SECRETS = "secrets"
    AGENT_FILES = "agent_files"


@dataclass
class ECCAgent:
    id: str = field(default_factory=lambda: f"agent-{uuid4().hex[:8]}")
    name: str = ""
    role: AgentRole = AgentRole.DOMAIN_EXPERT
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    system_prompt: str = ""
    model: str = ""
    provider: str = ""
    temperature: float = 0.3
    max_tokens: int = 8192
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ECCSkill:
    id: str = field(default_factory=lambda: f"skill-{uuid4().hex[:8]}")
    name: str = ""
    category: SkillCategory = SkillCategory.RESEARCH
    description: str = ""
    version: str = "1.0.0"
    agent_compatibility: list[AgentRole] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    entry_point: str = ""
    config_schema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ECCHook:
    id: str = field(default_factory=lambda: f"hook-{uuid4().hex[:8]}")
    name: str = ""
    type: HookType = HookType.PRE_TOOL
    description: str = ""
    script: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ECCMemory:
    id: str = field(default_factory=lambda: f"mem-{uuid4().hex[:8]}")
    type: MemoryType = MemoryType.SESSION_SUMMARY
    key: str = ""
    value: str = ""
    tags: list[str] = field(default_factory=list)
    project_id: str | None = None
    language: str | None = None
    importance: float = 0.5
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ECCRule:
    id: str = field(default_factory=lambda: f"rule-{uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    language: str | None = None
    project_pattern: str | None = None
    rule_content: str = ""
    always_load: bool = True
    priority: int = 100
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityFinding:
    id: str = field(default_factory=lambda: f"sec-{uuid4().hex[:8]}")
    scan_type: SecurityScanType = SecurityScanType.PROMPT_INJECTION
    severity: str = "medium"
    title: str = ""
    description: str = ""
    file_path: str | None = None
    line_number: int | None = None
    evidence: str = ""
    remediation: str = ""
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ECCConfig:
    agents_dir: str = ""
    skills_dir: str = ""
    hooks_dir: str = ""
    memory_dir: str = ""
    rules_dir: str = ""
    auto_load_skills: bool = True
    auto_load_hooks: bool = True
    memory_retention_days: int = 30
    max_memory_entries: int = 10000
    security_scan_on_startup: bool = True
    continuous_learning: bool = True


@dataclass
class WorkflowStep:
    id: str = field(default_factory=lambda: f"step-{uuid4().hex[:8]}")
    name: str = ""
    agent_role: AgentRole = AgentRole.DOMAIN_EXPERT
    skill_id: str | None = None
    description: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


@dataclass
class ECCWorkflow:
    id: str = field(default_factory=lambda: f"wf-{uuid4().hex[:8]}")
    name: str = "plan -> test -> implement -> review -> verify -> remember -> improve"
    steps: list[WorkflowStep] = field(default_factory=list)
    current_step: int = 0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ECCStatus:
    agents_loaded: int = 0
    skills_loaded: int = 0
    hooks_loaded: int = 0
    memory_entries: int = 0
    rules_loaded: int = 0
    active_workflows: int = 0
    security_scan_last_run: datetime | None = None
    security_findings_count: int = 0
    ecc_home: str = ""
    ecc_cli_available: bool = False
    ecc_llm_src_available: bool = False
