"""Domain models for the security audit framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class TrustBoundaryType(str, Enum):
    NETWORK = "network"
    PROCESS = "process"
    USER_KERNEL = "user_kernel"
    CONTAINER_HOST = "container_host"
    TENANT = "tenant"
    SERVICE = "service"
    DATA_STORE = "data_store"
    EXTERNAL_API = "external_api"


class InputSurfaceType(str, Enum):
    HTTP_ENDPOINT = "http_endpoint"
    RPC_METHOD = "rpc_method"
    MESSAGE_QUEUE = "message_queue"
    FILE_UPLOAD = "file_upload"
    CLI_ARGUMENT = "cli_argument"
    ENV_VARIABLE = "env_variable"
    CONFIG_FILE = "config_file"
    DATABASE_INPUT = "database_input"
    IPC_CHANNEL = "ipc_channel"
    WEBHOOK = "webhook"


class CoverageUnitType(str, Enum):
    TRUST_BOUNDARY = "trust_boundary"
    INPUT_SURFACE = "input_surface"
    ATTACK_CLASS = "attack_class"
    DATA_FLOW = "data_flow"
    AUTH_PATH = "auth_path"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SIDE_CHANNEL = "side_channel"
    SUPPLY_CHAIN = "supply_chain"


class AttackClass(str, Enum):
    INJECTION = "injection"
    AUTH_BYPASS = "auth_bypass"
    IDOR = "idor"
    XSS = "xss"
    CSRF = "csrf"
    SSRF = "ssrf"
    XXE = "xxe"
    DESERIALIZATION = "deserialization"
    PATH_TRAVERSAL = "path_traversal"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RACE_CONDITION = "race_condition"
    SIDE_CHANNEL = "side_channel"
    SUPPLY_CHAIN = "supply_chain"
    CONFIGURATION = "configuration"
    CRYPTO = "crypto"


class Verdict(str, Enum):
    CONFIRMED = "confirmed"
    NEEDS_VALIDATION = "needs_validation"
    REJECTED = "rejected"


class HunterStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    DISPROVED = "disproved"
    INCONCLUSIVE = "inconclusive"


@dataclass
class TrustBoundary:
    id: str = field(default_factory=lambda: f"tb-{uuid4().hex[:8]}")
    name: str = ""
    type: TrustBoundaryType = TrustBoundaryType.NETWORK
    description: str = ""
    components: list[str] = field(default_factory=list)
    crossing_mechanisms: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InputSurface:
    id: str = field(default_factory=lambda: f"is-{uuid4().hex[:8]}")
    name: str = ""
    type: InputSurfaceType = InputSurfaceType.HTTP_ENDPOINT
    location: str = ""
    description: str = ""
    parameters: list[dict[str, Any]] = field(default_factory=list)
    authentication: str = ""
    authorization: str = ""
    rate_limiting: str = ""
    validation: str = ""
    trust_boundary_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PriorEvidence:
    id: str = field(default_factory=lambda: f"pe-{uuid4().hex[:8]}")
    source: str = ""
    title: str = ""
    summary: str = ""
    url: str | None = None
    cve_ids: list[str] = field(default_factory=list)
    relevant_components: list[str] = field(default_factory=list)
    attack_classes: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CoverageUnit:
    id: str = field(default_factory=lambda: f"cu-{uuid4().hex[:8]}")
    type: CoverageUnitType = CoverageUnitType.INPUT_SURFACE
    name: str = ""
    description: str = ""
    reference_ids: list[str] = field(default_factory=list)
    attack_classes: list[str] = field(default_factory=list)
    priority: int = 1
    status: HunterStatus = HunterStatus.PENDING
    assigned_hunter: str | None = None
    findings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


@dataclass
class Architecture:
    trust_boundaries: list[TrustBoundary] = field(default_factory=list)
    input_surfaces: list[InputSurface] = field(default_factory=list)
    prior_evidence: list[PriorEvidence] = field(default_factory=list)
    coverage_ledger: list[CoverageUnit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Finding:
    id: str = field(default_factory=lambda: f"f-{uuid4().hex[:8]}")
    title: str = ""
    description: str = ""
    verdict: Verdict = Verdict.NEEDS_VALIDATION
    severity: str = "medium"
    attack_class: str = ""
    coverage_unit_id: str = ""
    hunter_id: str = ""
    evidence: list[str] = field(default_factory=list)
    steps_to_reproduce: list[str] = field(default_factory=list)
    impact: str = ""
    remediation: str = ""
    references: list[str] = field(default_factory=list)
    confidence: float = 0.5
    verifier_id: str | None = None
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verification_notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Hunter:
    id: str = field(default_factory=lambda: f"h-{uuid4().hex[:8]}")
    name: str = ""
    specialization: list[str] = field(default_factory=list)
    coverage_unit_id: str = ""
    status: HunterStatus = HunterStatus.PENDING
    checks_performed: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class Verifier:
    id: str = field(default_factory=lambda: f"v-{uuid4().hex[:8]}")
    name: str = ""
    finding_id: str = ""
    status: VerificationStatus = VerificationStatus.PENDING
    conclusion: Verdict | None = None
    reasoning: str = ""
    evidence_reviewed: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


@dataclass
class CoverageCriticResult:
    coverage_unit_id: str = ""
    gaps_found: list[str] = field(default_factory=list)
    missing_attack_classes: list[str] = field(default_factory=list)
    recommended_new_units: list[str] = field(default_factory=list)
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)