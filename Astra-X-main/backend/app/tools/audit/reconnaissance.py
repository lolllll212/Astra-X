"""Reconnaissance tool for Phase 1 - architecture mapping and coverage ledger creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.tools.audit.models import (
    Architecture,
    AttackClass,
    CoverageUnit,
    CoverageUnitType,
    InputSurface,
    InputSurfaceType,
    PriorEvidence,
    TrustBoundary,
    TrustBoundaryType,
)
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class ReconnaissanceTool(Tool):
    """Phase 1: Map architecture, trust boundaries, input surfaces, prior evidence, and create deterministic coverage ledger."""

    @property
    def name(self) -> str:
        return "reconnaissance"

    @property
    def description(self) -> str:
        return "Perform Phase 1 reconnaissance: map architecture, trust boundaries, input surfaces, prior evidence, and generate deterministic coverage ledger."

    @property
    def capabilities(self) -> list[str]:
        return ["reconnaissance", "architecture_mapping", "coverage_ledger_generation"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="target_description",
                    type_="string",
                    description="High-level description of the target system",
                    required=True,
                ),
                ToolParameter(
                    name="architecture_markdown",
                    type_="string",
                    description="Existing architecture.md content if available",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="source_code_paths",
                    type_="array",
                    description="List of source code paths to analyze",
                    required=False,
                    default=[],
                ),
                ToolParameter(
                    name="config_files",
                    type_="array",
                    description="List of configuration files to analyze",
                    required=False,
                    default=[],
                ),
                ToolParameter(
                    name="output_dir",
                    type_="string",
                    description="Directory to write architecture.md and coverage-ledger.json",
                    required=True,
                ),
                ToolParameter(
                    name="include_attack_classes",
                    type_="array",
                    description="Attack classes to consider (e.g., 'injection', 'auth_bypass', 'idor')",
                    required=False,
                    default=[],
                ),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        target_description = kwargs.get("target_description", "")
        architecture_markdown = kwargs.get("architecture_markdown", "")
        source_code_paths = kwargs.get("source_code_paths", [])
        config_files = kwargs.get("config_files", [])
        output_dir = kwargs.get("output_dir", "")
        include_attack_classes = kwargs.get("include_attack_classes", [])

        if not output_dir:
            return ToolResult(success=False, error="output_dir is required")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        architecture = Architecture()

        # Parse existing architecture.md if provided
        if architecture_markdown:
            architecture = self._parse_architecture_markdown(architecture_markdown)

        # Analyze source code paths for trust boundaries and input surfaces
        for path_str in source_code_paths:
            path = Path(path_str)
            if path.exists():
                self._analyze_source_code(path, architecture)

        # Analyze config files
        for config_str in config_files:
            config_path = Path(config_str)
            if config_path.exists():
                self._analyze_config_file(config_path, architecture)

        # Generate coverage ledger from architecture
        coverage_ledger = self._generate_coverage_ledger(architecture, include_attack_classes)
        architecture.coverage_ledger = coverage_ledger

        # Write architecture.md
        arch_md = self._generate_architecture_markdown(architecture, target_description)
        (output_path / "architecture.md").write_text(arch_md, encoding="utf-8")

        # Write coverage-ledger.json
        ledger_data = self._coverage_ledger_to_json(architecture)
        (output_path / "coverage-ledger.json").write_text(
            json.dumps(ledger_data, indent=2), encoding="utf-8"
        )

        return ToolResult(
            success=True,
            output=f"Reconnaissance complete. Generated architecture.md and coverage-ledger.json in {output_dir}",
            metadata={
                "trust_boundaries": len(architecture.trust_boundaries),
                "input_surfaces": len(architecture.input_surfaces),
                "prior_evidence": len(architecture.prior_evidence),
                "coverage_units": len(architecture.coverage_ledger),
                "architecture_md": str(output_path / "architecture.md"),
                "coverage_ledger_json": str(output_path / "coverage-ledger.json"),
            },
        )

    def _parse_architecture_markdown(self, markdown: str) -> Architecture:
        """Parse existing architecture.md into Architecture model."""
        arch = Architecture()
        # Simple parsing - in practice this would be more sophisticated
        return arch

    def _analyze_source_code(self, path: Path, architecture: Architecture) -> None:
        """Analyze source code for trust boundaries and input surfaces."""
        # This would do actual static analysis in a real implementation
        # For now, add some heuristic-based detection
        for file_path in path.rglob("*"):
            if file_path.suffix in (".py", ".js", ".ts", ".go", ".java", ".rs", ".cpp", ".c"):
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                self._detect_trust_boundaries(file_path, content, architecture)
                self._detect_input_surfaces(file_path, content, architecture)

    def _detect_trust_boundaries(self, file_path: Path, content: str, architecture: Architecture) -> None:
        """Heuristically detect trust boundaries from code patterns."""
        patterns = {
            TrustBoundaryType.NETWORK: ["socket", "bind", "listen", "http", "grpc", "websocket"],
            TrustBoundaryType.PROCESS: ["subprocess", "popen", "exec", "fork", "spawn"],
            TrustBoundaryType.USER_KERNEL: ["ioctl", "syscall", "mmap", "mprotect", "ptrace"],
            TrustBoundaryType.CONTAINER_HOST: ["docker", "containerd", "kubernetes", "cgroup", "namespace"],
            TrustBoundaryType.TENANT: ["tenant", "multi-tenant", "isolation", "namespace"],
            TrustBoundaryType.SERVICE: ["microservice", "service mesh", "istio", "linkerd", "consul"],
            TrustBoundaryType.DATA_STORE: ["database", "sql", "redis", "mongodb", "cassandra"],
            TrustBoundaryType.EXTERNAL_API: ["requests", "httpclient", "fetch", "axios", "external api"],
        }

        for boundary_type, keywords in patterns.items():
            if any(kw in content.lower() for kw in keywords):
                # Check if we already have this boundary type
                existing = next(
                    (tb for tb in architecture.trust_boundaries if tb.type == boundary_type),
                    None
                )
                if not existing:
                    architecture.trust_boundaries.append(
                        TrustBoundary(
                            name=f"{boundary_type.value}_boundary",
                            type=boundary_type,
                            description=f"Detected from {file_path.name}",
                            components=[str(file_path)],
                        )
                    )

    def _detect_input_surfaces(self, file_path: Path, content: str, architecture: Architecture) -> None:
        """Heuristically detect input surfaces from code patterns."""
        patterns = {
            InputSurfaceType.HTTP_ENDPOINT: ["@app.route", "@app.get", "@app.post", "fastapi", "flask", "express", "router"],
            InputSurfaceType.RPC_METHOD: ["grpc", "rpc", "protobuf", "thrift"],
            InputSurfaceType.MESSAGE_QUEUE: ["kafka", "rabbitmq", "pubsub", "sqs", "queue"],
            InputSurfaceType.FILE_UPLOAD: ["upload", "multipart", "file(", "request.files"],
            InputSurfaceType.CLI_ARGUMENT: ["argparse", "click", "sys.argv", "command line"],
            InputSurfaceType.ENV_VARIABLE: ["os.environ", "os.getenv", "getenv", "dotenv"],
            InputSurfaceType.CONFIG_FILE: ["config", "yaml", "toml", "json", ".ini", ".conf"],
            InputSurfaceType.DATABASE_INPUT: ["execute", "query", "cursor", "orm", "sqlalchemy"],
            InputSurfaceType.IPC_CHANNEL: ["ipc", "pipe", "socket", "shared memory", "mmap"],
            InputSurfaceType.WEBHOOK: ["webhook", "callback", "webhook_handler"],
        }

        for surface_type, keywords in patterns.items():
            if any(kw in content.lower() for kw in keywords):
                architecture.input_surfaces.append(
                    InputSurface(
                        name=f"{surface_type.value}_{file_path.stem}",
                        type=surface_type,
                        location=str(file_path),
                        description=f"Detected in {file_path.name}",
                    )
                )

    def _analyze_config_file(self, path: Path, architecture: Architecture) -> None:
        """Analyze configuration files for additional context."""
        content = path.read_text(encoding="utf-8", errors="ignore")
        # Look for service definitions, ports, endpoints, etc.
        if "port" in content.lower() or "endpoint" in content.lower():
            architecture.input_surfaces.append(
                InputSurface(
                    name=f"config_{path.stem}",
                    type=InputSurfaceType.CONFIG_FILE,
                    location=str(path),
                    description=f"Configuration file with potential endpoints",
                )
            )

    def _generate_coverage_ledger(
        self, architecture: Architecture, include_attack_classes: list[str]
    ) -> list[CoverageUnit]:
        """Generate deterministic coverage ledger from architecture."""
        ledger = []

        # Coverage units for trust boundaries
        for tb in architecture.trust_boundaries:
            ledger.append(
                CoverageUnit(
                    type=CoverageUnitType.TRUST_BOUNDARY,
                    name=f"trust_boundary_{tb.name}",
                    description=f"Coverage for {tb.type.value} trust boundary: {tb.description}",
                    reference_ids=[tb.id],
                    attack_classes=include_attack_classes or [
                        "privilege_escalation",
                        "confused_deputy",
                        "toctou",
                        "side_channel",
                    ],
                    priority=3,
                )
            )

        # Coverage units for input surfaces
        for isurf in architecture.input_surfaces:
            ledger.append(
                CoverageUnit(
                    type=CoverageUnitType.INPUT_SURFACE,
                    name=f"input_surface_{isurf.name}",
                    description=f"Coverage for {isurf.type.value} input surface: {isurf.description}",
                    reference_ids=[isurf.id],
                    attack_classes=include_attack_classes or [
                        "injection",
                        "xss",
                        "deserialization",
                        "idor",
                        "auth_bypass",
                        "rate_limit_bypass",
                    ],
                    priority=3,
                )
            )

        # Coverage units for prior evidence
        for evidence in architecture.prior_evidence:
            for attack_class in evidence.attack_classes:
                ledger.append(
                    CoverageUnit(
                        type=CoverageUnitType.ATTACK_CLASS,
                        name=f"evidence_{evidence.id}_{attack_class}",
                        description=f"Coverage for {attack_class} based on prior evidence: {evidence.title}",
                        reference_ids=[evidence.id],
                        attack_classes=[attack_class],
                        priority=5,  # Higher priority for known issues
                    )
                )

        # Default attack class coverage units if no prior evidence
        default_classes = include_attack_classes or [
            "injection",
            "auth_bypass",
            "idor",
            "xss",
            "csrf",
            "ssrf",
            "xxe",
            "deserialization",
            "path_traversal",
            "privilege_escalation",
            "race_condition",
            "side_channel",
            "supply_chain",
            "configuration",
            "crypto",
        ]

        for attack_class in default_classes:
            ledger.append(
                CoverageUnit(
                    type=CoverageUnitType.ATTACK_CLASS,
                    name=f"attack_class_{attack_class}",
                    description=f"General coverage for {attack_class} attack class",
                    reference_ids=[],
                    attack_classes=[attack_class],
                    priority=2,
                )
            )

        return ledger

    def _generate_architecture_markdown(self, architecture: Architecture, target_description: str) -> str:
        """Generate architecture.md from Architecture model."""
        lines = [
            f"# Architecture: {target_description}",
            "",
            "## Overview",
            target_description,
            "",
            "## Trust Boundaries",
            "",
        ]

        for tb in architecture.trust_boundaries:
            lines.extend([
                f"### {tb.name} ({tb.type.value})",
                f"**Description:** {tb.description}",
                f"**Components:** {', '.join(tb.components) if tb.components else 'None identified'}",
                f"**Crossing Mechanisms:** {', '.join(tb.crossing_mechanisms) if tb.crossing_mechanisms else 'None identified'}",
                f"**Assumptions:** {', '.join(tb.assumptions) if tb.assumptions else 'None documented'}",
                "",
            ])

        lines.extend([
            "## Input Surfaces",
            "",
        ])

        for isurf in architecture.input_surfaces:
            lines.extend([
                f"### {isurf.name} ({isurf.type.value})",
                f"**Location:** {isurf.location}",
                f"**Description:** {isurf.description}",
                f"**Parameters:** {json.dumps(isurf.parameters, indent=2) if isurf.parameters else 'None identified'}",
                f"**Authentication:** {isurf.authentication or 'Not specified'}",
                f"**Authorization:** {isurf.authorization or 'Not specified'}",
                f"**Rate Limiting:** {isurf.rate_limiting or 'Not specified'}",
                f"**Validation:** {isurf.validation or 'Not specified'}",
                "",
            ])

        lines.extend([
            "## Prior Evidence",
            "",
        ])

        for evidence in architecture.prior_evidence:
            lines.extend([
                f"### {evidence.title}",
                f"**Source:** {evidence.source}",
                f"**Summary:** {evidence.summary}",
                f"**CVEs:** {', '.join(evidence.cve_ids) if evidence.cve_ids else 'None'}",
                f"**Relevant Components:** {', '.join(evidence.relevant_components) if evidence.relevant_components else 'None'}",
                f"**Attack Classes:** {', '.join(evidence.attack_classes) if evidence.attack_classes else 'None'}",
                f"**Confidence:** {evidence.confidence}",
                f"**URL:** {evidence.url or 'N/A'}",
                "",
            ])

        lines.extend([
            "## Coverage Ledger Summary",
            "",
            f"Total Coverage Units: {len(architecture.coverage_ledger)}",
            "",
        ])

        for cu in architecture.coverage_ledger:
            lines.append(f"- **{cu.name}** ({cu.type.value}) - Priority: {cu.priority} - Attack Classes: {', '.join(cu.attack_classes)}")

        return "\n".join(lines)

    def _coverage_ledger_to_json(self, architecture: Architecture) -> dict[str, Any]:
        """Convert coverage ledger to JSON format."""
        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "coverage_units": [
                {
                    "id": cu.id,
                    "type": cu.type.value,
                    "name": cu.name,
                    "description": cu.description,
                    "reference_ids": cu.reference_ids,
                    "attack_classes": cu.attack_classes,
                    "priority": cu.priority,
                    "status": cu.status.value,
                    "assigned_hunter": cu.assigned_hunter,
                    "findings": cu.findings,
                }
                for cu in architecture.coverage_ledger
            ],
            "summary": {
                "total_units": len(architecture.coverage_ledger),
                "by_type": self._count_by_type(architecture.coverage_ledger),
                "by_priority": self._count_by_priority(architecture.coverage_ledger),
            },
        }

    def _count_by_type(self, units: list[CoverageUnit]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for u in units:
            counts[u.type.value] = counts.get(u.type.value, 0) + 1
        return counts

    def _count_by_priority(self, units: list[CoverageUnit]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for u in units:
            key = f"P{u.priority}"
            counts[key] = counts.get(key, 0) + 1
        return counts


# Import datetime for the tool
from datetime import datetime