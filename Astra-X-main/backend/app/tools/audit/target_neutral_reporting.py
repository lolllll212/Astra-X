"""Target-neutral reporting tool for Phase 6 - derive reports from verified records and coverage ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.tools.audit.models import Finding, Verdict
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class TargetNeutralReportingTool(Tool):
    """Phase 6: Target-neutral reporting - derive REPORT.md, FINDINGS-DETAIL.md, NEEDS-VALIDATION.md from verified records and coverage ledger."""

    @property
    def name(self) -> str:
        return "target_neutral_reporting"

    @property
    def description(self) -> str:
        return "Perform Phase 6 target-neutral reporting: generate REPORT.md, FINDINGS-DETAIL.md, and NEEDS-VALIDATION.md from verified findings and coverage ledger."

    @property
    def capabilities(self) -> list[str]:
        return ["reporting", "target_neutral_reporting", "markdown_generation"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="findings_path",
                    type_="string",
                    description="Path to final findings.json from independent verification",
                    required=True,
                ),
                ToolParameter(
                    name="coverage_ledger_path",
                    type_="string",
                    description="Path to coverage-ledger.json from reconnaissance",
                    required=True,
                ),
                ToolParameter(
                    name="architecture_path",
                    type_="string",
                    description="Path to architecture.md from reconnaissance",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="output_dir",
                    type_="string",
                    description="Directory to write reports",
                    required=True,
                ),
                ToolParameter(
                    name="target_name",
                    type_="string",
                    description="Name of the target system for report headers",
                    required=False,
                    default="Target System",
                ),
                ToolParameter(
                    name="include_executive_summary",
                    type_="boolean",
                    description="Whether to include executive summary in REPORT.md",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="include_methodology",
                    type_="boolean",
                    description="Whether to include methodology section",
                    required=False,
                    default=True,
                ),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        findings_path = kwargs.get("findings_path", "")
        coverage_ledger_path = kwargs.get("coverage_ledger_path", "")
        architecture_path = kwargs.get("architecture_path", "")
        output_dir = kwargs.get("output_dir", "")
        target_name = kwargs.get("target_name", "Target System")
        include_executive_summary = kwargs.get("include_executive_summary", True)
        include_methodology = kwargs.get("include_methodology", True)

        if not findings_path or not coverage_ledger_path or not output_dir:
            return ToolResult(success=False, error="findings_path, coverage_ledger_path, and output_dir are required")

        # Load data
        findings_data = json.loads(Path(findings_path).read_text(encoding="utf-8"))
        coverage_data = json.loads(Path(coverage_ledger_path).read_text(encoding="utf-8"))
        architecture_md = ""
        if architecture_path and Path(architecture_path).exists():
            architecture_md = Path(architecture_path).read_text(encoding="utf-8")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Parse findings
        findings = self._parse_findings(findings_data)

        # Generate reports
        report_md = self._generate_report_md(
            findings, coverage_data, architecture_md, target_name,
            include_executive_summary, include_methodology
        )
        (output_path / "REPORT.md").write_text(report_md, encoding="utf-8")

        findings_detail_md = self._generate_findings_detail_md(findings, target_name)
        (output_path / "FINDINGS-DETAIL.md").write_text(findings_detail_md, encoding="utf-8")

        needs_validation_md = self._generate_needs_validation_md(findings, target_name)
        (output_path / "NEEDS-VALIDATION.md").write_text(needs_validation_md, encoding="utf-8")

        # Also write JSON summary
        summary_json = self._generate_summary_json(findings, coverage_data)
        (output_path / "report-summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

        confirmed_count = sum(1 for f in findings if f.verdict == Verdict.CONFIRMED)
        needs_validation_count = sum(1 for f in findings if f.verdict == Verdict.NEEDS_VALIDATION)
        rejected_count = sum(1 for f in findings if f.verdict == Verdict.REJECTED)

        return ToolResult(
            success=True,
            output=f"Target-neutral reporting complete. Generated REPORT.md, FINDINGS-DETAIL.md, NEEDS-VALIDATION.md",
            metadata={
                "confirmed": confirmed_count,
                "needs_validation": needs_validation_count,
                "rejected": rejected_count,
                "report_md": str(output_path / "REPORT.md"),
                "findings_detail_md": str(output_path / "FINDINGS-DETAIL.md"),
                "needs_validation_md": str(output_path / "NEEDS-VALIDATION.md"),
                "summary_json": str(output_path / "report-summary.json"),
            },
        )

    def _parse_findings(self, data: dict) -> list[Finding]:
        """Parse findings from various JSON formats."""
        findings = []

        # Handle flat format
        if "findings" in data and isinstance(data["findings"], list):
            for f_data in data["findings"]:
                findings.append(self._dict_to_finding(f_data))

        # Handle organized format
        for verdict_key in ["confirmed", "needs_validation", "rejected"]:
            if verdict_key in data and isinstance(data[verdict_key], list):
                for f_data in data[verdict_key]:
                    findings.append(self._dict_to_finding(f_data))

        return findings

    def _dict_to_finding(self, f_data: dict) -> Finding:
        return Finding(
            id=f_data.get("id", ""),
            title=f_data.get("title", ""),
            description=f_data.get("description", ""),
            verdict=Verdict(f_data.get("verdict", "needs_validation")),
            severity=f_data.get("severity", "medium"),
            attack_class=f_data.get("attack_class", ""),
            coverage_unit_id=f_data.get("coverage_unit_id", ""),
            hunter_id=f_data.get("hunter_id", ""),
            evidence=f_data.get("evidence", []),
            steps_to_reproduce=f_data.get("steps_to_reproduce", []),
            impact=f_data.get("impact", ""),
            remediation=f_data.get("remediation", ""),
            references=f_data.get("references", []),
            confidence=f_data.get("confidence", 0.5),
            verifier_id=f_data.get("verifier_id"),
            verification_status=f_data.get("verification_status", "pending"),
            verification_notes=f_data.get("verification_notes", ""),
        )

    def _generate_report_md(
        self,
        findings: list[Finding],
        coverage_data: dict,
        architecture_md: str,
        target_name: str,
        include_executive_summary: bool,
        include_methodology: bool,
    ) -> str:
        """Generate REPORT.md"""
        lines = []

        # Header
        lines.append(f"# Security Audit Report: {target_name}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append(f"**Version:** 1.0")
        lines.append("")

        if include_executive_summary:
            lines.append("## Executive Summary")
            lines.append("")
            confirmed = [f for f in findings if f.verdict == Verdict.CONFIRMED]
            critical = [f for f in confirmed if f.severity == "critical"]
            high = [f for f in confirmed if f.severity == "high"]
            medium = [f for f in confirmed if f.severity == "medium"]
            low = [f for f in confirmed if f.severity == "low"]

            lines.append(f"This report presents the results of a comprehensive security audit of **{target_name}**.")
            lines.append(f"The audit identified **{len(confirmed)} confirmed vulnerabilities**, ")
            lines.append(f"including **{len(critical)} critical**, **{len(high)} high**, **{len(medium)} medium**, and **{len(low)} low** severity issues.")
            lines.append(f"Additionally, **{sum(1 for f in findings if f.verdict == Verdict.NEEDS_VALIDATION)} findings require further validation** ")
            lines.append(f"and **{sum(1 for f in findings if f.verdict == Verdict.REJECTED)} were rejected** after verification.")
            lines.append("")

            if critical:
                lines.append("### Critical Findings Requiring Immediate Attention")
                lines.append("")
                for f in critical:
                    lines.append(f"- **{f.title}** ({f.attack_class}) - {f.impact[:100]}...")
                lines.append("")

        if include_methodology:
            lines.append("## Methodology")
            lines.append("")
            lines.append("This audit followed a six-phase methodology:")
            lines.append("")
            lines.append("1. **Reconnaissance** - Architecture mapping, trust boundary identification, input surface enumeration, and prior evidence collection")
            lines.append("2. **Coverage-Led Hunting** - Deterministic coverage ledger generation, isolated hunter assignment per coverage unit, gap analysis via coverage critics")
            lines.append("3. **Candidate Validation** - Fresh verifiers assigned to each unique candidate to attempt disproof")
            lines.append("4. **Structured Output** - Findings organized by verdict (confirmed/needs_validation/rejected), schema validation")
            lines.append("5. **Independent Record Verification** - Independent agents verify final source claims; material replacements receive additional verification")
            lines.append("6. **Target-Neutral Reporting** - Reports derived solely from verified records and coverage ledger")
            lines.append("")

        # Coverage Summary
        lines.append("## Coverage Summary")
        lines.append("")
        total_units = coverage_data.get("summary", {}).get("total_units", 0)
        by_type = coverage_data.get("summary", {}).get("by_type", {})
        lines.append(f"**Total Coverage Units:** {total_units}")
        lines.append("")
        lines.append("### Coverage by Type")
        lines.append("")
        for unit_type, count in by_type.items():
            lines.append(f"- {unit_type}: {count}")
        lines.append("")

        # Confirmed Findings Summary
        confirmed = [f for f in findings if f.verdict == Verdict.CONFIRMED]
        if confirmed:
            lines.append("## Confirmed Vulnerabilities")
            lines.append("")
            lines.append(f"| ID | Title | Severity | Attack Class | Confidence |")
            lines.append(f"|----|-------|----------|--------------|------------|")
            for f in confirmed:
                lines.append(f"| {f.id} | {f.title} | {f.severity} | {f.attack_class} | {f.confidence:.2f} |")
            lines.append("")

        # Findings by Attack Class
        lines.append("## Findings by Attack Class")
        lines.append("")
        attack_class_counts = {}
        for f in findings:
            if f.verdict in (Verdict.CONFIRMED, Verdict.NEEDS_VALIDATION):
                attack_class_counts[f.attack_class] = attack_class_counts.get(f.attack_class, 0) + 1

        for attack_class, count in sorted(attack_class_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{attack_class}**: {count} finding(s)")
        lines.append("")

        return "\n".join(lines)

    def _generate_findings_detail_md(self, findings: list[Finding], target_name: str) -> str:
        """Generate FINDINGS-DETAIL.md"""
        lines = []

        lines.append(f"# Findings Detail: {target_name}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("")

        confirmed = [f for f in findings if f.verdict == Verdict.CONFIRMED]
        needs_validation = [f for f in findings if f.verdict == Verdict.NEEDS_VALIDATION]

        # Confirmed findings detail
        if confirmed:
            lines.append("## Confirmed Findings")
            lines.append("")
            for f in confirmed:
                lines.append(f"### {f.title} [{f.id}]")
                lines.append("")
                lines.append(f"**Severity:** {f.severity.upper()}")
                lines.append(f"**Attack Class:** {f.attack_class}")
                lines.append(f"**Coverage Unit:** {f.coverage_unit_id}")
                lines.append(f"**Hunter:** {f.hunter_id}")
                lines.append(f"**Verifier:** {f.verifier_id or 'N/A'}")
                lines.append(f"**Confidence:** {f.confidence:.2f}")
                lines.append(f"**Verification Status:** {f.verification_status}")
                lines.append("")
                lines.append(f"**Description:**")
                lines.append(f"{f.description}")
                lines.append("")
                if f.evidence:
                    lines.append("**Evidence:**")
                    for e in f.evidence:
                        lines.append(f"- {e}")
                    lines.append("")
                if f.steps_to_reproduce:
                    lines.append("**Steps to Reproduce:**")
                    for i, step in enumerate(f.steps_to_reproduce, 1):
                        lines.append(f"{i}. {step}")
                    lines.append("")
                if f.impact:
                    lines.append(f"**Impact:** {f.impact}")
                    lines.append("")
                if f.remediation:
                    lines.append(f"**Remediation:** {f.remediation}")
                    lines.append("")
                if f.references:
                    lines.append("**References:**")
                    for ref in f.references:
                        lines.append(f"- {ref}")
                    lines.append("")
                if f.verification_notes:
                    lines.append(f"**Verification Notes:** {f.verification_notes}")
                    lines.append("")
                lines.append("---")
                lines.append("")

        # Needs validation detail
        if needs_validation:
            lines.append("## Findings Requiring Validation")
            lines.append("")
            for f in needs_validation:
                lines.append(f"### {f.title} [{f.id}]")
                lines.append("")
                lines.append(f"**Severity:** {f.severity.upper()}")
                lines.append(f"**Attack Class:** {f.attack_class}")
                lines.append(f"**Coverage Unit:** {f.coverage_unit_id}")
                lines.append(f"**Hunter:** {f.hunter_id}")
                lines.append(f"**Confidence:** {f.confidence:.2f}")
                lines.append("")
                lines.append(f"**Description:**")
                lines.append(f"{f.description}")
                lines.append("")
                if f.evidence:
                    lines.append("**Evidence:**")
                    for e in f.evidence:
                        lines.append(f"- {e}")
                    lines.append("")
                if f.verification_notes:
                    lines.append(f"**Verification Notes:** {f.verification_notes}")
                    lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def _generate_needs_validation_md(self, findings: list[Finding], target_name: str) -> str:
        """Generate NEEDS-VALIDATION.md"""
        lines = []

        lines.append(f"# Findings Needing Validation: {target_name}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("")

        needs_validation = [f for f in findings if f.verdict == Verdict.NEEDS_VALIDATION]

        if not needs_validation:
            lines.append("No findings require validation at this time.")
            return "\n".join(lines)

        lines.append(f"## Summary: {len(needs_validation)} Findings Require Additional Validation")
        lines.append("")

        # Group by attack class
        by_attack_class = {}
        for f in needs_validation:
            if f.attack_class not in by_attack_class:
                by_attack_class[f.attack_class] = []
            by_attack_class[f.attack_class].append(f)

        for attack_class, class_findings in by_attack_class.items():
            lines.append(f"## {attack_class.replace('_', ' ').title()} ({len(class_findings)} findings)")
            lines.append("")
            for f in class_findings:
                lines.append(f"### {f.title} [{f.id}]")
                lines.append("")
                lines.append(f"**Severity:** {f.severity.upper()}")
                lines.append(f"**Coverage Unit:** {f.coverage_unit_id}")
                lines.append(f"**Hunter:** {f.hunter_id}")
                lines.append(f"**Confidence:** {f.confidence:.2f}")
                lines.append("")
                lines.append(f"**Description:** {f.description}")
                lines.append("")
                if f.evidence:
                    lines.append("**Evidence:**")
                    for e in f.evidence:
                        lines.append(f"- {e}")
                    lines.append("")
                if f.steps_to_reproduce:
                    lines.append("**Suggested Validation Steps:**")
                    for i, step in enumerate(f.steps_to_reproduce, 1):
                        lines.append(f"{i}. {step}")
                    lines.append("")
                if f.verification_notes:
                    lines.append(f"**Previous Verification Notes:** {f.verification_notes}")
                    lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def _generate_summary_json(self, findings: list[Finding], coverage_data: dict) -> dict[str, Any]:
        """Generate summary JSON."""
        confirmed = [f for f in findings if f.verdict == Verdict.CONFIRMED]
        needs_validation = [f for f in findings if f.verdict == Verdict.NEEDS_VALIDATION]
        rejected = [f for f in findings if f.verdict == Verdict.REJECTED]

        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "target": "Target System",
            "findings": {
                "confirmed": len(confirmed),
                "needs_validation": len(needs_validation),
                "rejected": len(rejected),
                "by_severity": {
                    "critical": len([f for f in confirmed if f.severity == "critical"]),
                    "high": len([f for f in confirmed if f.severity == "high"]),
                    "medium": len([f for f in confirmed if f.severity == "medium"]),
                    "low": len([f for f in confirmed if f.severity == "low"]),
                },
                "by_attack_class": self._count_by_attack_class(findings),
            },
            "coverage": {
                "total_units": coverage_data.get("summary", {}).get("total_units", 0),
                "by_type": coverage_data.get("summary", {}).get("by_type", {}),
                "by_priority": coverage_data.get("summary", {}).get("by_priority", {}),
            },
            "verification": {
                "independently_verified": len([f for f in confirmed if f.verification_status == "verified"]),
                "with_verifier": len([f for f in findings if f.verifier_id]),
            },
        }

    def _count_by_attack_class(self, findings: list[Finding]) -> dict[str, int]:
        counts = {}
        for f in findings:
            if f.verdict in (Verdict.CONFIRMED, Verdict.NEEDS_VALIDATION):
                counts[f.attack_class] = counts.get(f.attack_class, 0) + 1
        return counts


from datetime import datetime