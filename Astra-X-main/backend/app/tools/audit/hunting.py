"""Hunting tool for Phase 2 - coverage-led hunting with isolated hunters and coverage critics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.tools.audit.models import (
    Architecture,
    CoverageCriticResult,
    CoverageUnit,
    CoverageUnitType,
    Finding,
    Hunter,
    HunterStatus,
    Verdict,
)
from app.tools.base import Tool
from app.tools.context import ToolContext
from app.tools.models import ToolParameter, ToolSchema
from app.tools.result import ToolResult


class HuntingTool(Tool):
    """Phase 2: Coverage-led hunting - assign isolated hunters from ledger units, record checks, use coverage critics to find gaps."""

    @property
    def name(self) -> str:
        return "hunting"

    @property
    def description(self) -> str:
        return "Perform Phase 2 coverage-led hunting: assign isolated hunters from coverage ledger units, record their checks, and use coverage critics to find gaps."

    @property
    def capabilities(self) -> list[str]:
        return ["hunting", "coverage_led_hunting", "gap_analysis"]

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
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
                    description="Directory to write findings and hunter records",
                    required=True,
                ),
                ToolParameter(
                    name="hunter_count",
                    type_="integer",
                    description="Number of parallel hunters to simulate",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="attack_class_prompts",
                    type_="object",
                    description="Custom attack prompts per attack class",
                    required=False,
                    default={},
                ),
                ToolParameter(
                    name="run_coverage_critic",
                    type_="boolean",
                    description="Whether to run coverage critic after hunting",
                    required=False,
                    default=True,
                ),
            ],
        )

    async def _execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        coverage_ledger_path = kwargs.get("coverage_ledger_path", "")
        architecture_path = kwargs.get("architecture_path", "")
        output_dir = kwargs.get("output_dir", "")
        hunter_count = kwargs.get("hunter_count", 5)
        attack_class_prompts = kwargs.get("attack_class_prompts", {})
        run_coverage_critic = kwargs.get("run_coverage_critic", True)

        if not coverage_ledger_path or not output_dir:
            return ToolResult(success=False, error="coverage_ledger_path and output_dir are required")

        # Load coverage ledger
        ledger_data = json.loads(Path(coverage_ledger_path).read_text(encoding="utf-8"))
        coverage_units = self._load_coverage_units(ledger_data)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Assign hunters to coverage units
        hunters = self._assign_hunters(coverage_units, hunter_count)

        # Simulate hunting for each hunter
        all_findings = []
        for hunter in hunters:
            unit = next((u for u in coverage_units if u.id == hunter.coverage_unit_id), None)
            if unit:
                findings = await self._run_hunter(hunter, unit, attack_class_prompts, context)
                all_findings.extend(findings)
                hunter.findings = [f.id for f in findings]
                hunter.status = HunterStatus.COMPLETED

        # Run coverage critic to find gaps
        critic_results = []
        if run_coverage_critic:
            critic_results = self._run_coverage_critic(coverage_units, all_findings, attack_class_prompts)

        # Write findings.json (initial)
        findings_data = self._findings_to_json(all_findings, Verdict.NEEDS_VALIDATION)
        (output_path / "findings.json").write_text(json.dumps(findings_data, indent=2), encoding="utf-8")

        # Write hunter records
        hunters_data = self._hunters_to_json(hunters)
        (output_path / "hunters.json").write_text(json.dumps(hunters_data, indent=2), encoding="utf-8")

        # Write coverage critic results
        critic_data = self._critic_results_to_json(critic_results)
        (output_path / "coverage-critic.json").write_text(json.dumps(critic_data, indent=2), encoding="utf-8")

        # Update coverage ledger with findings
        updated_ledger = self._update_coverage_ledger(ledger_data, hunters, all_findings)
        (output_path / "coverage-ledger.json").write_text(json.dumps(updated_ledger, indent=2), encoding="utf-8")

        return ToolResult(
            success=True,
            output=f"Hunting complete. {len(all_findings)} findings generated, {len(hunters)} hunters executed.",
            metadata={
                "hunters_executed": len(hunters),
                "findings_generated": len(all_findings),
                "coverage_critic_gaps": sum(len(c.gaps_found) for c in critic_results),
                "findings_json": str(output_path / "findings.json"),
                "hunters_json": str(output_path / "hunters.json"),
                "critic_json": str(output_path / "coverage-critic.json"),
            },
        )

    def _load_coverage_units(self, ledger_data: dict) -> list[CoverageUnit]:
        """Load coverage units from JSON."""
        units = []
        for cu_data in ledger_data.get("coverage_units", []):
            units.append(
                CoverageUnit(
                    id=cu_data.get("id", ""),
                    type=CoverageUnitType(cu_data.get("type", "attack_class")),
                    name=cu_data.get("name", ""),
                    description=cu_data.get("description", ""),
                    reference_ids=cu_data.get("reference_ids", []),
                    attack_classes=cu_data.get("attack_classes", []),
                    priority=cu_data.get("priority", 1),
                    status=HunterStatus(cu_data.get("status", "pending")),
                )
            )
        return units

    def _assign_hunters(self, coverage_units: list[CoverageUnit], hunter_count: int) -> list[Hunter]:
        """Assign hunters to coverage units, prioritizing by priority."""
        # Sort by priority (higher first)
        sorted_units = sorted(coverage_units, key=lambda u: -u.priority)

        hunters = []
        for i, unit in enumerate(sorted_units):
            if i >= hunter_count * 2:  # Limit total hunters
                break
            hunter = Hunter(
                name=f"hunter_{unit.type.value}_{i}",
                specialization=unit.attack_classes,
                coverage_unit_id=unit.id,
                status=HunterStatus.IN_PROGRESS,
            )
            hunters.append(hunter)
            unit.assigned_hunter = hunter.id
            unit.status = HunterStatus.IN_PROGRESS

        return hunters

    async def _run_hunter(
        self,
        hunter: Hunter,
        unit: CoverageUnit,
        attack_class_prompts: dict[str, str],
        context: ToolContext,
    ) -> list[Finding]:
        """Simulate a hunter running checks against a coverage unit."""
        findings = []

        for attack_class in unit.attack_classes:
            # Get custom prompt or use default
            prompt = attack_class_prompts.get(attack_class, self._get_default_prompt(attack_class))

            # Simulate the check - in real implementation this would use LLM
            check_result = self._simulate_check(hunter, unit, attack_class, prompt)

            if check_result["candidate_found"]:
                finding = Finding(
                    title=f"{attack_class.replace('_', ' ').title()} in {unit.name}",
                    description=check_result["description"],
                    verdict=Verdict.NEEDS_VALIDATION,
                    severity=check_result["severity"],
                    attack_class=attack_class,
                    coverage_unit_id=unit.id,
                    hunter_id=hunter.id,
                    evidence=check_result["evidence"],
                    steps_to_reproduce=check_result["steps"],
                    impact=check_result["impact"],
                    remediation=check_result["remediation"],
                    confidence=check_result["confidence"],
                )
                findings.append(finding)

            hunter.checks_performed.append(f"{attack_class}: {check_result['result']}")

        return findings

    def _get_default_prompt(self, attack_class: str) -> str:
        """Get default hunting prompt for an attack class."""
        prompts = {
            "injection": "Test all input surfaces for SQL injection, command injection, LDAP injection, and NoSQL injection vulnerabilities.",
            "auth_bypass": "Test authentication mechanisms for bypasses including password reset, session fixation, JWT weaknesses, and MFA bypass.",
            "idor": "Test for Insecure Direct Object References by manipulating object identifiers in API calls.",
            "xss": "Test for Cross-Site Scripting in all reflected, stored, and DOM-based contexts.",
            "csrf": "Test for Cross-Site Request Forgery by checking anti-CSRF token validation.",
            "ssrf": "Test for Server-Side Request Forgery by attempting to access internal services.",
            "xxe": "Test for XML External Entity injection in XML parsers.",
            "deserialization": "Test for insecure deserialization in all data formats (JSON, XML, YAML, pickle, etc.).",
            "path_traversal": "Test for directory traversal in file operations.",
            "privilege_escalation": "Test for vertical and horizontal privilege escalation paths.",
            "race_condition": "Test for TOCTOU and other race conditions in critical operations.",
            "side_channel": "Test for timing attacks, cache attacks, and other side channels.",
            "supply_chain": "Test for dependency confusion, malicious packages, and build pipeline weaknesses.",
            "configuration": "Test for insecure default configurations, debug endpoints, and exposed secrets.",
            "crypto": "Test for weak cryptography, hardcoded keys, and improper key management.",
        }
        return prompts.get(attack_class, f"Test for {attack_class} vulnerabilities.")

    def _simulate_check(
        self, hunter: Hunter, unit: CoverageUnit, attack_class: str, prompt: str
    ) -> dict[str, Any]:
        """Simulate a security check. In reality this would use an LLM to analyze code."""
        # This is a simulation - real implementation would do actual analysis
        import random

        # Simulate finding candidates based on priority and attack class
        candidate_probability = min(0.3 + (unit.priority * 0.1), 0.8)
        candidate_found = random.random() < candidate_probability

        if not candidate_found:
            return {
                "candidate_found": False,
                "result": "No candidate vulnerability found",
                "description": "",
                "severity": "info",
                "evidence": [],
                "steps": [],
                "impact": "",
                "remediation": "",
                "confidence": 0.0,
            }

        severities = {"injection": "high", "auth_bypass": "critical", "idor": "medium", "xss": "medium"}
        severity = severities.get(attack_class, "medium")

        return {
            "candidate_found": True,
            "result": f"Candidate {attack_class} vulnerability identified",
            "description": f"Potential {attack_class} vulnerability found in {unit.name}. {prompt}",
            "severity": severity,
            "evidence": [f"Code pattern matching {attack_class} found in {unit.reference_ids}"],
            "steps": [f"1. Identify {attack_class} entry point", f"2. Craft malicious payload", f"3. Observe vulnerable behavior"],
            "impact": f"Successful {attack_class} could lead to data breach, authentication bypass, or system compromise.",
            "remediation": f"Implement proper input validation, output encoding, and security controls for {attack_class}.",
            "confidence": 0.7,
        }

    def _run_coverage_critic(
        self,
        coverage_units: list[CoverageUnit],
        findings: list[Finding],
        attack_class_prompts: dict[str, str],
    ) -> list[CoverageCriticResult]:
        """Run coverage critic to find gaps in hunting coverage."""
        results = []

        # Group findings by coverage unit
        findings_by_unit = {}
        for f in findings:
            if f.coverage_unit_id not in findings_by_unit:
                findings_by_unit[f.coverage_unit_id] = []
            findings_by_unit[f.coverage_unit_id].append(f)

        for unit in coverage_units:
            unit_findings = findings_by_unit.get(unit.id, [])
            covered_attack_classes = {f.attack_class for f in unit_findings}
            missing_attack_classes = set(unit.attack_classes) - covered_attack_classes

            # Check for gaps in attack classes
            gaps = []
            if missing_attack_classes:
                gaps.append(f"Missing coverage for attack classes: {', '.join(missing_attack_classes)}")

            # Check if high-priority units have findings
            if unit.priority >= 4 and not unit_findings:
                gaps.append(f"High-priority unit {unit.name} has no findings - may need deeper analysis")

            # Check for attack classes not in any unit
            all_unit_attack_classes = set()
            for u in coverage_units:
                all_unit_attack_classes.update(u.attack_classes)

            provided_attack_classes = set(attack_class_prompts.keys())
            missing_from_ledger = provided_attack_classes - all_unit_attack_classes
            if missing_from_ledger:
                gaps.append(f"Attack classes in prompts but not in ledger: {', '.join(missing_from_ledger)}")

            results.append(
                CoverageCriticResult(
                    coverage_unit_id=unit.id,
                    gaps_found=gaps,
                    missing_attack_classes=list(missing_attack_classes),
                    recommended_new_units=[
                        f"Add coverage for {ac}" for ac in missing_attack_classes
                    ],
                    confidence=0.8 if not gaps else 0.5,
                )
            )

        return results

    def _findings_to_json(self, findings: list[Finding], default_verdict: Verdict) -> dict[str, Any]:
        """Convert findings to JSON format."""
        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "findings": [
                {
                    "id": f.id,
                    "title": f.title,
                    "description": f.description,
                    "verdict": f.verdict.value,
                    "severity": f.severity,
                    "attack_class": f.attack_class,
                    "coverage_unit_id": f.coverage_unit_id,
                    "hunter_id": f.hunter_id,
                    "evidence": f.evidence,
                    "steps_to_reproduce": f.steps_to_reproduce,
                    "impact": f.impact,
                    "remediation": f.remediation,
                    "references": f.references,
                    "confidence": f.confidence,
                    "verifier_id": f.verifier_id,
                    "verification_status": f.verification_status.value,
                    "verification_notes": f.verification_notes,
                }
                for f in findings
            ],
            "summary": {
                "total": len(findings),
                "by_verdict": self._count_by_verdict(findings),
                "by_severity": self._count_by_severity(findings),
                "by_attack_class": self._count_by_attack_class(findings),
            },
        }

    def _hunters_to_json(self, hunters: list[Hunter]) -> dict[str, Any]:
        """Convert hunters to JSON format."""
        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "hunters": [
                {
                    "id": h.id,
                    "name": h.name,
                    "specialization": h.specialization,
                    "coverage_unit_id": h.coverage_unit_id,
                    "status": h.status.value,
                    "checks_performed": h.checks_performed,
                    "findings": h.findings,
                }
                for h in hunters
            ],
        }

    def _critic_results_to_json(self, results: list[CoverageCriticResult]) -> dict[str, Any]:
        """Convert coverage critic results to JSON."""
        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "critic_results": [
                {
                    "coverage_unit_id": r.coverage_unit_id,
                    "gaps_found": r.gaps_found,
                    "missing_attack_classes": r.missing_attack_classes,
                    "recommended_new_units": r.recommended_new_units,
                    "confidence": r.confidence,
                }
                for r in results
            ],
        }

    def _update_coverage_ledger(
        self, ledger_data: dict, hunters: list[Hunter], findings: list[Finding]
    ) -> dict:
        """Update coverage ledger with hunter results."""
        findings_by_unit = {}
        for f in findings:
            if f.coverage_unit_id not in findings_by_unit:
                findings_by_unit[f.coverage_unit_id] = []
            findings_by_unit[f.coverage_unit_id].append(f.id)

        for cu in ledger_data.get("coverage_units", []):
            if cu["id"] in findings_by_unit:
                cu["findings"] = findings_by_unit[cu["id"]]
                cu["status"] = "completed"
            elif cu.get("assigned_hunter"):
                cu["status"] = "completed"

        ledger_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        return ledger_data

    def _count_by_verdict(self, findings: list[Finding]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.verdict.value] = counts.get(f.verdict.value, 0) + 1
        return counts

    def _count_by_severity(self, findings: list[Finding]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    def _count_by_attack_class(self, findings: list[Finding]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in findings:
            counts[f.attack_class] = counts.get(f.attack_class, 0) + 1
        return counts


from datetime import datetime