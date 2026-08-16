"""Pydantic schemas for structured agent findings + shared LLM-JSON parsing."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "Severity",
    "FindingCategory",
    "Finding",
    "AgentReport",
    "ReviewSummary",
    "AgentName",
    "severity_weight",
    "aggregate_severity",
    "strip_code_fences",
    "llm_output_is_parseable",
    "parse_llm_findings",
]


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(StrEnum):
    SECURITY = "security"
    CODE_QUALITY = "code_quality"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    CORRECTNESS = "correctness"
    STYLE = "style"


AgentName = Literal["index", "security", "code_quality", "test", "codey"]


class Finding(BaseModel):
    category: FindingCategory
    severity: Severity = Severity.INFO
    title: str = Field(..., description="Short title of the finding")
    description: str = Field("", description="Detailed explanation")
    file_path: str | None = Field(None, description="File the finding relates to")
    line_start: int | None = None
    line_end: int | None = None
    evidence: str = Field("", description="Code snippet or tool output backing the finding")
    recommendation: str = Field("", description="Suggested fix or next step")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Agent confidence in this finding")


class AgentReport(BaseModel):
    agent: AgentName
    status: Literal["completed", "skipped", "error"] = "completed"
    summary: str = Field("", description="High-level summary of agent findings")
    findings: list[Finding] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    token_usage: int = Field(0, description="Approximate tokens consumed")
    error: str | None = None

    def finding_count(self) -> int:
        return len(self.findings)

    def findings_by_severity(self) -> dict[Severity, list[Finding]]:
        result: dict[Severity, list[Finding]] = {}
        for f in self.findings:
            result.setdefault(f.severity, []).append(f)
        return result


class ReviewSummary(BaseModel):
    overall_severity: Severity = Severity.INFO
    summary: str = Field("", description="Executive summary of the review")
    commit_hash: str = ""
    commit_message: str = ""
    files_reviewed: list[str] = Field(default_factory=list)
    dependent_files_checked: list[str] = Field(default_factory=list)
    agent_reports: dict[str, AgentReport] = Field(default_factory=dict)
    total_findings: int = 0
    recommendation: Literal["approve", "request_changes", "block"] = "approve"
    errors: list[str] = Field(default_factory=list)
    pruned_chunks: list[str] = Field(default_factory=list)

    def all_findings(self) -> list[Finding]:
        return [f for r in self.agent_reports.values() for f in r.findings]


_WEIGHTS = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1, Severity.INFO: 0}


def severity_weight(s: Severity) -> int:
    return _WEIGHTS.get(s, 0)


def aggregate_severity(findings: list[Finding]) -> Severity:
    return max((f.severity for f in findings), key=severity_weight, default=Severity.INFO)


# --- Shared LLM-JSON parsing -------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Remove an enclosing markdown ``` code fence, if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return t


def llm_output_is_parseable(text: str) -> bool:
    """True when *text* parses as a JSON array (possibly empty)."""
    try:
        json.loads(strip_code_fences(text))
        return True
    except json.JSONDecodeError:
        return False


def parse_llm_findings(text: str, category: FindingCategory) -> tuple[list[Finding], str | None]:
    """Parse LLM JSON output into ``Finding`` objects, defensively.

    Returns ``(findings, error)`` — malformed items (e.g. ``"severity": null``)
    are skipped rather than crashing the whole parse, and the count is returned
    as an error note. Unparseable top-level JSON yields ``([], None)``.
    """
    try:
        data = json.loads(strip_code_fences(text))
    except json.JSONDecodeError:
        return [], None
    if not isinstance(data, list):
        return [], "LLM output was not a JSON array"
    findings: list[Finding] = []
    malformed = 0
    for item in data:
        try:
            sev_raw = item.get("severity", "info")
            if sev_raw is None:
                raise ValueError("severity is null")
            conf_raw = item.get("confidence", 0.7)
            findings.append(Finding(
                category=category,
                severity=Severity(str(sev_raw).lower()),
                title=item.get("title", ""),
                description=item.get("description", ""),
                file_path=item.get("file_path"),
                line_start=item.get("line_start"),
                line_end=item.get("line_end"),
                evidence=item.get("evidence", ""),
                recommendation=item.get("recommendation", ""),
                confidence=float(conf_raw) if conf_raw is not None else 0.7,
            ))
        except (ValueError, TypeError):
            malformed += 1
    error = f"{malformed} malformed LLM finding(s) skipped" if malformed else None
    return findings, error
