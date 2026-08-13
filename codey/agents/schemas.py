"""Pydantic schemas for structured agent findings.

Every agent emits an ``AgentReport`` containing a list of ``Finding`` objects,
each with severity, evidence, and file/line references.  The orchestrator
synthesizes these into a ``ReviewReport``.
"""

from __future__ import annotations

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


AgentName = Literal[
    "index", "security", "code_quality", "test", "codey"
]


class Finding(BaseModel):
    """A single structured finding from an agent."""

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
    """Standalone report emitted by a single agent."""

    agent: AgentName
    status: Literal["completed", "skipped", "error"] = "completed"
    summary: str = Field("", description="High-level summary of agent findings")
    findings: list[Finding] = Field(default_factory=list)
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Agent-specific metadata (tool output, counts, etc.)",
    )
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
    """The orchestrator's final synthesized review."""

    overall_severity: Severity = Severity.INFO
    summary: str = Field("", description="Executive summary of the review")
    commit_hash: str = ""
    commit_message: str = ""
    files_reviewed: list[str] = Field(default_factory=list)
    dependent_files_checked: list[str] = Field(default_factory=list)
    agent_reports: dict[str, AgentReport] = Field(default_factory=dict)
    total_findings: int = 0
    recommendation: Literal["approve", "request_changes", "block"] = "approve"

    def all_findings(self) -> list[Finding]:
        out: list[Finding] = []
        for r in self.agent_reports.values():
            out.extend(r.findings)
        return out


def severity_weight(s: Severity) -> int:
    return {
        Severity.CRITICAL: 4,
        Severity.HIGH: 3,
        Severity.MEDIUM: 2,
        Severity.LOW: 1,
        Severity.INFO: 0,
    }.get(s, 0)


def aggregate_severity(findings: list[Finding]) -> Severity:
    """Compute the maximum severity across findings."""
    if not findings:
        return Severity.INFO
    return max(findings, key=lambda f: severity_weight(f.severity)).severity
