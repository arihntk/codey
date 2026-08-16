"""Tests for codey.agents.schemas — severity, findings, reports."""

from __future__ import annotations

import pytest

from codey.agents.schemas import (
    AgentReport,
    Finding,
    FindingCategory,
    ReviewSummary,
    Severity,
    aggregate_severity,
    severity_weight,
)


def test_severity_weights():
    assert severity_weight(Severity.CRITICAL) == 4
    assert severity_weight(Severity.HIGH) == 3
    assert severity_weight(Severity.MEDIUM) == 2
    assert severity_weight(Severity.LOW) == 1
    assert severity_weight(Severity.INFO) == 0


def test_aggregate_severity_empty_is_info():
    assert aggregate_severity([]) == Severity.INFO


def test_aggregate_severity_returns_max():
    fs = [
        Finding(category=FindingCategory.SECURITY, severity=Severity.LOW, title="a"),
        Finding(category=FindingCategory.SECURITY, severity=Severity.CRITICAL, title="b"),
        Finding(category=FindingCategory.SECURITY, severity=Severity.HIGH, title="c"),
    ]
    assert aggregate_severity(fs) == Severity.CRITICAL


def test_finding_defaults():
    f = Finding(category=FindingCategory.SECURITY, title="t")
    assert f.severity == Severity.INFO
    assert f.description == ""
    assert f.file_path is None
    assert f.line_start is None
    assert f.line_end is None
    assert f.evidence == ""
    assert f.recommendation == ""
    assert f.confidence == 1.0


def test_finding_confidence_bounds_enforced():
    with pytest.raises(Exception):  # pydantic ValidationError
        Finding(category=FindingCategory.SECURITY, title="t", confidence=1.5)
    with pytest.raises(Exception):
        Finding(category=FindingCategory.SECURITY, title="t", confidence=-0.1)


def test_agent_name_literal():
    # AgentName is a Literal; just sanity-check the values accepted by the report.
    for name in ("index", "security", "code_quality", "test", "codey"):
        r = AgentReport(agent=name, summary="s")
        assert r.agent == name


def test_agent_report_counts_and_grouping():
    f_hi = Finding(category=FindingCategory.SECURITY, severity=Severity.HIGH, title="h")
    f_lo = Finding(category=FindingCategory.SECURITY, severity=Severity.LOW, title="l")
    f_lo2 = Finding(category=FindingCategory.SECURITY, severity=Severity.LOW, title="l2")
    r = AgentReport(agent="security", summary="s", findings=[f_hi, f_lo, f_lo2])
    assert r.finding_count() == 3
    grouped = r.findings_by_severity()
    assert len(grouped[Severity.HIGH]) == 1
    assert len(grouped[Severity.LOW]) == 2


def test_agent_report_defaults():
    r = AgentReport(agent="index")
    assert r.status == "completed"
    assert r.summary == ""
    assert r.findings == []
    assert r.metadata == {}
    assert r.token_usage == 0
    assert r.error is None


def test_review_summary_defaults():
    r = ReviewSummary()
    assert r.overall_severity == Severity.INFO
    assert r.recommendation == "approve"
    assert r.total_findings == 0
    assert r.files_reviewed == []
    assert r.errors == []
    assert r.all_findings() == []


def test_review_summary_all_findings_aggregates():
    f1 = Finding(category=FindingCategory.SECURITY, severity=Severity.HIGH, title="a")
    f2 = Finding(category=FindingCategory.CODE_QUALITY, severity=Severity.LOW, title="b")
    r = ReviewSummary(
        agent_reports={
            "security": AgentReport(agent="security", findings=[f1]),
            "code_quality": AgentReport(agent="code_quality", findings=[f2]),
        },
    )
    assert {f.title for f in r.all_findings()} == {"a", "b"}
