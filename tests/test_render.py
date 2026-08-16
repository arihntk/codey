"""Tests for codey.render and codey.progress."""

from __future__ import annotations

from rich.console import Console

from codey.agents.schemas import (
    AgentReport,
    Finding,
    FindingCategory,
    ReviewSummary,
    Severity,
)
from codey.progress import ProgressEmitter, _agent_labels, make_callback
from codey.render.report import (
    print_finding_table,
    render_review,
    severity_style,
)


def _console():
    return Console(record=True, width=120, force_terminal=False)


def test_severity_style_mapping():
    assert severity_style(Severity.CRITICAL) == "bold red on red"
    assert severity_style(Severity.HIGH) == "bold red"
    assert severity_style(Severity.INFO) == "dim"


def test_render_review_smoke():
    review = ReviewSummary(
        overall_severity=Severity.HIGH,
        summary="# Summary\nbody",
        commit_hash="abc123",
        commit_message="m",
        recommendation="request_changes",
        agent_reports={
            "security": AgentReport(agent="security", summary="found 1", findings=[
                Finding(category=FindingCategory.SECURITY, severity=Severity.HIGH, title="SQLi")
            ]),
        },
        total_findings=1,
    )
    console = _console()
    render_review(review, console=console)
    text = console.export_text()
    assert "Codey Review" in text
    assert "request_changes" in text.replace(" ", "").lower() or "Request Changes" in text


def test_render_review_with_errors_and_pruned():
    review = ReviewSummary(
        summary="s",
        errors=["[security] failed"],
        pruned_chunks=["a.py:1-10", "b.py:2-3"],
        agent_reports={},
    )
    console = _console()
    render_review(review, console=console)
    text = console.export_text()
    assert "Agent errors" in text
    assert "Pruned" in text


def test_print_finding_table_empty():
    console = _console()
    print_finding_table([], console=console)
    assert "No findings" in console.export_text()


def test_print_finding_table_sorts_by_severity():
    findings = [
        Finding(category=FindingCategory.SECURITY, severity=Severity.LOW, title="low"),
        Finding(category=FindingCategory.SECURITY, severity=Severity.CRITICAL, title="crit"),
    ]
    console = _console()
    print_finding_table(findings, console=console)
    text = console.export_text()
    assert text.index("crit") < text.index("low")


def test_progress_emitter_disabled_is_silent():
    console = _console()
    emitter = ProgressEmitter(console=console, enabled=False)
    emitter.emit("hello")
    emitter.emit_agent_start("security")
    emitter.emit_agent_done("security", findings=2)
    emitter.done()
    assert console.export_text() == ""


def test_progress_emitter_enabled_prints():
    console = _console()
    emitter = ProgressEmitter(console=console, enabled=True)
    emitter.emit("hello")
    emitter.done()
    assert "hello" in console.export_text()


def test_agent_labels_fallback():
    labels = _agent_labels()
    assert labels["index"] == "Indexing repository"
    assert labels["security"] == "Running security analysis"


def test_make_callback_emits_for_known_nodes():
    console = _console()
    emitter = ProgressEmitter(console=console, enabled=True)
    cb = make_callback(emitter)
    cb({"security": {"agent_reports": {"security": AgentReport(agent="security", findings=[
        Finding(category=FindingCategory.SECURITY, title="x"),
    ])}}})
    assert "Running security analysis" in console.export_text()


def test_make_callback_codey_node():
    console = _console()
    emitter = ProgressEmitter(console=console, enabled=True)
    cb = make_callback(emitter)
    cb({"codey": {}})
    assert "Synthesising final review" in console.export_text()
