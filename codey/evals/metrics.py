"""Scoring — match agent findings against ground-truth labels, compute quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from re import match as _re_match

from codey.agents.context import ReviewContext
from codey.agents.schemas import AgentReport, Finding, Severity, severity_weight
from codey.evals.scenarios import ExpectedIssue

__all__ = ["ScenarioMetrics", "score_findings", "evidence_grounded"]

_INFO = Severity.INFO


@dataclass
class ScenarioMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    severity_exact: float | None = None  # over matched pairs
    severity_within_one: float | None = None
    evidence_ratio: float | None = None  # non-INFO findings with non-empty evidence
    false_positives: list[Finding] = field(default_factory=list)
    false_negatives: list[ExpectedIssue] = field(default_factory=list)


def _label_agent_name(finding: Finding, reports: dict[str, AgentReport]) -> str:
    for name, report in reports.items():
        if any(f is finding for f in report.findings):
            return name
    return ""


def _matches(label: ExpectedIssue, finding: Finding, *, agent: str) -> bool:
    if finding.severity == _INFO:
        return False  # informational findings never count towards precision/recall
    if label.agent != agent:
        return False
    if label.category != finding.category:
        return False
    if label.file_path and finding.file_path != label.file_path:
        return False
    if label.line_start is not None:
        if finding.line_start is None:
            return False
        hi = label.line_end or label.line_start
        if not (label.line_start <= finding.line_start <= hi):
            return False
    if label.keywords:
        hay = f"{finding.title}\n{finding.evidence}\n{finding.description}".lower()
        if not any(k.lower() in hay for k in label.keywords):
            return False
    return True


def _closeness(label: ExpectedIssue, finding: Finding) -> tuple:
    exact_line = 0 if (label.line_start is not None and finding.line_start == label.line_start) else 1
    dist = abs((finding.line_start or 0) - (label.line_start or 0))
    return (exact_line, dist, -severity_weight(finding.severity))


def score_findings(
    reports: dict[str, AgentReport],
    expected: tuple[ExpectedIssue, ...],
    *,
    expect_clean: bool = False,
) -> ScenarioMetrics:
    findings = [f for r in reports.values() for f in r.findings if f.severity != _INFO]
    positive = [e for e in expected if not e.expect_absent]
    negative = [e for e in expected if e.expect_absent]

    matched_ids: set[int] = set()
    pairs: list[tuple[ExpectedIssue, Finding]] = []
    for label in sorted(positive, key=lambda e: -severity_weight(e.severity)):
        candidates = [
            f for f in findings
            if id(f) not in matched_ids and _matches(label, f, agent=_label_agent_name(f, reports))
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda f: _closeness(label, f))
        matched_ids.add(id(best))
        pairs.append((label, best))

    matched_labels = {p[0] for p in pairs}
    fp_findings = [f for f in findings if id(f) not in matched_ids]
    if expect_clean:
        fp_findings = list(findings)
    # Negative labels violated by a finding that already matched a positive label
    # are an extra false positive (the finding is TP *and* contradicts a negative).
    neg_violations = sum(
        1 for label in negative
        if any(_matches(label, f, agent=_label_agent_name(f, reports)) for f in findings if id(f) in matched_ids)
    )
    fn_labels = [e for e in positive if e not in matched_labels]

    tp, fp, fn = len(pairs), len(fp_findings) + neg_violations, len(fn_labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if pairs:
        exact = sum(1 for label, f in pairs if f.severity == label.severity) / len(pairs)
        within = sum(
            1 for label, f in pairs
            if abs(severity_weight(f.severity) - severity_weight(label.severity)) <= 1
        ) / len(pairs)
    else:
        exact = within = None

    evidence_ratio = None
    if findings:
        evidence_ratio = sum(1 for f in findings if f.evidence.strip()) / len(findings)

    return ScenarioMetrics(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        severity_exact=exact, severity_within_one=within,
        evidence_ratio=evidence_ratio,
        false_positives=fp_findings,
        false_negatives=fn_labels,
    )


def _evidence_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _plain_line(line: str) -> str:
    """Strip the ``NNNN | `` window prefix attached by ``attach_evidence``."""
    m = _re_match(r"\d+\s*\|\s*(.*)$", line)
    return m.group(1).strip() if m else line


def evidence_grounded(finding: Finding, ctx: ReviewContext) -> bool:
    """True when at least one evidence line appears verbatim in the raw diff/source."""
    if not finding.evidence.strip() or not finding.file_path:
        return False
    hay = (ctx.raw_full_diff or "") + "\n" + (ctx.file_sources.get(finding.file_path) or "")
    for line in _evidence_lines(finding.evidence):
        plain = _plain_line(line)
        if plain and plain in hay:
            return True
    return False
