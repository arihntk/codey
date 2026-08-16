"""Tests for the evaluation layer — scoring, matching, stub LLM, fake-mode e2e."""

from __future__ import annotations

from codey.agents.context import ReviewContext
from codey.agents.schemas import (
    AgentReport,
    Finding,
    FindingCategory,
    Severity,
)
from codey.evals.metrics import evidence_grounded, score_findings
from codey.evals.runner import StubLLM, run_evals
from codey.evals.scenarios import ExpectedIssue


def _finding(sev=Severity.HIGH, title="t", category=FindingCategory.SECURITY,
             file_path="a.py", line_start=1, confidence=1.0, evidence=""):
    return Finding(
        category=category, severity=sev, title=title, file_path=file_path,
        line_start=line_start, confidence=confidence, evidence=evidence,
    )


def _label(agent="security", sev=Severity.HIGH, category=FindingCategory.SECURITY,
           file_path="a.py", line_start=1, keywords=(), expect_absent=False):
    return ExpectedIssue(
        agent=agent, category=category, severity=sev,
        file_path=file_path, line_start=line_start, keywords=tuple(keywords),
        expect_absent=expect_absent,
    )


def _reports(agent, findings):
    return {agent: AgentReport(agent=agent, findings=list(findings))}


# ---------------------------------------------------------------------------
# matching + metrics
# ---------------------------------------------------------------------------

def test_score_tp_fp_fn():
    findings = [
        _finding(Severity.CRITICAL, title="OpenAI API key", line_start=3),
        _finding(Severity.HIGH, title="noise", line_start=9),
    ]
    expected = (_label(sev=Severity.CRITICAL, line_start=3, keywords=("OpenAI API key",)),)
    m = score_findings(_reports("security", findings), expected)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.precision == 0.5 and m.recall == 1.0 and m.f1 == 0.6666666666666666
    assert m.false_positives == [findings[1]]
    assert m.false_negatives == []


def test_score_missing_label_is_fn():
    m = score_findings(_reports("security", []), (_label(sev=Severity.CRITICAL),))
    assert (m.tp, m.fp, m.fn) == (0, 0, 1)
    assert m.recall == 0.0 and m.precision == 0.0 and m.f1 == 0.0


def test_score_severity_mismatch_counts_tp_but_calibration_penalised():
    expected = (_label(sev=Severity.CRITICAL, line_start=3),)
    m = score_findings(_reports("security", [_finding(Severity.HIGH, line_start=3)]), expected)
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)
    assert m.severity_exact == 0.0
    assert m.severity_within_one == 1.0


def test_score_negative_label_violation_is_fp():
    expected = (_label(sev=Severity.HIGH, line_start=3, keywords=("password",), expect_absent=True),)
    m = score_findings(_reports("security", [_finding(Severity.HIGH, title="password", line_start=3)]), expected)
    assert (m.tp, m.fp, m.fn) == (0, 1, 0)
    assert m.false_positives == [_finding(Severity.HIGH, title="password", line_start=3)]


def test_score_agent_and_category_must_match():
    wrong_agent = _reports("code_quality", [_finding(Severity.CRITICAL)])
    m = score_findings(wrong_agent, (_label(sev=Severity.CRITICAL),))
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_score_line_range_matching():
    expected = (
        ExpectedIssue("security", FindingCategory.SECURITY, Severity.HIGH, "a.py", 5, 10, ("bug",)),
    )
    m = score_findings(_reports("security", [_finding(Severity.HIGH, title="a bug", line_start=7)]), expected)
    assert (m.tp, m.fp, m.fn) == (1, 0, 0)


def test_score_info_findings_ignored():
    expected = ()
    m = score_findings(_reports("security", [_finding(Severity.INFO)]), expected)
    assert (m.tp, m.fp, m.fn) == (0, 0, 0)
    assert m.evidence_ratio is None


def test_score_expect_clean_counts_every_non_info_finding():
    expected = ()
    m = score_findings(
        _reports("security", [_finding(Severity.MEDIUM, title="x"), _finding(Severity.HIGH, title="y")]),
        expected, expect_clean=True,
    )
    assert (m.tp, m.fp, m.fn) == (0, 2, 0)


def test_score_evidence_ratio():
    expected = ()
    m = score_findings(
        _reports("security", [_finding(Severity.HIGH, evidence="code"), _finding(Severity.HIGH, evidence="")]),
        expected,
    )
    assert m.evidence_ratio == 0.5


def test_evidence_grounded_from_diff():
    ctx = ReviewContext(
        repo_path="/r", git_hash="h", commit_message="m",
        raw_full_diff="+def bad():\n+    return 'secret'\n",
        file_sources={},
    )
    f = _finding(Severity.HIGH, evidence="    return 'secret'", file_path="a.py")
    assert evidence_grounded(f, ctx) is True


def test_evidence_grounded_from_source_with_window_prefix():
    ctx = ReviewContext(
        repo_path="/r", git_hash="h", commit_message="m", raw_full_diff="",
        file_sources={"a.py": "line one\nline two\nline three\n"},
    )
    f = _finding(Severity.HIGH, evidence="     2 | line two", file_path="a.py")
    assert evidence_grounded(f, ctx) is True


def test_evidence_grounded_false_when_absent():
    ctx = ReviewContext(repo_path="/r", git_hash="h", commit_message="m", raw_full_diff="", file_sources={})
    assert evidence_grounded(_finding(Severity.HIGH, evidence="totally made up", file_path="a.py"), ctx) is False
    assert evidence_grounded(_finding(Severity.HIGH, evidence="", file_path="a.py"), ctx) is False


# ---------------------------------------------------------------------------
# stub LLM
# ---------------------------------------------------------------------------

class _Msg:
    def __init__(self, type_, content):
        self.type = type_
        self.content = content


def test_stub_llm_routes_by_system_prompt():
    llm = StubLLM()
    assert llm.invoke([_Msg("system", "You are a senior security analyst.")]).content == "[]"
    assert llm.invoke([_Msg("system", "You are a senior code quality reviewer.")]).content == "[]"
    assert llm.invoke([_Msg("system", "You are a test expert.")]).content == "[]"
    synth = llm.invoke([_Msg("system", "You are Codey, the lead code reviewer.")]).content
    assert '"recommendation": "approve"' in synth
    judge = llm.invoke([_Msg("system", "You are an impartial evaluator of AI code reviews.")]).content
    assert '"overall": 5.0' in judge
    assert llm.invoke([_Msg("human", "hello")]).content == "[]"


# ---------------------------------------------------------------------------
# fake-mode e2e (no API key required)
# ---------------------------------------------------------------------------

def test_fake_mode_excludes_llm_only_and_tests_enabled():
    report = run_evals(mode="fake")
    ids = {s.id for s in report.scenarios}
    assert "secret-openai-key" in ids
    assert "injection-sql" not in ids
    assert "test-failure" not in ids
    assert "dependent-files-affected" not in ids


def test_fake_mode_secret_scenario_scores_perfectly():
    report = run_evals(mode="fake", scenario_ids=["secret-openai-key"])
    s = report.scenarios[0]
    assert s.status == "completed"
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0
    assert s.severity_exact == 1.0
    assert s.recommendation == "block" and s.recommendation_match is True


def test_fake_mode_clean_commit_approves_without_findings():
    report = run_evals(mode="fake", scenario_ids=["clean-commit"])
    s = report.scenarios[0]
    assert s.fp == 0 and s.fn == 0
    assert s.recommendation == "approve" and s.recommendation_match is True


def test_fake_mode_non_head_and_pruning():
    report = run_evals(mode="fake", scenario_ids=["non-head-commit", "large-diff-budget"])
    by_id = {s.id: s for s in report.scenarios}
    assert by_id["non-head-commit"].recommendation == "block"
    assert by_id["non-head-commit"].tp == 1
    assert by_id["large-diff-budget"].pruned_chunks > 0
    assert by_id["large-diff-budget"].pipeline_checks["pruned"] is True


def test_fake_mode_judge_forced_runs_stub_judge():
    report = run_evals(mode="fake", judge=True, scenario_ids=["secret-openai-key"])
    s = report.scenarios[0]
    assert s.judge is not None
    assert s.judge.overall == 5.0
    assert report.judge_enabled is True
