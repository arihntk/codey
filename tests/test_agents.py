"""Tests for codey.agents — orchestrator, security, quality, test, index, evidence."""

from __future__ import annotations

import json

from codey.agents.code_quality_agent import run_code_quality_agent
from codey.agents.codey_agent import (
    _default_recommendation,
    _degrade_recommendation,
    _rec_rank,
    run_codey_agent,
)
from codey.agents.context import DiffChunk, ReviewContext
from codey.agents.evidence import attach_evidence
from codey.agents.index_agent import run_index_agent
from codey.agents.schemas import (
    AgentReport,
    Finding,
    FindingCategory,
    Severity,
)
from codey.agents.security_agent import (
    _diff_truncated_note,
    _hardcoded_summary,
    _llm_output_is_parseable,
    _parse_bandit_json,
    _parse_gitleaks_json,
    _parse_llm_findings,
    _parse_semgrep_json,
    _parse_tool_results,
    _should_skip,
    _strip_code_fences,
    run_security_agent,
)
from codey.agents.test_agent import _detect_frameworks, run_test_agent
from codey.cache.ast_cache import CacheDB
from codey.index.indexer import index_repository
from tests.conftest import FakeLLM, commit, init_repo


def _finding(sev=Severity.INFO, title="t", confidence=1.0):
    return Finding(category=FindingCategory.SECURITY, severity=sev, title=title, confidence=confidence)


def _ctx(**kw):
    defaults = dict(
        repo_path=__import__("pathlib").Path("/repo"),
        git_hash="h",
        commit_message="m",
    )
    defaults.update(kw)
    return ReviewContext(**defaults)


# ---------------------------------------------------------------------------
# codey_agent (orchestrator)
# ---------------------------------------------------------------------------

def test_rec_rank_ordering():
    assert _rec_rank("block") > _rec_rank("request_changes") > _rec_rank("approve")
    assert _rec_rank("bogus") == 0


def test_degrade_recommendation():
    assert _degrade_recommendation("approve") == "request_changes"
    assert _degrade_recommendation("request_changes") == "request_changes"
    assert _degrade_recommendation("block") == "block"


def test_default_recommendation_levels():
    assert _default_recommendation([]) == "approve"
    assert _default_recommendation([_finding(Severity.LOW)]) == "approve"
    assert _default_recommendation([_finding(Severity.MEDIUM)]) == "approve"
    assert _default_recommendation([_finding(Severity.HIGH, confidence=0.9)]) == "request_changes"
    assert _default_recommendation([_finding(Severity.HIGH, confidence=0.5)]) == "approve"
    assert _default_recommendation([_finding(Severity.CRITICAL)]) == "block"


def test_default_recommendation_errors_degrade():
    assert _default_recommendation([], errors=["boom"]) == "request_changes"
    assert _default_recommendation([_finding(Severity.CRITICAL)], errors=["boom"]) == "block"


def test_run_codey_agent_no_llm_deterministic():
    reports = {"security": AgentReport(agent="security", findings=[_finding(Severity.CRITICAL)])}
    review = run_codey_agent(_ctx(), reports, None)
    assert review.overall_severity == Severity.CRITICAL
    assert review.recommendation == "block"
    assert review.total_findings == 1


def test_run_codey_agent_llm_cannot_downgrade_critical():
    reports = {"security": AgentReport(agent="security", findings=[_finding(Severity.CRITICAL)])}
    llm = FakeLLM(content=json.dumps({
        "overall_severity": "low",
        "recommendation": "approve",
        "summary": "all good",
    }))
    review = run_codey_agent(_ctx(), reports, llm)
    assert review.overall_severity == Severity.CRITICAL
    assert review.recommendation == "block"
    assert review.summary == "all good"


def test_run_codey_agent_llm_can_upgrade():
    llm = FakeLLM(content=json.dumps({
        "overall_severity": "high",
        "recommendation": "request_changes",
        "summary": "needs work",
    }))
    review = run_codey_agent(_ctx(), {}, llm)
    assert review.overall_severity == Severity.HIGH
    assert review.recommendation == "request_changes"


def test_run_codey_agent_errors_degrade_llm_approve():
    llm = FakeLLM(content=json.dumps({
        "overall_severity": "info",
        "recommendation": "approve",
        "summary": "fine",
    }))
    review = run_codey_agent(_ctx(), {}, llm, prior_errors=["[security] failed"])
    assert review.recommendation == "request_changes"  # never approve on error
    assert "[security] failed" in review.errors


def test_run_codey_agent_llm_invalid_json_falls_back():
    llm = FakeLLM(content="not json at all")
    review = run_codey_agent(_ctx(), {}, llm)
    assert review.overall_severity == Severity.INFO
    assert review.recommendation == "approve"  # no findings, no agent errors
    assert any("synthesis failed" in e for e in review.errors)


# ---------------------------------------------------------------------------
# security agent — parsing helpers
# ---------------------------------------------------------------------------

def test_should_skip():
    assert _should_skip("style.css") is True
    assert _should_skip("img.PNG") is True
    assert _should_skip("README.md") is True
    assert _should_skip("config.json") is False
    assert _should_skip("app.py") is False
    assert _should_skip("config.yaml") is False


def test_parse_bandit_json():
    raw = json.dumps({"results": [{
        "test_id": "B101", "issue_text": "assert used", "filename": "a.py",
        "line_number": 5, "code": "assert x", "issue_severity": "LOW",
        "issue_confidence": "HIGH", "issue_cwe": {"link": "https://cwe/703"},
    }]})
    fs = _parse_bandit_json(raw)
    assert len(fs) == 1
    assert fs[0].severity == Severity.LOW
    assert fs[0].file_path == "a.py"
    assert fs[0].line_start == 5
    assert fs[0].confidence == 0.9


def test_parse_bandit_json_invalid_is_empty():
    assert _parse_bandit_json("not json") == []


def test_parse_semgrep_json():
    raw = json.dumps({"results": [{
        "check_id": "python.lang.security.x", "path": "a.py",
        "start": {"line": 3}, "end": {"line": 3},
        "extra": {"severity": "ERROR", "message": "bad", "lines": "code", "fix": "do this"},
    }]})
    fs = _parse_semgrep_json(raw)
    assert len(fs) == 1
    assert fs[0].severity == Severity.HIGH
    assert fs[0].evidence == "code"


def test_parse_gitleaks_json():
    raw = json.dumps([{
        "RuleID": "aws-secret", "Description": "leak", "File": "a.py",
        "StartLine": 1, "EndLine": 1, "Secret": "AKIA1234",
    }])
    fs = _parse_gitleaks_json(raw)
    assert len(fs) == 1
    assert fs[0].severity == Severity.CRITICAL
    assert "AKIA1234" in fs[0].evidence


def test_parse_tool_results_dispatch():
    assert _parse_tool_results("bandit", "not json") == []
    assert _parse_tool_results("unknown", "{}") == []


def test_strip_code_fences():
    assert _strip_code_fences("```json\n[]\n```") == "[]"
    assert _strip_code_fences("[]") == "[]"


def test_llm_output_is_parseable():
    assert _llm_output_is_parseable("[]") is True
    assert _llm_output_is_parseable("```\n[]\n```") is True
    assert _llm_output_is_parseable("nope") is False


def test_parse_llm_findings_null_severity_skipped():
    findings, err = _parse_llm_findings(json.dumps([
        {"severity": None, "title": "bad"},
        {"severity": "high", "title": "good"},
    ]))
    assert len(findings) == 1
    assert findings[0].title == "good"
    assert err is not None  # 1 malformed skipped


def test_parse_llm_findings_non_list_error():
    findings, err = _parse_llm_findings(json.dumps({"not": "array"}))
    assert findings == []
    assert err == "LLM output was not a JSON array"


def test_parse_llm_findings_unparseable():
    findings, err = _parse_llm_findings("garbage")
    assert findings == []
    assert err is None


def test_hardcoded_summary():
    assert _hardcoded_summary([]) == "(none)"
    out = _hardcoded_summary([_finding(Severity.CRITICAL, title="[hardcoded] OpenAI API key detected")])
    assert "OpenAI API key" in out


def test_diff_truncated_note():
    assert _diff_truncated_note("") == ""
    assert _diff_truncated_note("short") == ""
    assert "8000" in _diff_truncated_note("x" * 9000)


# ---------------------------------------------------------------------------
# security agent — end to end (no external tools installed)
# ---------------------------------------------------------------------------

def test_run_security_agent_no_llm_with_secret(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "leak.py").write_text('API_KEY = "sk-abcd1234efgh5678ijkl9012mnop3456"\n', encoding="utf-8")
    commit(r, "leak")

    from codey.review.git import get_changed_files, get_commit_diff

    changed = get_changed_files(r)
    diffs = get_commit_diff(r)
    ctx = ReviewContext(
        repo_path=r,
        git_hash="HEAD",
        commit_message="leak",
        changed_files=changed,
        full_diff="\n".join(diffs.values()),
        raw_full_diff="\n".join(diffs.values()),
    )
    report = run_security_agent(ctx, llm=None)
    assert any(f.severity == Severity.CRITICAL for f in report.findings)
    assert any("OpenAI API key" in f.title for f in report.findings)


def test_run_security_agent_with_llm_keeps_hardcoded(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "leak.py").write_text('API_KEY = "sk-abcd1234efgh5678ijkl9012mnop3456"\n', encoding="utf-8")
    commit(r, "leak")

    from codey.review.git import get_changed_files, get_commit_diff

    changed = get_changed_files(r)
    diffs = get_commit_diff(r)
    ctx = ReviewContext(
        repo_path=r,
        git_hash="HEAD",
        commit_message="leak",
        changed_files=changed,
        full_diff="\n".join(diffs.values()),
        raw_full_diff="\n".join(diffs.values()),
    )
    # LLM returns an empty array — must not drop the deterministic finding.
    llm = FakeLLM(content="[]")
    report = run_security_agent(ctx, llm=llm)
    assert any("OpenAI API key" in f.title for f in report.findings)


def test_run_security_agent_clean_diff_no_llm(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "x.py").write_text("x = 1\n", encoding="utf-8")
    commit(r, "init")
    ctx = ReviewContext(
        repo_path=r,
        git_hash="HEAD",
        commit_message="init",
        changed_files=["x.py"],
        full_diff="diff --git a/x.py b/x.py\n@@ -0,0 +1 @@\n+x = 1\n",
        raw_full_diff="diff --git a/x.py b/x.py\n@@ -0,0 +1 @@\n+x = 1\n",
    )
    report = run_security_agent(ctx, llm=None)
    assert report.status == "completed"
    assert all(f.severity == Severity.INFO for f in report.findings)


# ---------------------------------------------------------------------------
# code quality agent
# ---------------------------------------------------------------------------

def test_code_quality_skips_without_diff():
    report = run_code_quality_agent(_ctx(), llm=FakeLLM(content="[]"))
    assert report.status == "skipped"


def test_code_quality_skips_without_llm():
    ctx = _ctx(full_diff="diff --git a/x b/x\n@@ -1,1 +1,1 @@\n-x\n+x\n")
    report = run_code_quality_agent(ctx, llm=None)
    assert report.status == "skipped"


def test_code_quality_with_llm_evidence_enforced():
    ctx = _ctx(
        full_diff="diff --git a/x.py b/x.py\n@@ -1,1 +1,1 @@\n-x\n+x\n",
        diff_chunks=[DiffChunk(
            "x.py", "python", "m", "module",
            "diff --git a/x.py b/x.py\n@@ -1,1 +1,1 @@\n+x\n", 1, 1,
        )],
    )
    # LLM returns a finding with a hallucinated path -> no evidence -> dropped.
    llm = FakeLLM(content=json.dumps([
        {"severity": "high", "title": "bad", "file_path": "nope.py", "line_start": 1},
    ]))
    report = run_code_quality_agent(ctx, llm=llm)
    assert all(f.severity == Severity.INFO for f in report.findings)


def test_code_quality_null_severity_does_not_crash():
    ctx = _ctx(full_diff="diff --git a/x.py b/x.py\n@@ -1,1 +1,1 @@\n-x\n+x\n")
    llm = FakeLLM(content=json.dumps([
        {"severity": None, "title": "bad"},
    ]))
    report = run_code_quality_agent(ctx, llm=llm)  # must not raise
    assert report.status == "error"  # malformed item is reported, not fatal


# ---------------------------------------------------------------------------
# test agent
# ---------------------------------------------------------------------------

def test_detect_frameworks_pytest(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    (r / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    fw = _detect_frameworks(r)
    assert ("pytest", ["pytest", "-x", "--tb=short"]) in fw


def test_detect_frameworks_npm(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    (r / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    fw = _detect_frameworks(r)
    assert any(name == "npm" for name, _ in fw)


def test_detect_frameworks_pyproject_without_pytest_ignored(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    (r / "pyproject.toml").write_text("[tool.uv]\n", encoding="utf-8")
    assert _detect_frameworks(r) == []


def test_detect_frameworks_none(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    assert _detect_frameworks(r) == []


def test_run_test_agent_opt_in_skip(repo):
    ctx = ReviewContext(
        repo_path=repo, git_hash="HEAD", commit_message="m",
        changed_files=["main.py"], full_diff="",
    )
    report = run_test_agent(ctx, llm=None)
    assert report.status == "skipped"
    assert "run-tests" in report.summary


def test_run_test_agent_no_framework(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    ctx = ReviewContext(
        repo_path=r, git_hash="HEAD", commit_message="m",
        changed_files=["x.py"], full_diff="", run_tests=True,
    )
    report = run_test_agent(ctx, llm=None)
    assert report.status == "skipped"
    assert "No test suite" in report.summary


# ---------------------------------------------------------------------------
# index agent
# ---------------------------------------------------------------------------

def test_run_index_agent_no_llm(repo):
    db = CacheDB()
    index_repository(repo, db)
    ctx = _ctx(repo_path=repo, git_hash=None)
    # use the real git hash
    from codey.index.indexer import git_head_hash

    ctx.git_hash = git_head_hash(repo)
    report, summary = run_index_agent(ctx, db, None)
    assert report.status == "completed"
    assert report.findings[0].category == FindingCategory.ARCHITECTURE
    assert summary  # symbol overview
    db.close()


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def test_attach_evidence_from_chunk():
    ctx = _ctx(diff_chunks=[DiffChunk("x.py", "python", "m", "module", "line 1\nline 2\nline 3\n", 1, 3)])
    f = Finding(category=FindingCategory.CORRECTNESS, title="t", file_path="x.py", line_start=2)
    attach_evidence([f], ctx)
    assert f.evidence  # extracted from the chunk


def test_attach_evidence_preserves_existing():
    ctx = _ctx(diff_chunks=[DiffChunk("x.py", "python", "m", "module", "x", 1, 1)])
    f = Finding(category=FindingCategory.CORRECTNESS, title="t", file_path="x.py", evidence="kept")
    attach_evidence([f], ctx)
    assert f.evidence == "kept"


def test_attach_evidence_hallucinated_line_gets_nothing():
    ctx = _ctx(diff_chunks=[DiffChunk("x.py", "python", "m", "module", "x", 1, 1)])
    f = Finding(category=FindingCategory.CORRECTNESS, title="t", file_path="x.py", line_start=999)
    attach_evidence([f], ctx)
    assert f.evidence == ""


def test_attach_evidence_no_file_path_is_noop():
    ctx = _ctx(diff_chunks=[DiffChunk("x.py", "python", "m", "module", "x", 1, 1)])
    f = Finding(category=FindingCategory.CORRECTNESS, title="t")
    attach_evidence([f], ctx)
    assert f.evidence == ""


def test_attach_evidence_from_source_fallback():
    ctx = _ctx(file_sources={"x.py": "\n".join(f"line {i}" for i in range(1, 11))})
    f = Finding(category=FindingCategory.CORRECTNESS, title="t", file_path="x.py", line_start=5)
    attach_evidence([f], ctx)
    assert "line 5" in f.evidence


def test_attach_evidence_out_of_range_line_gets_nothing():
    ctx = _ctx(file_sources={"x.py": "a\nb\nc\n"})
    f = Finding(category=FindingCategory.CORRECTNESS, title="t", file_path="x.py", line_start=100)
    attach_evidence([f], ctx)
    assert f.evidence == ""
