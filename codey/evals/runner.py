"""Eval runner — materialize scenario repos, run the full review pipeline, score results."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from codey.agents.schemas import Severity
from codey.cache.ast_cache import CacheDB
from codey.evals.metrics import evidence_grounded, score_findings
from codey.evals.report import AggregateScore, EvalRunReport, ExpectedIssueOut, ScenarioScore
from codey.evals.scenarios import SCENARIOS, EvalFile, EvalScenario
from codey.llm.factory import ResolvedLLM

__all__ = ["StubLLM", "StubResponse", "run_evals"]

_FAKE_EXCLUDED_TAGS = {"llm-only", "tests-enabled"}


class StubResponse:
    """Minimal langchain-style response for the fake LLM."""

    def __init__(self, content: str) -> None:
        self.content = content


_SYSTEM_ROUTES: dict[str, str] = {
    "lead code reviewer": '{"overall_severity": "info", "recommendation": "approve", '
    '"summary": "Deterministic synthesis (fake mode)."}',
    "impartial evaluator": '{"groundedness": 5.0, "completeness": 5.0, "actionability": 5.0, '
    '"precision": 5.0, "overall": 5.0, "comment": "fake-mode judge"}',
    "code architecture analyst": "Indexed repository; no architecture concerns.",
    "concise code-review assistant": "Diff summarized.",
    "test expert": "[]",
    "senior security analyst": "[]",
    "senior code quality reviewer": "[]",
}


class StubLLM:
    """Deterministic fake LLM that routes responses by system-prompt fingerprint.

    Returns ``[]`` (no LLM findings) for the finding-producing agents, a benign
    synthesis object for the orchestrator, and a perfect rubric for the judge —
    so fake mode exercises the deterministic components and pipeline plumbing.
    """

    def __init__(self) -> None:
        self.calls: list = []

    def invoke(self, messages: list, **kwargs: object) -> StubResponse:
        self.calls.append(messages)
        for message in messages:
            if getattr(message, "type", "") != "system":
                continue
            text = getattr(message, "content", "") or ""
            for fingerprint, response in _SYSTEM_ROUTES.items():
                if fingerprint in text:
                    return StubResponse(response)
        return StubResponse("[]")


# --- git helpers (evals must not depend on the test suite) ------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "eval@codey.local")
    _git(repo, "config", "user.name", "Codey Eval")


def _write_files(repo: Path, files: tuple[EvalFile, ...]) -> None:
    for f in files:
        path = repo / f.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f.content, encoding="utf-8")


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


# --- selection --------------------------------------------------------------


def _select_scenarios(mode: str, scenario_ids: list[str] | None, tags: list[str] | None) -> list[EvalScenario]:
    selected = list(SCENARIOS)
    if scenario_ids:
        wanted = set(scenario_ids)
        selected = [s for s in selected if s.id in wanted]
    elif tags:
        wanted = set(tags)
        selected = [s for s in selected if wanted & set(s.tags)]
    if mode == "fake":
        selected = [s for s in selected if not (_FAKE_EXCLUDED_TAGS & set(s.tags))]
    return selected


# --- per-scenario scoring ---------------------------------------------------


def _expected_out(e) -> ExpectedIssueOut:
    return ExpectedIssueOut(
        agent=e.agent, category=e.category.value, severity=e.severity.value,
        file_path=e.file_path, line_start=e.line_start, line_end=e.line_end,
        keywords=list(e.keywords), expect_absent=e.expect_absent,
    )


def _run_scenario(
    sc: EvalScenario,
    *,
    primary: object,
    summarizer: ResolvedLLM,
    db: CacheDB,
    run_tests: bool,
    tmp_root: Path,
    progress: Callable[[str], None] | None,
) -> tuple[ScenarioScore, object | None, str]:
    from codey.review.git import get_changed_files
    from codey.review.pipeline import run_pipeline

    repo_dir = tmp_root / sc.id
    repo_dir.mkdir(exist_ok=True)
    repo = repo_dir / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()

    _init_repo(repo)
    for i, files in enumerate(sc.commits):
        _write_files(repo, files)
        _commit(repo, f"commit {i}")

    if progress is not None:
        progress(f"eval {sc.id}: reviewing {sc.review_commit}")
    changed = get_changed_files(repo, commit=sc.review_commit)

    t0 = time.monotonic()
    try:
        result = run_pipeline(
            repo, db,
            primary_llm=primary, summarizer_llm=summarizer,
            commit=sc.review_commit,
            run_tests=run_tests and sc.run_tests,
        )
    except Exception as e:  # scenario-level failure, keep going
        return ScenarioScore(id=sc.id, description=sc.description, status="error", error=str(e)), None, ""

    duration = time.monotonic() - t0
    review = result.review
    ctx = result.ctx

    metrics = score_findings(review.agent_reports, sc.expected_issues, expect_clean=sc.expect_clean)
    non_info = [f for r in review.agent_reports.values() for f in r.findings if f.severity != Severity.INFO]
    grounded = [evidence_grounded(f, ctx) for f in non_info]
    grounded_ratio = mean(grounded) if grounded else None
    tokens = sum(r.token_usage for r in review.agent_reports.values())

    changed_count = len(changed)
    coverage = (len(review.files_reviewed) / changed_count) if changed_count else None
    recommendation_match = (
        None if sc.expected_recommendation is None else review.recommendation == sc.expected_recommendation
    )
    pipeline_checks = {
        "pruned": sc.expect_pruned_chunks == (len(review.pruned_chunks) > 0),
        "deps": all(d in review.dependent_files_checked for d in sc.expected_dependent_files),
    }

    score = ScenarioScore(
        id=sc.id, description=sc.description,
        tp=metrics.tp, fp=metrics.fp, fn=metrics.fn,
        precision=metrics.precision, recall=metrics.recall, f1=metrics.f1,
        severity_exact=metrics.severity_exact, severity_within_one=metrics.severity_within_one,
        evidence_ratio=metrics.evidence_ratio, evidence_grounded=grounded_ratio,
        recommendation=review.recommendation, recommendation_match=recommendation_match,
        coverage=coverage,
        files_reviewed=len(review.files_reviewed), changed_files=changed_count,
        pruned_chunks=len(review.pruned_chunks),
        dependent_files_checked=review.dependent_files_checked,
        pipeline_checks=pipeline_checks,
        false_positives=metrics.false_positives,
        false_negatives=[_expected_out(e) for e in metrics.false_negatives],
        duration_s=duration, tokens=tokens,
    )
    return score, review, (ctx.raw_full_diff or "")


def _aggregate(scores: list[ScenarioScore], total: float) -> AggregateScore:
    completed = [s for s in scores if s.status == "completed"]
    tp = sum(s.tp for s in completed)
    fp = sum(s.fp for s in completed)
    fn = sum(s.fn for s in completed)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    def _avg(key: str) -> float | None:
        values = [getattr(s, key) for s in completed if getattr(s, key) is not None]
        return mean(values) if values else None

    rec_values = [s.recommendation_match for s in completed if s.recommendation_match is not None]
    checks = [all(s.pipeline_checks.values()) for s in completed if s.pipeline_checks]
    judge_values = [s.judge.overall for s in completed if s.judge is not None]

    return AggregateScore(
        scenarios_completed=len(completed),
        scenarios_error=sum(1 for s in scores if s.status == "error"),
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        severity_exact=_avg("severity_exact"),
        severity_within_one=_avg("severity_within_one"),
        evidence_ratio=_avg("evidence_ratio"),
        evidence_grounded=_avg("evidence_grounded"),
        recommendation_accuracy=mean(rec_values) if rec_values else None,
        pipeline_checks_passed=mean(checks) if checks else None,
        judge_overall=mean(judge_values) if judge_values else None,
        tokens=sum(s.tokens for s in completed),
        duration_s=total,
    )


# --- entry point ------------------------------------------------------------


def run_evals(
    *,
    mode: str = "real",
    scenario_ids: list[str] | None = None,
    tags: list[str] | None = None,
    judge: bool | None = None,
    run_tests: bool = False,
    keep_repos: bool = False,
    progress: Callable[[str], None] | None = None,
) -> EvalRunReport:
    """Run the golden-scenario evaluation suite and return a scored report."""
    if mode not in ("real", "fake"):
        raise ValueError(f"mode must be 'real' or 'fake', got {mode!r}")

    scenarios = _select_scenarios(mode, scenario_ids, tags)
    judge_enabled = (mode == "real") if judge is None else judge

    tmp_root = Path(tempfile.mkdtemp(prefix="codey-eval-"))
    db = CacheDB(tmp_root / "cache.db")

    if mode == "real":
        from codey.llm.factory import build_llm, build_summarizer

        primary = build_llm().model
        summarizer = build_summarizer()
    else:
        primary = StubLLM()
        summarizer = ResolvedLLM(model=StubLLM(), preset=object(), model_name="stub", api_key="", base_url=None)

    scores: list[ScenarioScore] = []
    start = time.monotonic()
    try:
        for sc in scenarios:
            score, review, raw_diff = _run_scenario(
                sc, primary=primary, summarizer=summarizer, db=db,
                run_tests=run_tests, tmp_root=tmp_root, progress=progress,
            )
            if judge_enabled and score.status == "completed" and review is not None:
                from codey.evals.judge import judge_review

                judge_score, judge_tokens, judge_err = judge_review(
                    primary, description=sc.description, diff=raw_diff, review=review,
                )
                score.judge = judge_score
                score.tokens += judge_tokens
                if judge_score is None and judge_err:
                    score.notes.append(f"judge: {judge_err}")
            scores.append(score)
    finally:
        db.close()
        if not keep_repos:
            shutil.rmtree(tmp_root, ignore_errors=True)

    total = time.monotonic() - start
    agg = _aggregate(scores, total)
    return EvalRunReport(
        mode=mode, judge_enabled=judge_enabled,
        scenario_count=len(scores), scenarios=scores,
        aggregate=agg, duration_s=total, tokens=agg.tokens,
    )
