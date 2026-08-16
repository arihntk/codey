"""Tests for codey.review.pipeline — budgeting, summarization, non-HEAD."""

from __future__ import annotations

from pathlib import Path

from codey.agents.context import DiffChunk, ReviewContext
from codey.cache.ast_cache import CacheDB
from codey.review.pipeline import (
    PipelineResult,
    _prune_to_budget,
    _summarise_if_needed,
    run_pipeline,
)
from tests.conftest import FakeLLM, commit, init_repo
from tests.test_llm import (  # noqa: F401  (ensure import machinery works)
    test_summarize_diff_large_calls_llm,
)


def _ctx(chunks, max_tokens):
    return ReviewContext(
        repo_path=Path("/repo"),
        git_hash="h",
        commit_message="m",
        diff_chunks=chunks,
        max_tokens=max_tokens,
    )


def _chunk(path, text, start, end):
    return DiffChunk(path, "python", "sym", "function", text, start, end)


def test_prune_to_budget_no_chunks_is_noop():
    ctx = _ctx([], 10)
    _prune_to_budget(ctx)
    assert ctx.pruned_chunks == []


def test_prune_to_budget_under_budget_keeps_all():
    chunks = [_chunk("a.py", "x" * 40, 1, 10), _chunk("b.py", "y" * 40, 1, 10)]
    ctx = _ctx(chunks, max_tokens=100)
    _prune_to_budget(ctx)
    assert ctx.pruned_chunks == []
    assert len(ctx.diff_chunks) == 2


def test_prune_to_budget_over_budget_prunes_and_respects_budget():
    # 3 chunks of 10 tokens each; budget of 10 -> only 1 survives.
    chunks = [
        _chunk("a.py", "a" * 40, 1, 10),
        _chunk("a.py", "b" * 40, 11, 20),
        _chunk("b.py", "c" * 40, 1, 10),
    ]
    ctx = _ctx(chunks, max_tokens=10)
    _prune_to_budget(ctx)
    assert len(ctx.diff_chunks) == 1
    assert len(ctx.pruned_chunks) == 2
    # Every pruned entry is "path:start-end".
    assert all(":" in p for p in ctx.pruned_chunks)
    # Budget is respected.
    total = sum(len(c.diff_text) // 4 for c in ctx.diff_chunks)
    assert total <= 10


def test_prune_to_budget_keeps_one_chunk_per_file():
    # 2 files, 2 chunks each (10 tokens each); budget 20 fits one per file.
    chunks = [
        _chunk("a.py", "a" * 40, 1, 10),
        _chunk("a.py", "b" * 40, 11, 20),
        _chunk("b.py", "c" * 40, 1, 10),
        _chunk("b.py", "d" * 40, 11, 20),
    ]
    ctx = _ctx(chunks, max_tokens=20)
    _prune_to_budget(ctx)
    kept_files = {c.file_path for c in ctx.diff_chunks}
    assert kept_files == {"a.py", "b.py"}
    assert len(ctx.diff_chunks) == 2


def test_summarise_if_needed_safeguard_off_is_noop():
    diffs = {"a.py": "x" * 30000}
    _summarise_if_needed(safeguard=False, summarizer=FakeLLM(), diffs=diffs, paths=["a.py"])
    assert diffs["a.py"].startswith("x" * 30000)


def test_summarise_if_needed_replaces_large_diffs():
    from codey.llm.factory import ResolvedLLM

    fake = FakeLLM(content="summary")
    summarizer = ResolvedLLM(fake, object(), "m", "k", None)
    diffs = {"a.py": "x" * 30000}
    _summarise_if_needed(safeguard=True, summarizer=summarizer, diffs=diffs, paths=["a.py"])
    assert diffs["a.py"] == "summary"


def test_run_pipeline_with_diff_override(repo):
    db = CacheDB()
    result = run_pipeline(
        repo,
        db,
        primary_llm=None,
        summarizer_llm=None,
        diff_override={"main.py": "diff --git a/main.py b/main.py\n@@ -1,1 +1,1 @@\n-x\n+x\n"},
        changed_files_override=["main.py"],
    )
    assert isinstance(result, PipelineResult)
    assert result.review is not None
    assert "security" in result.review.agent_reports
    db.close()


def test_run_pipeline_non_head_commit(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("def one():\n    return 1\n", encoding="utf-8")
    commit(r, "c1")
    (r / "a.py").write_text("def one():\n    return 1\n\ndef two():\n    return 2\n", encoding="utf-8")
    commit(r, "c2")

    db = CacheDB()
    # Review c1 (HEAD~1) — must materialize a worktree and still produce a review.
    result = run_pipeline(r, db, primary_llm=None, summarizer_llm=None, commit="HEAD~1")
    assert result.review.commit_hash == result.ctx.git_hash
    assert result.review is not None
    db.close()
