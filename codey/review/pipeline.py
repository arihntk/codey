"""Review pipeline — diff acquisition, chunking, dep fetching, budgeting, graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codey.agents.context import ReviewContext
from codey.cache.ast_cache import CacheDB
from codey.graph.build import run_review
from codey.index.indexer import git_head_hash
from codey.llm.summarize import summarize_diffs
from codey.review.chunking import chunk_diff
from codey.review.deps import enrich_context
from codey.review.git import get_changed_files, get_commit_diff, get_latest_commit

__all__ = ["PipelineResult", "run_pipeline"]

_LARGE_DIFF_THRESHOLD = 24_000  # chars above which a file's diff is summarised


class PipelineResult:
    def __init__(self, review, ctx, large_diffs=None):
        self.review = review
        self.ctx = ctx
        self.large_diffs = large_diffs or []


def run_pipeline(
    repo_path: Path | str,
    db: CacheDB,
    *,
    primary_llm: object | None = None,
    summarizer_llm: object | None = None,
    progress_callback=None,
    diff_override: dict[str, str] | None = None,
    changed_files_override: list[str] | None = None,
    commit: str = "HEAD",
    run_tests: bool = False,
) -> Any:
    repo = Path(repo_path).resolve()

    commit_info = get_latest_commit(repo, commit=commit)
    git_hash = commit_info.hash if commit_info else (git_head_hash(repo) or "unknown")
    commit_message = commit_info.message if commit_info else ""

    if diff_override is not None:
        diffs = diff_override
        changed_files = changed_files_override or list(diff_override.keys())
    else:
        diffs = get_commit_diff(repo, commit=commit)
        changed_files = get_changed_files(repo, commit=commit)

    if not diffs and not changed_files:
        changed_files = changed_files_override or []

    # For non-HEAD commits, materialize a worktree so indexing/chunking/tooling
    # operate on the reviewed commit's tree. The diff is computed from the real
    # repo above (commit-relative, correct either way).
    from codey.review.git import materialize_commit, remove_worktree, resolve_commit

    worktree: Path | None = None
    try:
        head_hash = resolve_commit(repo, "HEAD")
        scan_repo = repo
        if git_hash and head_hash and git_hash != head_hash:
            worktree = materialize_commit(repo, git_hash)
            scan_repo = worktree

        # Index the scan tree; cache is keyed on the CANONICAL repo path.
        from codey.index.callgraph import build_call_graph
        from codey.index.indexer import index_repository

        index_result = index_repository(scan_repo, db, cache_repo_path=repo)
        build_call_graph(scan_repo, index_result.git_hash, db, cache_repo_path=repo)

        chunk_list = chunk_diff(
            diffs, db=db, repo_path=str(repo), git_hash=index_result.git_hash,
        )

        # Snapshot the raw diff BEFORE large-diff summarisation mutates `diffs`.
        raw_full_diff = "\n".join(diffs.values()) if diffs else ""

        large_diffs: list[str] = []
        if summarizer_llm is not None:
            large_diffs = [path for path, text in diffs.items() if len(text) > _LARGE_DIFF_THRESHOLD]
            _summarise_if_needed(
                safeguard=primary_llm is not None,
                summarizer=summarizer_llm, diffs=diffs, paths=large_diffs,
            )

        full_diff = "\n".join(diffs.values()) if diffs else ""

        ctx = ReviewContext(
            repo_path=scan_repo,
            git_hash=git_hash,
            commit_message=commit_message,
            changed_files=changed_files,
            diff_chunks=chunk_list,
            full_diff=full_diff,
            raw_full_diff=raw_full_diff,
            db=db,
            run_tests=run_tests,
            cache_repo_path=repo,
        )

        enrich_context(ctx, db)
        _prune_to_budget(ctx)

        review = run_review(ctx, primary_llm=primary_llm, progress_callback=progress_callback)

        return PipelineResult(review=review, ctx=ctx, large_diffs=large_diffs)
    finally:
        if worktree is not None:
            remove_worktree(repo, worktree)


def _summarise_if_needed(*, safeguard: bool, summarizer, diffs: dict[str, str], paths: list[str]) -> None:
    if not safeguard or summarizer is None or not paths:
        return
    for s in summarize_diffs(summarizer, {p: diffs[p] for p in paths}):
        diffs[s.path] = s.summary


def _prune_to_budget(ctx: ReviewContext) -> None:
    """Drop low-priority chunks past ``ctx.max_tokens``; record every drop.

    The largest chunk of every file is guaranteed a seat, then remaining budget
    is filled largest-first. Every dropped chunk is recorded on ``pruned_chunks``.
    """
    max_tokens = ctx.max_tokens
    if not ctx.diff_chunks:
        return
    total = sum(len(c.diff_text) // 4 for c in ctx.diff_chunks)
    if total <= max_tokens:
        ctx.pruned_chunks = []
        return

    kept: list = []
    kept_budget = 0
    largest_by_file: dict[str, object] = {}
    for chunk in ctx.diff_chunks:
        cur = largest_by_file.get(chunk.file_path)
        if cur is None or len(chunk.diff_text) > len(cur.diff_text):
            largest_by_file[chunk.file_path] = chunk
    for chunk in largest_by_file.values():
        cost = len(chunk.diff_text) // 4
        if kept_budget + cost <= max_tokens:
            kept.append(chunk)
            kept_budget += cost

    remaining = [c for c in ctx.diff_chunks if c not in kept]
    remaining.sort(key=lambda c: len(c.diff_text), reverse=True)
    for chunk in remaining:
        cost = len(chunk.diff_text) // 4
        if kept_budget + cost > max_tokens:
            continue
        kept.append(chunk)
        kept_budget += cost

    kept_names = {id(c) for c in kept}
    ctx.pruned_chunks = [
        f"{c.file_path}:{c.line_start}-{c.line_end}"
        for c in ctx.diff_chunks
        if id(c) not in kept_names
    ]
    kept.sort(key=lambda c: (c.file_path, c.line_start))
    ctx.diff_chunks = kept
