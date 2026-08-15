"""Review pipeline — orchestrates diff acquisition, chunking, dep fetching,
context budgeting, and graph execution.

Entry point: ``run_pipeline(repo_path, ...)`` builds the ReviewContext,
runs the graph, and returns the final ReviewSummary.
"""

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

# Threshold (chars) above which a file's diff is summarised via the cheap model.
_LARGE_DIFF_THRESHOLD = 24_000


class PipelineResult:
    """Wrapper with the final review + contextual metadata."""

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
    """Run the full review pipeline on the given commit.

    Args:
        repo_path: Path to the git repo.
        db: Open CacheDB instance.
        primary_llm: Main review model (or None to skip LLM calls).
        summarizer_llm: Cheap/fast model for diff summarisation (or None).
        progress_callback: Optional callback receiving graph stream chunks.
        diff_override: Override per-file diffs (skip git diff acquisition).
        changed_files_override: Override changed files list.
        commit: Commit ref to review (defaults to ``HEAD``, i.e. the latest commit).
        run_tests: Allow the test agent to execute detected test commands.
            Defaults to False — executing tests runs code from the repo
            under review and requires explicit opt-in (``--run-tests``).

    Returns:
        PipelineResult wrapping the ReviewSummary and ReviewContext.
    """
    repo = Path(repo_path).resolve()

    # 1. Acquire diff + commit info.
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

    # 2. If we're reviewing a non-HEAD commit, materialize its tree in a
    #    temporary worktree so indexing / chunking / tooling operate on the
    #    reviewed commit's files — not the current working tree. The diff was
    #    already computed from the real repo above (diff is commit-relative,
    #    so it's correct either way). Symbol mapping, dependent files, file
    #    sources, and the security tools all use `scan_repo`.
    from codey.review.git import materialize_commit, remove_worktree, resolve_commit

    worktree: Path | None = None
    try:
        head_hash = resolve_commit(repo, "HEAD")
        if git_hash and head_hash and git_hash != head_hash:
            worktree = materialize_commit(repo, git_hash)
            scan_repo = worktree
        else:
            scan_repo = repo

        # 3. Index the scan tree (so symbol table is fresh for chunking + deps).
        #    This also builds the call graph.
        from codey.index.callgraph import build_call_graph
        from codey.index.indexer import index_repository

        index_result = index_repository(scan_repo, db)
        build_call_graph(scan_repo, index_result.git_hash, db)

        # 4. Chunk diffs (function/class level) using cached symbol table.
        chunk_list = chunk_diff(
            diffs,
            db=db,
            repo_path=str(scan_repo),
            git_hash=index_result.git_hash,
        )

        # Snapshot the raw diff BEFORE large-diff summarisation mutates `diffs`.
        # Deterministic scanners (hardcoded-secret detector) must run against the
        # actual code, not an LLM-generated summary.
        raw_full_diff = "\n".join(diffs.values()) if diffs else ""

        # 5. Summarise large diffs via the cheap/fast model.
        #    `_summarise_if_needed` replaces the raw diff text with a compact
        #    summary, and `full_diff` is assembled *after* this step (below), so
        #    agents see the summaries instead of raw hunks for very large changes.
        large_diffs: list[str] = []
        if summarizer_llm is not None:
            for path, text in diffs.items():
                if len(text) > _LARGE_DIFF_THRESHOLD:
                    large_diffs.append(path)
            _summarise_if_needed(
                safeguard=primary_llm is not None,
                summarizer=summarizer_llm,
                diffs=diffs,
                paths=large_diffs,
            )

        # 6. Assemble full diff text for context.
        full_diff = "\n".join(diffs.values()) if diffs else ""

        # 7. Build the review context (against the scan tree).
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
        )

        # 8. Fetch dependent (affected-but-unmodified) files + source snippets.
        enrich_context(ctx, db)

        # 9. Enforce context window budget on chunks.
        _prune_to_budget(ctx)

        # 10. Run the LangGraph review graph.
        review = run_review(
            ctx,
            primary_llm=primary_llm,
            progress_callback=progress_callback,
        )

        return PipelineResult(review=review, ctx=ctx, large_diffs=large_diffs)
    finally:
        if worktree is not None:
            remove_worktree(repo, worktree)


def _summarise_if_needed(*, safeguard: bool, summarizer, diffs: dict[str, str], paths: list[str]) -> None:
    """If any diffs exceed the large threshold, summarise them via the cheap model.

    This replaces the raw diff text in the diffs dict with a compact summary,
    so downstream agents get facts instead of raw hunks for very large changes.
    """
    if not safeguard or summarizer is None or not paths:
        return
    summaries = summarize_diffs(summarizer, {p: diffs[p] for p in paths})
    for s in summaries:
        diffs[s.path] = s.summary


def _prune_to_budget(ctx: ReviewContext) -> None:
    """Drop low-priority chunks if the total context exceeds the model budget.

    Uses ``ctx.max_tokens`` as the budget (no hardcoded constant). Selection:
    the largest chunk of every file is guaranteed a seat (so no file is
    silently dropped wholesale), then remaining budget is filled largest-first.
    Every dropped chunk is recorded on ``ctx.pruned_chunks`` so the user knows
    coverage was truncated — never a silent omission.
    """
    max_tokens = ctx.max_tokens
    if not ctx.diff_chunks:
        return
    total = sum(len(c.diff_text) // 4 for c in ctx.diff_chunks)
    if total <= max_tokens:
        ctx.pruned_chunks = []
        return

    # Pass 1: keep the largest chunk of each file (fairness — every changed
    # file keeps at least a slice under review).
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

    # Pass 2: fill remaining budget largest-first.
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
    # Re-sort by file path + line for readability.
    kept.sort(key=lambda c: (c.file_path, c.line_start))
    ctx.diff_chunks = kept
