"""Shared context passed to every agent in the review graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codey.cache.ast_cache import CacheDB

__all__ = ["ReviewContext", "DiffChunk"]


@dataclass
class DiffChunk:
    """A logical chunk of a diff (function/class level)."""

    file_path: str
    language: str
    symbol: str  # function/class name or "<module>"
    symbol_kind: str  # function, class, method, module
    diff_text: str
    line_start: int
    line_end: int
    full_source: str = ""  # full file source for context


@dataclass
class ReviewContext:
    """Everything an agent needs to review a commit."""

    repo_path: Path
    git_hash: str
    commit_message: str
    changed_files: list[str] = field(default_factory=list)
    dependent_files: list[str] = field(default_factory=list)
    diff_chunks: list[DiffChunk] = field(default_factory=list)
    full_diff: str = ""
    raw_full_diff: str = ""  # un-summarized diff, for deterministic scanners
    db: CacheDB | None = None
    file_sources: dict[str, str] = field(default_factory=dict)  # path -> full source
    index_summary: str = ""  # architecture/design summary from indexer
    max_tokens: int = 100_000  # context window budget
    run_tests: bool = False  # execute detected test commands (opt-in; see CLI)
    cache_repo_path: Path | None = field(
        default=None,
        # Canonical repo used as the cache key when repo_path is a temporary
        # worktree (non-HEAD reviews). File reads use repo_path; DB keys use
        # this so temp worktrees never pollute or miss the cache.
    )
    pruned_chunks: list[str] = field(
        default_factory=list,
        # Chunk ranges dropped by context-budget pruning, e.g. 'path:10-25'.
        # Empty when nothing was dropped.
    )
