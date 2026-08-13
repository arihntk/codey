"""Dependent chunk fetcher — finds and loads affected-but-unmodified code.

For a set of changed files, uses the call graph / import graph (via
reverse_dependencies) to find one-hop dependent files. Loads relevant source
snippets to include in the agent context, respecting the context budget.
"""

from __future__ import annotations

from pathlib import Path

from codey.cache.ast_cache import CacheDB
from codey.index.callgraph import reverse_dependencies

__all__ = ["fetch_dependent_files", "load_dependent_sources", "enrich_context"]

_MAX_DEPENDENT_FILES = 15
_MAX_SOURCE_CHARS = 8000


def fetch_dependent_files(
    repo_path: Path,
    git_hash: str,
    db: CacheDB,
    changed_files: list[str],
) -> list[str]:
    """Find files that depend on changed files, excluding the changed files themselves."""
    if not changed_files:
        return []
    repo = repo_path.resolve()
    affected = [str((repo / f).resolve()) for f in changed_files]
    deps = reverse_dependencies(
        repo,
        git_hash,
        db,
        affected,
        repo=repo,
    )
    # Filter out files that no longer exist.
    return [d for d in deps if (repo / d).is_file()][:_MAX_DEPENDENT_FILES]


def load_dependent_sources(
    repo: Path,
    dependent_files: list[str],
    *,
    max_chars: int = _MAX_SOURCE_CHARS,
) -> dict[str, str]:
    """Load truncated source code for dependent files."""
    sources: dict[str, str] = {}
    for rel in dependent_files:
        fpath = repo / rel
        if not fpath.is_file():
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... [truncated]"
            sources[rel] = content
        except OSError:
            continue
    return sources


def enrich_context(
    ctx,
    db: CacheDB,
) -> None:
    """Populate dependent_files and file_sources on the ReviewContext in-place."""
    ctx.dependent_files = fetch_dependent_files(
        ctx.repo_path, ctx.git_hash, db, ctx.changed_files,
    )
    ctx.file_sources = load_dependent_sources(ctx.repo_path, ctx.dependent_files)
    # Also include full source for changed files.
    for rel in ctx.changed_files:
        fpath = ctx.repo_path / rel
        if fpath.is_file():
            try:
                ctx.file_sources[rel] = fpath.read_text(
                    encoding="utf-8", errors="replace",
                )[:_MAX_SOURCE_CHARS]
            except OSError:
                pass