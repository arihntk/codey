"""Repository indexer — parses source files and caches AST/symbol metadata.

Uses ``git ls-files`` when in a git repo to respect .gitignore; falls back
to a directory walk with basic ignore patterns for non-git trees.

Git-hash diffing: on each run the indexer computes the current HEAD hash.
If that hash is already indexed, it no-ops. Otherwise it reuses unchanged
file entries from the last indexed hash (by content hash) and only
re-parses files whose content has changed.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter_language_pack import get_parser

from codey.cache.ast_cache import CacheDB, FileEntry, SymbolRecord
from codey.index.languages import detect_language
from codey.index.symbols import extract_symbols, serialize_ast
from codey.process import allowlist_env

__all__ = ["IndexResult", "index_repository", "git_head_hash", "list_repo_files"]

# Patterns to skip when walking without git.
_DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "dist", "build", ".eggs", ".tox", ".cache", "htmlcov", ".idea",
    ".vscode",
}
_DEFAULT_IGNORE_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".class"}


@dataclass
class IndexResult:
    """Summary of an indexing run."""

    git_hash: str
    total_files: int = 0
    parsed_files: int = 0
    reused_files: int = 0
    skipped_files: int = 0
    symbols_extracted: int = 0
    changed_files: list[str] = field(default_factory=list)


def git_head_hash(repo_path: Path | str) -> str | None:
    """Return the current HEAD commit hash, or None if not a git repo.

    Runs with an allowlisted env: git may execute repo-supplied config hooks
    (core.fsmonitor, pager, filters) that must never see credentials.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=allowlist_env(),
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def list_repo_files(repo_path: Path) -> list[Path]:
    """List all source files in the repo, respecting .gitignore via git ls-files.

    Allowlisted env: git config (core.fsmonitor etc.) can run repo-supplied
    commands during ls-files.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(repo_path),
            capture_output=True,
            text=False,
            timeout=30,
            check=False,
            env=allowlist_env(),
        )
        if proc.returncode == 0:
            files = [
                (repo_path / f.decode("utf-8", errors="replace")).resolve()
                for f in proc.stdout.split(b"\x00")
                if f.strip()
            ]
            return [f for f in files if f.is_file()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return _walk_files(repo_path)


def _walk_files(repo_path: Path) -> list[Path]:
    """Fallback: walk directory tree without git."""
    results: list[Path] = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _DEFAULT_IGNORE_DIRS]
        for name in files:
            if Path(name).suffix in _DEFAULT_IGNORE_SUFFIXES:
                continue
            results.append(Path(root) / name)
    return results


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _parse_file(path: Path, language: str) -> tuple[str, str, list[SymbolRecord], int]:
    """Parse a single file, returning (ast_json, symbols_json, symbols, byte_count)."""
    raw = path.read_bytes()
    try:
        parser = get_parser(language)
        tree = parser.parse(raw)
    except Exception:
        return "{}", "[]", [], len(raw)
    symbols = extract_symbols(tree, str(path.name), language)
    ast_json = serialize_ast(tree)
    import json
    symbols_json = json.dumps(
        [{"n": s.name, "q": s.qualified_name, "k": s.kind, "ls": s.line_start, "le": s.line_end} for s in symbols],
        separators=(",", ":"),
    )
    return ast_json, symbols_json, symbols, len(raw)


def index_repository(
    repo_path: Path | str,
    db: CacheDB,
    *,
    force: bool = False,
    cache_repo_path: Path | str | None = None,
) -> IndexResult:
    """Index a repository, reusing cached entries for unchanged files.

    Args:
        repo_path: Absolute path to the repo root (used for file reads).
        db: Open CacheDB instance.
        force: If True, re-parse all files even if hash is unchanged.
        cache_repo_path: Canonical repo path used as the cache key. When the
            repo being indexed lives in a temporary worktree (non-HEAD
            reviews), pass the real repo here so cache entries are keyed
            canonically and reused across runs instead of accumulating rows
            for deleted tmpdir paths.

    Returns:
        IndexResult with counts and changed file list.
    """
    repo = Path(repo_path).resolve()
    cache_key = str(Path(cache_repo_path or repo).resolve())
    head_hash = git_head_hash(repo) or "no-git"
    result = IndexResult(git_hash=head_hash)

    if not force and db.has_indexed_hash(cache_key, head_hash):
        existing = db.list_file_rel_paths(cache_key, head_hash)
        result.total_files = len(existing)
        result.reused_files = len(existing)
        return result

    last_hash = db.last_indexed_hash(cache_key) if not force else None
    last_hashes: dict[str, str] = {}
    if last_hash:
        last_hashes = db.file_entry_hashes(cache_key, last_hash)

    db.upsert_index_run(cache_key, head_hash, file_count=0)

    files = list_repo_files(repo)
    all_symbols: list[SymbolRecord] = []

    for fpath in files:
        rel = str(fpath.relative_to(repo))
        language = detect_language(fpath)
        if not language:
            result.skipped_files += 1
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            result.skipped_files += 1
            continue

        chash = _content_hash(content)
        result.total_files += 1

        if not force and last_hashes.get(rel) == chash and db.get_file_entry(cache_key, last_hash, rel):
            result.reused_files += 1
            # Reuse the file entry AND symbols under the new hash, otherwise
            # file_entry_hashes(new_hash) never contains this file and it gets
            # re-parsed on every subsequent commit (cache thrash).
            old_entry = db.get_file_entry(cache_key, last_hash, rel)
            assert old_entry is not None
            db.upsert_file_entry(cache_key, head_hash, old_entry)
            file_syms = db.symbols_in_file(cache_key, last_hash, rel)
            if file_syms:
                db.bulk_upsert_symbols(cache_key, head_hash, file_syms)
                all_symbols.extend(file_syms)
            continue

        result.parsed_files += 1
        result.changed_files.append(rel)

        ast_json, symbols_json, symbols, byte_count = _parse_fached(fpath, language)
        entry = FileEntry(
            rel_path=rel,
            language=language,
            content_hash=chash,
            ast_json=ast_json,
            symbols_json=symbols_json,
            mtime=fpath.stat().st_mtime,
            byte_count=byte_count,
        )
        db.upsert_file_entry(cache_key, head_hash, entry)
        if symbols:
            db.bulk_upsert_symbols(cache_key, head_hash, symbols)
            all_symbols.extend(symbols)
        result.symbols_extracted += len(symbols)

    db.upsert_index_run(cache_key, head_hash, file_count=result.total_files)
    return result


def _parse_fached(path: Path, language: str):
    return _parse_file(path, language)
