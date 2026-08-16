"""Repository indexer — parses source files and caches AST/symbol metadata."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter_language_pack import get_parser

from codey.cache.ast_cache import CacheDB, FileEntry, SymbolRecord
from codey.index.languages import detect_language
from codey.index.symbols import extract_symbols
from codey.process import allowlist_env

__all__ = ["IndexResult", "index_repository", "git_head_hash", "list_repo_files"]

_DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "dist", "build", ".eggs", ".tox", ".cache", "htmlcov", ".idea", ".vscode",
}
_DEFAULT_IGNORE_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".class"}


@dataclass
class IndexResult:
    git_hash: str
    total_files: int = 0
    parsed_files: int = 0
    reused_files: int = 0
    skipped_files: int = 0
    symbols_extracted: int = 0
    changed_files: list[str] = field(default_factory=list)


def git_head_hash(repo_path: Path | str) -> str | None:
    """Current HEAD hash, or None if not a git repo (allowlisted env)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_path), capture_output=True,
            text=True, timeout=10, check=False, env=allowlist_env(),
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def list_repo_files(repo_path: Path) -> list[Path]:
    """List source files, respecting .gitignore via ``git ls-files`` (allowlisted env)."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(repo_path), capture_output=True, timeout=30, check=False, env=allowlist_env(),
        )
        if proc.returncode == 0:
            files = [
                (repo_path / f.decode("utf-8", errors="replace")).resolve()
                for f in proc.stdout.split(b"\x00") if f.strip()
            ]
            return [f for f in files if f.is_file()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return _walk_files(repo_path)


def _walk_files(repo_path: Path) -> list[Path]:
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


def _parse_file(path: Path, language: str) -> tuple[list[SymbolRecord], int]:
    raw = path.read_bytes()
    try:
        parser = get_parser(language)
        tree = parser.parse(raw)
    except Exception:
        return [], len(raw)
    return extract_symbols(tree, str(path.name), language), len(raw)


def index_repository(
    repo_path: Path | str,
    db: CacheDB,
    *,
    force: bool = False,
    cache_repo_path: Path | str | None = None,
) -> IndexResult:
    """Index a repo, reusing cached entries for unchanged files.

    ``cache_repo_path`` is the canonical repo used as the cache key when
    ``repo_path`` is a temporary worktree (non-HEAD reviews).
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
            old_entry = db.get_file_entry(cache_key, last_hash, rel)
            if old_entry is None:
                raise RuntimeError(
                    f"cache inconsistency: {rel} hashed equal under {last_hash} "
                    f"but its file entry is missing"
                )
            db.upsert_file_entry(cache_key, head_hash, old_entry)
            file_syms = db.symbols_in_file(cache_key, last_hash, rel)
            if file_syms:
                db.bulk_upsert_symbols(cache_key, head_hash, file_syms)
                all_symbols.extend(file_syms)
            continue

        result.parsed_files += 1
        result.changed_files.append(rel)

        symbols, byte_count = _parse_file(fpath, language)
        entry = FileEntry(rel, language, chash, fpath.stat().st_mtime, byte_count)
        db.upsert_file_entry(cache_key, head_hash, entry)
        if symbols:
            db.bulk_upsert_symbols(cache_key, head_hash, symbols)
            all_symbols.extend(symbols)
        result.symbols_extracted += len(symbols)

    db.upsert_index_run(cache_key, head_hash, file_count=result.total_files)
    return result
