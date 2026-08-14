"""Compute diffs between a commit and its first parent and the list of changed files."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CommitInfo", "resolve_commit", "get_latest_commit", "get_commit_diff", "get_changed_files"]


@dataclass
class CommitInfo:
    hash: str
    message: str
    author: str


def _git(args: list[str], repo: Path) -> str:
    proc = subprocess.run(
        ["git"] + args,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def resolve_commit(repo: Path, ref: str) -> str | None:
    """Resolve a commit ref to its full hash, or ``None`` if it doesn't exist.

    Accepts any git revision spec accepted by ``git rev-parse``: a full or
    abbreviated hash, a branch/tag name, ``HEAD``, ``HEAD~3``, etc.
    """
    h = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], repo).strip()
    return h or None


def get_latest_commit(repo: Path, *, commit: str = "HEAD") -> CommitInfo | None:
    """Get info (hash, message, author) for the given commit ref."""
    h = _git(["rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"], repo).strip()
    if not h:
        return None
    msg = _git(["log", "-1", "--pretty=%s", commit], repo).strip()
    author = _git(["log", "-1", "--pretty=%an", commit], repo).strip()
    return CommitInfo(hash=h, message=msg, author=author)


def get_changed_files(repo: Path, *, commit: str = "HEAD") -> list[str]:
    """Return the list of file paths changed in the given commit (vs its first parent)."""
    # Try diff against parent (covers normal commits).
    parent = _git(["rev-parse", f"{commit}~1"], repo).strip()
    if parent:
        raw = _git(["diff", "--name-only", f"{commit}~1", commit], repo)
    else:
        # Initial commit: show all files in the commit.
        raw = _git(["show", "--name-only", "--pretty=format:", commit], repo)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def get_commit_diff(repo: Path, *, commit: str = "HEAD") -> dict[str, str]:
    """Get per-file diffs for a commit, keyed by file path.

    Returns a mapping of {file_path: diff_text}.
    """
    parent = _git(["rev-parse", f"{commit}~1"], repo).strip()
    if parent:
        raw = _git(["diff", f"{commit}~1", commit], repo)
    else:
        raw = _git(["show", "--pretty=format:", commit], repo)

    return _split_diff_by_file(raw)


def _split_diff_by_file(raw_diff: str) -> dict[str, str]:
    """Split a combined git diff into per-file diff hunks."""
    result: dict[str, str] = {}
    if not raw_diff.strip():
        return result
    lines = raw_diff.splitlines(keepends=True)
    current_file: str = ""
    current_hunk: list[str] = []

    for line in lines:
        if line.startswith("diff --git "):
            if current_file and current_hunk:
                result[current_file] = "".join(current_hunk)
            current_file = ""
            current_hunk = [line]
        elif line.startswith("+++ b/"):
            current_file = line[6:].rstrip("\n")
            current_hunk.append(line)
        elif line.startswith("--- a/"):
            current_hunk.append(line)
        else:
            current_hunk.append(line)

    if current_file and current_hunk:
        result[current_file] = "".join(current_hunk)

    return result
