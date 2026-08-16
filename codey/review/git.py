"""Diff acquisition + commit operations (message amend, worktrees)."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from codey.process import allowlist_env

__all__ = [
    "CommitInfo", "resolve_commit", "get_latest_commit", "get_commit_diff",
    "get_changed_files", "get_commit_full_message", "amend_commit_message",
    "has_staged_changes", "materialize_commit", "remove_worktree",
]


@dataclass
class CommitInfo:
    hash: str
    message: str
    author: str


def _git(args: list[str], repo: Path) -> str:
    proc = subprocess.run(
        ["git"] + args, cwd=str(repo), capture_output=True, text=True,
        timeout=15, check=False, env=allowlist_env(),
    )
    return proc.stdout if proc.returncode == 0 else ""


def _git_proc(args: list[str], repo: Path, *, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args, cwd=str(repo), input=input_text, text=True, capture_output=True,
        timeout=15, check=False, env=allowlist_env(),
    )


def resolve_commit(repo: Path, ref: str) -> str | None:
    h = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], repo).strip()
    return h or None


def get_latest_commit(repo: Path, *, commit: str = "HEAD") -> CommitInfo | None:
    h = _git(["rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"], repo).strip()
    if not h:
        return None
    msg = _git(["log", "-1", "--pretty=%s", commit], repo).strip()
    author = _git(["log", "-1", "--pretty=%an", commit], repo).strip()
    return CommitInfo(hash=h, message=msg, author=author)


def get_changed_files(repo: Path, *, commit: str = "HEAD") -> list[str]:
    parent = _git(["rev-parse", f"{commit}~1"], repo).strip()
    if parent:
        raw = _git(["diff", "--name-only", f"{commit}~1", commit], repo)
    else:
        raw = _git(["show", "--name-only", "--pretty=format:", commit], repo)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def get_commit_diff(repo: Path, *, commit: str = "HEAD") -> dict[str, str]:
    parent = _git(["rev-parse", f"{commit}~1"], repo).strip()
    raw = _git(["diff", f"{commit}~1", commit], repo) if parent else _git(["show", "--pretty=format:", commit], repo)
    return _split_diff_by_file(raw)


def _split_diff_by_file(raw_diff: str) -> dict[str, str]:
    """Split a combined git diff into {new_path: diff_text} hunks.

    Deleted/renamed/binary files produce no ``+++ b/`` line, so the path is
    taken from the ``diff --git`` header instead. Paths may contain spaces or
    be C-style quoted — both forms are parsed.
    """
    result: dict[str, str] = {}
    if not raw_diff.strip():
        return result
    current_file = ""
    current_hunk: list[str] = []

    def _flush() -> None:
        if current_file and current_hunk:
            result[current_file] = "".join(current_hunk)

    for line in raw_diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            _flush()
            current_file = _header_new_path(line)
            current_hunk = [line]
        elif line.startswith("+++ "):
            p = line[4:].strip()
            if p.startswith("b/"):
                current_file = p[2:]
            current_hunk.append(line)
        else:
            current_hunk.append(line)

    _flush()
    return result


def _header_new_path(header_line: str) -> str:
    """Extract the b/ (new) path from a ``diff --git a/<old> b/<new>`` line.

    Handles unquoted paths with spaces and git's C-style quoting (``"b/foo bar"``,
    ``\\"`` escapes). The b-side is the last token, so match from the ``b/`` prefix.
    """
    rest = header_line[len("diff --git "):].strip()
    m = re.search(r'(?:^| )"?b/(.*)$', rest)
    if not m:
        tokens = rest.split()
        return tokens[-1][2:] if tokens and len(tokens[-1]) > 2 else ""
    new = m.group(1)
    if new.endswith('"'):
        new = new[:-1]
    return new.replace('\\"', '"').replace("\\\\", "\\")


def get_commit_full_message(repo: Path, *, commit: str = "HEAD") -> str:
    return _git(["log", "-1", "--pretty=%B", commit], repo)


def materialize_commit(repo: Path, commit_hash: str) -> Path:
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="codey-review-"))
    proc = _git_proc(["worktree", "add", "--detach", str(tmpdir), commit_hash], repo)
    if proc.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(
            f"Could not create worktree for {commit_hash[:12]}: "
            f"{proc.stderr.strip() or 'git worktree add failed'}"
        )
    return tmpdir


def remove_worktree(repo: Path, worktree: Path) -> None:
    import shutil

    _git_proc(["worktree", "remove", "--force", str(worktree)], repo)
    shutil.rmtree(worktree, ignore_errors=True)


def has_staged_changes(repo: Path) -> bool:
    return _git_proc(["diff", "--cached", "--quiet"], repo).returncode != 0


def amend_commit_message(repo: Path, new_message: str) -> tuple[bool, str]:
    if has_staged_changes(repo):
        return False, "staged changes present; refusing to amend commit message"
    proc = _git_proc(["commit", "--amend", "-F", "-"], repo, input_text=new_message)
    if proc.returncode != 0:
        return False, (proc.stderr.strip() or "git commit --amend failed")
    return True, "amended"
