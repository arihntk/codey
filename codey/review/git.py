"""Compute diffs between a commit and its first parent and the list of changed files."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from codey.process import allowlist_env

__all__ = [
    "CommitInfo",
    "resolve_commit",
    "get_latest_commit",
    "get_commit_diff",
    "get_changed_files",
    "get_commit_full_message",
    "amend_commit_message",
    "has_staged_changes",
    "materialize_commit",
]


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
        env=allowlist_env(),
    )
    return proc.stdout if proc.returncode == 0 else ""


def _git_proc(args: list[str], repo: Path, *, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=str(repo),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
        env=allowlist_env(),
    )


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
    """Split a combined git diff into per-file diff hunks.

    Keys are the *new* file path (the ``b/`` side). Files that are deleted,
    renamed without content changes, or changed as binary produce no
    ``+++ b/`` line — the file name is taken from the ``diff --git`` header
    instead so those changes are never silently invisible.
    """
    result: dict[str, str] = {}
    if not raw_diff.strip():
        return result
    lines = raw_diff.splitlines(keepends=True)
    current_file: str = ""
    current_hunk: list[str] = []

    def _flush() -> None:
        if current_file and current_hunk:
            result[current_file] = "".join(current_hunk)

    for line in lines:
        if line.startswith("diff --git "):
            _flush()
            # diff --git a/<old> b/<new>
            rest = line[len("diff --git "):]
            parts = rest.split()
            new_path = parts[1][2:] if len(parts) > 1 else ""
            if new_path == "/dev/null":
                # Deletion: the diff belongs to the removed file.
                new_path = parts[0][2:] if parts else ""
            current_file = new_path
            current_hunk = [line]
        elif line.startswith("+++ "):
            # +++ b/<path> or +++ /dev/null — normalize to b/ side.
            p = line[4:].strip()
            if p.startswith("b/"):
                current_file = p[2:]
            elif p == "/dev/null":
                pass  # deletion; current_file already set from header
            current_hunk.append(line)
        elif line.startswith("--- "):
            current_hunk.append(line)
        else:
            current_hunk.append(line)

    _flush()
    return result


def get_commit_full_message(repo: Path, *, commit: str = "HEAD") -> str:
    """Return the full commit message (subject + body) for the given commit."""
    return _git(["log", "-1", "--pretty=%B", commit], repo)


def materialize_commit(repo: Path, commit_hash: str) -> Path:
    """Create a temporary detached worktree of *commit_hash* and return its path.

    Used to review non-HEAD commits against their own tree (indexing,
    chunking, and tooling must see the commit's files, not the working tree).
    The caller is responsible for removing the worktree afterwards via
    :func:`remove_worktree`.

    Raises:
        RuntimeError: if the worktree could not be created.
    """
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="codey-review-"))
    proc = _git_proc(
        ["worktree", "add", "--detach", str(tmpdir), commit_hash],
        repo,
    )
    if proc.returncode != 0:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(
            f"Could not create worktree for {commit_hash[:12]}: "
            f"{proc.stderr.strip() or 'git worktree add failed'}"
        )
    return tmpdir


def remove_worktree(repo: Path, worktree: Path) -> None:
    """Remove a temporary worktree created by :func:`materialize_commit`."""
    import shutil

    _git_proc(["worktree", "remove", "--force", str(worktree)], repo)
    shutil.rmtree(worktree, ignore_errors=True)


def has_staged_changes(repo: Path) -> bool:
    """True when there are changes staged in the index (vs HEAD)."""
    proc = _git_proc(["diff", "--cached", "--quiet"], repo)
    return proc.returncode != 0


def amend_commit_message(repo: Path, new_message: str) -> tuple[bool, str]:
    """Amend HEAD's commit message to ``new_message``.

    Refuses to amend when there are staged changes (so the commit tree is not
    altered unexpectedly). Returns ``(ok, message)`` where ``message`` is either
    a short success note or the captured stderr on failure.
    """
    if has_staged_changes(repo):
        return False, "staged changes present; refusing to amend commit message"

    # No staged changes at this point (guarded above), so --amend only rewrites
    # the message — the commit tree is preserved.
    proc = _git_proc(
        ["commit", "--amend", "-F", "-"],
        repo,
        input_text=new_message,
    )
    if proc.returncode != 0:
        return False, (proc.stderr.strip() or "git commit --amend failed")
    return True, "amended"
