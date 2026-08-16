"""Tests for codey.review.git — diff acquisition, commit ops, worktrees."""

from __future__ import annotations

import subprocess

from codey.review.git import (
    CommitInfo,
    _header_new_path,
    _split_diff_by_file,
    amend_commit_message,
    get_changed_files,
    get_commit_diff,
    get_commit_full_message,
    get_latest_commit,
    has_staged_changes,
    materialize_commit,
    remove_worktree,
    resolve_commit,
)
from tests.conftest import commit, init_repo


def test_resolve_commit(repo):
    full = resolve_commit(repo, "HEAD")
    assert full is not None
    assert len(full) == 40
    assert resolve_commit(repo, full[:7]) == full  # abbreviated
    assert resolve_commit(repo, "does-not-exist") is None


def test_get_latest_commit(repo):
    info = get_latest_commit(repo)
    assert isinstance(info, CommitInfo)
    assert info.hash == resolve_commit(repo, "HEAD")
    assert info.message == "add mul function"
    assert info.author == "Test"


def test_get_changed_files(repo):
    files = get_changed_files(repo)
    assert "main.py" in files


def test_get_commit_diff_modify(repo):
    diffs = get_commit_diff(repo)
    assert "main.py" in diffs
    assert "def mul" in diffs["main.py"]


def test_get_commit_diff_initial_commit(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("x=1\n", encoding="utf-8")
    commit(r, "init")
    # Only one commit -> initial-commit path.
    diffs = get_commit_diff(r)
    assert "a.py" in diffs


def test_split_diff_by_file_empty():
    assert _split_diff_by_file("") == {}
    assert _split_diff_by_file("   \n") == {}


def test_header_new_path_spaces_and_quotes():
    assert _header_new_path('diff --git a/my file.py b/my file.py') == "my file.py"
    assert _header_new_path('diff --git "a/my file.py" "b/my file.py"') == "my file.py"
    assert _header_new_path('diff --git "a/weird\\"name.py" "b/weird\\"name.py"') == 'weird"name.py'
    assert _header_new_path("diff --git a/x b/y") == "y"


def test_get_commit_diff_deletion(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "a.py").write_text("x=1\n", encoding="utf-8")
    (r / "keep.py").write_text("y=1\n", encoding="utf-8")
    commit(r, "init")
    subprocess.run(["git", "rm", "-q", "a.py"], cwd=str(r), check=True)
    commit(r, "delete")
    diffs = get_commit_diff(r)
    assert "a.py" in diffs  # deleted file is not invisible
    assert "keep.py" not in diffs


def test_get_commit_diff_binary(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "b.bin").write_bytes(b"\x00\x01")
    commit(r, "init")
    (r / "b.bin").write_bytes(b"\x00\x01\x02\x03")
    commit(r, "change bin")
    diffs = get_commit_diff(r)
    assert "b.bin" in diffs
    assert "Binary files" in diffs["b.bin"]


def test_get_commit_diff_rename_with_space(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    init_repo(r)
    (r / "old name.py").write_text("x=1\n", encoding="utf-8")
    commit(r, "init")
    subprocess.run(["git", "mv", "old name.py", "new name.py"], cwd=str(r), check=True)
    commit(r, "rename")
    diffs = get_commit_diff(r)
    assert "new name.py" in diffs


def test_get_commit_full_message(repo):
    assert get_commit_full_message(repo).strip() == "add mul function"


def test_has_staged_changes(repo):
    assert has_staged_changes(repo) is False
    (repo / "new.py").write_text("x=1\n")
    subprocess.run(["git", "add", "new.py"], cwd=str(repo), check=True)
    assert has_staged_changes(repo) is True


def test_amend_commit_message(repo):
    ok, info = amend_commit_message(repo, "new message body")
    assert ok, info
    assert get_commit_full_message(repo).strip() == "new message body"


def test_amend_refuses_with_staged_changes(repo):
    (repo / "new.py").write_text("x=1\n")
    subprocess.run(["git", "add", "new.py"], cwd=str(repo), check=True)
    ok, info = amend_commit_message(repo, "should not happen")
    assert ok is False
    assert "staged" in info


def test_materialize_and_remove_worktree(repo):
    head = resolve_commit(repo, "HEAD~1")
    wt = materialize_commit(repo, head)
    try:
        assert (wt / "main.py").is_file()
        # The worktree checks out the OLD tree (before mul was added).
        assert "def mul" not in (wt / "main.py").read_text(encoding="utf-8")
    finally:
        remove_worktree(repo, wt)
    assert not wt.exists()


def test_materialize_commit_invalid_hash_raises(repo):
    import pytest

    with pytest.raises(RuntimeError):
        materialize_commit(repo, "0" * 40)
