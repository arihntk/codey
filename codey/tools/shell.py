"""Allow-listed shell tools for agent use (grep, cat, ls, git).

These are exposed to LangGraph agents as ``@tool``-decorated functions.
The allow-list prevents arbitrary command execution while giving agents the
read-only inspection capabilities they need for code review.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from codey.process import scrubbed_env

__all__ = [
    "ShellResult",
    "run_grep",
    "run_cat",
    "run_ls",
    "run_git",
    "list_tools",
    "build_tools_for_agents",
]

RepoPath = str | Path


@dataclass
class ShellResult:
    """Result of a shell tool invocation."""

    ok: bool
    stdout: str
    stderr: str = ""
    returncode: int = 0

    def __str__(self) -> str:
        return self.stdout if self.ok else f"[exit {self.returncode}] {self.stderr}"

    def truncate(self, max_chars: int = 20_000) -> ShellResult:
        if len(self.stdout) <= max_chars:
            return self
        return ShellResult(
            ok=self.ok,
            stdout=self.stdout[:max_chars] + "\n... [truncated]",
            stderr=self.stderr,
            returncode=self.returncode,
        )


# --- Allow-list ----------------------------------------------------------

_GIT_SUBCOMMANDS = {
    "log", "diff", "show", "status", "branch", "blame", "ls-files",
    "rev-parse", "name-rev", "config --get",
}

_MAX_OUTPUT = 100_000


def _validate_repo(repo: RepoPath) -> Path:
    p = Path(repo).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def _run(args: list[str], *, cwd: Path, timeout: int = 30) -> ShellResult:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=scrubbed_env(),
        )
        stdout = proc.stdout[:_MAX_OUTPUT]
        if len(proc.stdout) > _MAX_OUTPUT:
            stdout += "\n... [truncated]"
        return ShellResult(
            ok=proc.returncode == 0,
            stdout=stdout,
            stderr=proc.stderr[:_MAX_OUTPUT],
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(ok=False, stdout="", stderr="Command timed out", returncode=-1)
    except FileNotFoundError as e:
        return ShellResult(ok=False, stdout="", stderr=str(e), returncode=-1)


def _binary_available(name: str) -> bool:
    return shutil.which(name) is not None


# --- Individual tools ----------------------------------------------------


def run_grep(
    pattern: str,
    repo: RepoPath = ".",
    *,
    glob: str = "",
    ignore_case: bool = False,
    context: int = 0,
) -> ShellResult:
    """Search file contents with ripgrep (preferred) or grep fallback."""
    repo_path = _validate_repo(repo)
    args: list[str] = []
    if _binary_available("rg"):
        args = ["rg", "--no-heading", "-n", "--max-count=500"]
        if ignore_case:
            args.append("-i")
        if context:
            args.append(f"-C{context}")
        if glob:
            args += ["-g", glob]
        args += [pattern, str(repo_path)]
    elif _binary_available("grep"):
        args = ["grep", "-rn"]
        if ignore_case:
            args.append("-i")
        if context:
            args.append(f"-C{context}")
        if glob:
            args += ["--include", glob]
        args += [pattern, str(repo_path)]
    else:
        return ShellResult(ok=False, stdout="", stderr="No grep/rg binary found", returncode=-1)
    return _run(args, cwd=repo_path).truncate()


def run_cat(file_path: str, repo: RepoPath = ".") -> ShellResult:
    """Read a file's contents (path relative to repo root)."""
    repo_path = _validate_repo(repo)
    target = (repo_path / file_path).resolve()
    try:
        # prevent path traversal
        target.relative_to(repo_path)
    except ValueError:
        return ShellResult(ok=False, stdout="", stderr="Path escapes repo root", returncode=-1)
    if not target.is_file():
        return ShellResult(ok=False, stdout="", stderr=f"Not a file: {file_path}", returncode=-1)
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ShellResult(ok=False, stdout="", stderr=str(e), returncode=-1)
    if len(content) > _MAX_OUTPUT:
        content = content[:_MAX_OUTPUT] + "\n... [truncated]"
    return ShellResult(ok=True, stdout=content)


def run_ls(path: str = "", repo: RepoPath = ".") -> ShellResult:
    """List directory contents (path relative to repo root)."""
    repo_path = _validate_repo(repo)
    target = (repo_path / path).resolve() if path else repo_path
    try:
        target.relative_to(repo_path)
    except ValueError:
        return ShellResult(ok=False, stdout="", stderr="Path escapes repo root", returncode=-1)
    if not target.exists():
        return ShellResult(ok=False, stdout="", stderr=f"Not found: {path}", returncode=-1)
    if target.is_dir():
        entries = sorted(
            (e.name + "/" if (target / e.name).is_dir() else e.name)
            for e in target.iterdir()
            if not e.name.startswith(".")
        )
        return ShellResult(ok=True, stdout="\n".join(entries))
    return ShellResult(ok=True, stdout=target.name)


def run_git(args: list[str], repo: RepoPath = ".") -> ShellResult:
    """Run an allow-listed git subcommand."""
    repo_path = _validate_repo(repo)
    if not args:
        return ShellResult(ok=False, stdout="", stderr="No git args provided", returncode=-1)
    sub = args[0]
    joined = " ".join(args)
    if sub not in _GIT_SUBCOMMANDS and joined not in _GIT_SUBCOMMANDS:
        return ShellResult(
            ok=False,
            stdout="",
            stderr=f"Disallowed git subcommand: {joined}",
            returncode=-1,
        )
    full_args = ["git"] + args
    return _run(full_args, cwd=repo_path).truncate()


# --- LangGraph tool adapters --------------------------------------------


def list_tools() -> list[str]:
    return ["run_grep", "run_cat", "run_ls", "run_git"]


def build_tools_for_agents(repo: RepoPath) -> list:
    """Return langchain @tool wrappers bound to a specific repo.

    Returned tools are pure functions suitable for ``create_react_agent``.
    """
    from langchain_core.tools import tool

    @tool
    def grep(pattern: str, glob: str = "", ignore_case: bool = False, context: int = 0) -> str:
        """Search file contents for a regex pattern. Use glob to filter files (e.g. '*.py').

        Args:
            pattern: Regular expression to search for.
            glob: Optional file pattern filter (e.g. "*.py").
            ignore_case: Case-insensitive search.
            context: Number of context lines around matches.
        """
        return str(run_grep(pattern, repo, glob=glob, ignore_case=ignore_case, context=context))

    @tool
    def cat(file_path: str) -> str:
        """Read a file's contents. Pass a path relative to the repo root."""
        return str(run_cat(file_path, repo))

    @tool
    def ls(path: str = "") -> str:
        """List directory contents. Pass a path relative to the repo root, or empty for root."""
        return str(run_ls(path, repo))

    @tool
    def git(args: str) -> str:
        """Run a read-only git subcommand. Pass args as a space-separated string (e.g. "log --oneline -5")."""
        parsed = args.split()
        return str(run_git(parsed, repo))

    return [grep, cat, ls, git]
