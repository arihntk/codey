"""Shared fixtures and fakes for the codey test suite."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory, monkeypatch):
    """Point the cache and config at temp dirs for every test.

    The DB path and config path are module-level constants in ``codey``, so
    in addition to setting the env vars we reset the constants in case the
    modules were already imported (frozen constants would otherwise point at
    the real home dir).
    """
    import sys

    cache_dir = tmp_path_factory.mktemp("codey-cache")
    config_dir = tmp_path_factory.mktemp("codey-config")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))

    ast_cache = sys.modules.get("codey.cache.ast_cache")
    if ast_cache is not None:
        ast_cache.DEFAULT_DB_PATH = cache_dir / "codey" / "codey.db"
    store = sys.modules.get("codey.config.store")
    if store is not None:
        store.CONFIG_DIR = config_dir / "codey"
        store.CONFIG_FILE = store.CONFIG_DIR / "config.json"
    yield


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(repo: Path, *args: str) -> str:
    """Run a git command in *repo* and return stdout (no error checking)."""
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True,
    )
    return proc.stdout


def init_repo(repo: Path) -> None:
    """git init + a deterministic identity so commits work everywhere."""
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)


def commit(repo: Path, message: str) -> None:
    """Stage everything and commit."""
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=str(repo), check=True)


@pytest.fixture
def repo(tmp_path):
    """A git repo with two commits over a single Python file (has a diff)."""
    r = tmp_path / "test-repo"
    r.mkdir()
    init_repo(r)
    (r / "main.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def sub(a, b):\n"
        "    return add(a, -b)\n",
        encoding="utf-8",
    )
    commit(r, "initial")
    (r / "main.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def sub(a, b):\n"
        "    return add(a, -b)\n\n"
        "def mul(a, b):\n"
        "    return a * b\n",
        encoding="utf-8",
    )
    commit(r, "add mul function")
    return r


# ---------------------------------------------------------------------------
# LLM fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal stand-in for a langchain AIMessage."""

    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class RateLimitError(Exception):
    """A retryable 429 with a Retry-After header."""

    def __init__(self, retry_after="0"):
        super().__init__("rate limited")
        self.status_code = 429
        self.response = _Resp(429, {"Retry-After": retry_after})


class ServerError(Exception):
    """A retryable 5xx."""

    def __init__(self):
        super().__init__("boom")
        self.status_code = 500
        self.response = _Resp(500, {})


class _Resp:
    def __init__(self, status_code, headers):
        self.status_code = status_code
        self.headers = headers


class FakeLLM:
    """A chat-model fake that returns canned content after optional failures."""

    def __init__(self, content="", usage=None, failures=0, error_factory=RateLimitError):
        self.content = content
        self.usage = usage or {"input_tokens": 10, "output_tokens": 5}
        self.failures = failures
        self.error_factory = error_factory
        self.calls = []

    def invoke(self, messages, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            raise self.error_factory()
        self.calls.append(messages)
        return FakeResponse(self.content, dict(self.usage))
