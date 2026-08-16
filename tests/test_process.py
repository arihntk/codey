"""Tests for codey.process — environment isolation for subprocesses."""

from __future__ import annotations

from codey.process import allowlist_env, scrubbed_env


def test_scrubbed_env_removes_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("MY_CUSTOM_TOKEN", "secret")
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setenv("KEEP_ME", "safe")
    env = scrubbed_env()
    assert "OPENAI_API_KEY" not in env
    assert "MY_CUSTOM_TOKEN" not in env
    assert "DB_PASSWORD" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["KEEP_ME"] == "safe"


def test_scrubbed_env_keeps_non_credential_vars(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/u")
    env = scrubbed_env()
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/u"


def test_allowlist_env_only_keeps_safe_vars(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("LANG", "C")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("SOME_APP_CONFIG", "should-drop")
    env = allowlist_env()
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/u"
    assert env["LANG"] == "C"
    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "SOME_APP_CONFIG" not in env


def test_allowlist_env_never_leaks_credentials_by_construction(monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", "-----BEGIN...")
    monkeypatch.setenv("CREDENTIALS", "json")
    monkeypatch.setenv("AUTH_TOKEN", "x")
    env = allowlist_env()
    assert "PRIVATE_KEY" not in env
    assert "CREDENTIALS" not in env
    assert "AUTH_TOKEN" not in env
