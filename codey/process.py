"""Environment helpers for subprocess isolation.

The LLM API key is passed directly to the model client and must never be
injected into ``os.environ``, because the security and test agents spawn
subprocesses (bandit, semgrep, gitleaks, pytest, npm, …) that execute code
from the repository under review — a malicious repo could read the key from
the inherited environment.

:func:`scrubbed_env` returns a copy of the current environment with provider
key variables (and anything that looks like a credential) removed, for use
as the ``env=`` argument to ``subprocess.run``.

Security model: a blocklist of credential-looking variable names is
whack-a-mole (PRIVATE_KEY, CREDENTIALS, AUTH, … are all missed). The correct
isolation for code that runs *repo-supplied* commands (git hooks / fsmonitor
via ``.git/config``, npm scripts, etc.) is an **allowlist** — only the
variables the subprocess genuinely needs survive. :func:`allowlist_env` is
used for those; ``scrubbed_env`` remains for tools that may need broader
environment access but must never see credentials.
"""

from __future__ import annotations

import os

__all__ = ["scrubbed_env", "allowlist_env"]

# Explicit provider credential variable names.
_CREDENTIAL_VARS = frozenset({
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
})

# Generic name fragments that indicate a credential variable.
_CREDENTIAL_FRAGMENTS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD")

# Variables subprocesses genuinely need. Everything else is dropped when
# running commands that could execute repo-supplied hooks or scripts.
_ALLOWLIST_PREFIXES = ("PATH",)
_ALLOWLIST_EXACT = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP", "USER", "SHELL", "TERM", "HOSTNAME")


def scrubbed_env() -> dict[str, str]:
    """Return ``os.environ`` without credential variables.

    Removes well-known provider key vars and any var whose name contains an
    API-key/token/secret/password fragment. Never raises.
    """
    env = dict(os.environ)
    for name in list(env):
        upper = name.upper()
        if upper in _CREDENTIAL_VARS:
            env.pop(name, None)
            continue
        if any(frag in upper for frag in _CREDENTIAL_FRAGMENTS):
            env.pop(name, None)
    return env


def allowlist_env() -> dict[str, str]:
    """Return a minimal environment for untrusted subprocess execution.

    Only PATH and a small set of locale/temp/user variables survive. Used
    when the subprocess may run code or hooks supplied by the repository
    under review (git fsmonitor/hooks via .git/config, npm/go/cargo test
    scripts) — credentials are excluded by construction rather than by
    enumerating credential names.
    """
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in _ALLOWLIST_EXACT or name.startswith(_ALLOWLIST_PREFIXES):
            env[name] = value
    return env
