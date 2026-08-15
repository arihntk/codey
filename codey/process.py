"""Environment helpers for subprocess isolation.

The LLM API key is passed directly to the model client and must never be
injected into ``os.environ``, because the security and test agents spawn
subprocesses (bandit, semgrep, gitleaks, pytest, npm, …) that execute code
from the repository under review — a malicious repo could read the key from
the inherited environment.

:func:`scrubbed_env` returns a copy of the current environment with provider
key variables (and anything that looks like a credential) removed, for use
as the ``env=`` argument to ``subprocess.run``.
"""

from __future__ import annotations

import os

__all__ = ["scrubbed_env"]

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
