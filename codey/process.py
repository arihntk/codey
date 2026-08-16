"""Subprocess environment isolation.

The LLM API key is passed directly to the model client, never injected into
``os.environ``, because the security/test agents spawn subprocesses that execute
repo code — a malicious repo could read the key from the inherited environment.

``scrubbed_env`` removes credential-looking variables; ``allowlist_env`` keeps
only what the subprocess genuinely needs (used for git hooks/npm/test scripts).
"""

from __future__ import annotations

import os

__all__ = ["scrubbed_env", "allowlist_env"]

_CREDENTIAL_VARS = frozenset({
    "OPENAI_API_KEY", "OPENAI_ORG_ID", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
    "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
})
_CREDENTIAL_FRAGMENTS = ("API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD")
_ALLOWLIST_PREFIXES = ("PATH",)
_ALLOWLIST_EXACT = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP", "USER", "SHELL", "TERM", "HOSTNAME")


def scrubbed_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in list(env):
        upper = name.upper()
        if upper in _CREDENTIAL_VARS or any(frag in upper for frag in _CREDENTIAL_FRAGMENTS):
            env.pop(name, None)
    return env


def allowlist_env() -> dict[str, str]:
    return {
        name: value for name, value in os.environ.items()
        if name in _ALLOWLIST_EXACT or name.startswith(_ALLOWLIST_PREFIXES)
    }
