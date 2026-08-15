"""High-precision hardcoded-secret leak detector for source code.

This module is a deterministic, zero-dependency first pass at finding
credentials / confidential tokens embedded directly in source code.  It is
designed to be:

* **high-precision** — prefix-based rules for well-known secret formats
  combined with Shannon-entropy checks to exclude obvious placeholders
  (`"changeme"`, `"your-key"`, `<API_KEY>`, empty strings).
* **non-exhaustive by design** — it intentionally covers secrets with
  recognisable prefixes AND a generic keyword/entropy heuristic.  It is
  explicitly **not** the last word: LLM judgement runs on top of it, and is
  expected to catch confidential-info leaks that have no easy regex anchor
  (PII, internal URLs, account numbers, provider-specific labels).
* **a fallback** — when the LLM is unavailable, its findings are the primary
  signal.  When the LLM is available, its findings seed the LLM synthesis as
  additional context (and the LLM is told to keep the findings already
  present and add its own).

Findings produced here use ``FindingCategory.SECURITY`` and carry evidence
copied verbatim from the supplied diff/file text.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from codey.agents.schemas import Finding, FindingCategory, Severity

__all__ = ["detect_hardcoded_secrets", "shannon_entropy"]

# ---------------------------------------------------------------------------
# Rule catalog
# ---------------------------------------------------------------------------

_MIN_SECRET_LEN = 8           # Tokens shorter than this are almost never secrets.
# Low floor: only rejects degenerate repetition strings ("aaaa…", entropy 0).
# Human-chosen passwords like "supersecretpass" (~2.8 bits/char) still pass.
_MIN_SECRET_ENTROPY = 2.0
_MAX_EVIDENCE_LEN = 160       # Truncate evidence lines to a sane width.


_SEVERITY_RANK = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


def _severity_rank(s: Severity) -> int:
    return _SEVERITY_RANK.get(s, 0)


@dataclass(frozen=True)
class _SecretRule:
    id: str
    label: str
    severity: Severity
    # Compiled regex with at least one capture group holding the secret value.
    pattern: re.Pattern[str]
    # Whether to apply an entropy check on the captured value (to filter
    # placeholders).  Prefix-matched secrets (sk-…, ghp_…) are reliable
    # enough that the entropy check is skipped.
    require_entropy: bool


# Well-known secret prefixes / formats.  Order matters only for tie-breaking
# the *title* — each line of the diff is checked against every rule.
_PREFIX_RULES: tuple[_SecretRule, ...] = (
    _SecretRule(
        id="openai_api_key",
        label="OpenAI API key",
        severity=Severity.CRITICAL,
        pattern=re.compile(
            r"\b(sk-[A-Za-z0-9_\-]{20,})\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="github_pat",
        label="GitHub personal access token",
        severity=Severity.CRITICAL,
        pattern=re.compile(
            r"(gh[pousr]_[A-Za-z0-9]{20,})\b|(github_pat_[A-Za-z0-9_]{22,})\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="aws_access_key_id",
        label="AWS access key ID",
        severity=Severity.CRITICAL,
        pattern=re.compile(
            r"\b((AKIA|ASIA|AGPA|AROA|AIDA|ANPA|ANVA|APKA)[0-9A-Z]{16})\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="aws_secret_access_key",
        label="AWS secret access key",
        severity=Severity.CRITICAL,
        # 40-char base64. This rule is intentionally checked *alongside* the
        # generic b64 rule; prefix rules win when both match.
        pattern=re.compile(
            r'''(?<=["'`])([A-Za-z0-9/+=]{40})(?=["'`])''',
        ),
        require_entropy=True,
    ),
    _SecretRule(
        id="slack_token",
        label="Slack token",
        severity=Severity.HIGH,
        pattern=re.compile(
            r"\b(xox[baprs]-[A-Za-z0-9\-]{10,})\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="google_api_key",
        label="Google API key",
        severity=Severity.HIGH,
        pattern=re.compile(
            r"\b(AIza[0-9A-Za-z_\-]{35})\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="google_oauth_id",
        label="Google OAuth client ID",
        severity=Severity.MEDIUM,
        pattern=re.compile(
            r"\b([0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com)\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="stripe_live_key",
        label="Stripe live secret key",
        severity=Severity.CRITICAL,
        pattern=re.compile(
            r"\b(sk_live_[A-Za-z0-9]{24,})|\b(rk_live_[A-Za-z0-9]{24,})\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="twilio_api_key",
        label="Twilio API key",
        severity=Severity.HIGH,
        pattern=re.compile(
            r"\b(SK[0-9a-fA-F]{32})\b",
        ),
        require_entropy=False,
    ),
    _SecretRule(
        id="jwt",
        label="JWT token",
        severity=Severity.HIGH,
        # Three dot-separated base64url segments; first segment small (header).
        pattern=re.compile(
            r"\b(eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})\b",
        ),
        require_entropy=False,
    ),
)


# Generic keyword-based confidential value rules.  These cover passwords,
# tokens, private keys, and similar *labelled* secret assignments across
# many languages.  Entropy + placeholder filtering is **mandatory** here so
# values like `password = ""` or `token = "your-token-here"` are not flagged
# as real leaks.
_KEYWORD_RULES: tuple[_SecretRule, ...] = (
    _SecretRule(
        id="password_literal",
        label="Hardcoded password",
        severity=Severity.HIGH,
        pattern=re.compile(
            # `password`, `passwd`, `pwd` = "..." | '...' | `...`
            r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*"
            r"([\"'`])(?P<val>[^\"'`]+)\1",
        ),
        require_entropy=True,
    ),
    _SecretRule(
        id="secret_literal",
        label="Hardcoded secret",
        severity=Severity.HIGH,
        pattern=re.compile(
            r"(?i)\b(?:secret|secret[_-]?key|client[_-]?secret)\b\s*[:=]\s*"
            r"([\"'`])(?P<val>[^\"'`]+)\1",
        ),
        require_entropy=True,
    ),
    _SecretRule(
        id="token_literal",
        label="Hardcoded auth token",
        severity=Severity.HIGH,
        pattern=re.compile(
            # `token`, `access_token`, `refresh_token`, `auth_token`, `bearer_token` = "..."
            r"(?i)\b(?:[a-z0-9_]*token|bearer)\b\s*[:=]\s*"
            r"([\"'`])(?P<val>[^\"'`]+)\1",
        ),
        require_entropy=True,
    ),
    _SecretRule(
        id="api_key_literal",
        label="Hardcoded API key",
        severity=Severity.HIGH,
        pattern=re.compile(
            r"(?i)\b(?:api[_-]?key|apikey|access[_-]?key)\b\s*[:=]\s*"
            r"([\"'`])(?P<val>[^\"'`]+)\1",
        ),
        require_entropy=True,
    ),
    _SecretRule(
        id="private_key_literal",
        label="Hardcoded private key",
        severity=Severity.CRITICAL,
        # PEM private keys (RSA / EC / OPENSSH / generic).  This is a *format*
        # match (BEGIN … END block) so no entropy gate is needed.
        pattern=re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
            r"[\s\S]{1,8000}?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
        ),
        require_entropy=False,
    ),
)


# Placeholders commonly used in examples / fixtures. Values matching any of
# these are never treated as real secrets, even if they coincidentally pass
# the entropy check.
_PLACEHOLDER_RE = re.compile(
    r"""(?x)
      ^(?:
        # pure layout tokens
        [\s<>{}\[\]()'"`,;:]* |
        # phrase-like placeholders
        your[_\-\s]?[a-z0-9_\-\s]* |
        (?:example|sample|test|demo|dummy|placeholder|fake|none|null|
            undefined|changeme|todo|fixme|default|xxx+|yyy+|abc+|123+) |
        # Sentinel placeholders composed only of `x`/`.`/`?`
        x+|\.+|\?+
      )$
      """,
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def shannon_entropy(s: str) -> float:
    """Return the Shannon entropy (bits per character) of *s*.

    ``0`` for empty strings.  Real secrets tend to be ≥ 3 bits/char;
    placeholders and short words tend to be < 3.
    """
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_placeholder(value: str) -> bool:
    if not value or not value.strip():
        return True
    stripped = value.strip()
    if len(stripped) < _MIN_SECRET_LEN:
        return True
    if _PLACEHOLDER_RE.match(stripped):
        return True
    return False


def _passes_entropy(value: str, rule: _SecretRule) -> bool:
    """For keyword rules: keep the value only if it's not a placeholder.

    Keyword rules match *labelled* assignments (``password = "..."`` etc.) —
    a labelled assignment with a non-placeholder value is suspicious even if
    the value is a low-entropy human word (``supersecretpass`` is a real
    password at ~2.8 bits/char). The entropy floor here only rejects
    degenerate repetition strings (``"aaaa…"``, entropy 0) that the
    placeholder regex doesn't catch.
    """
    if not rule.require_entropy:
        return True
    if _is_placeholder(value):
        return False
    return shannon_entropy(value) >= _MIN_SECRET_ENTROPY


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_EVIDENCE_LEN:
        return text
    return text[: _MAX_EVIDENCE_LEN - 1] + "…"


# ---------------------------------------------------------------------------
# Line context
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    file_path: str
    line_no: int
    text: str


def _iter_diff_lines(diff: str, *, changed_only: bool = True) -> Iterable[_Line]:
    """Yield ``_Line`` objects from a unified diff.

    ``changed_only=True`` restricts output to ``+`` lines (additions) since
    hardcoded secrets are introduced in additions; removed secrets are
    *good* and should not produce findings.
    """
    cur_file = ""
    new_line = 0
    for raw in diff.splitlines():
        if raw.startswith("diff --git "):
            cur_file = ""
            continue
        if raw.startswith("+++ b/"):
            cur_file = raw[6:].strip()
            continue
        if raw.startswith("@@"):
            # @@ -a,b +c,d @@  → c is the start of the new hunk.
            m = re.search(r"\+(\d+)(?:,(\d+))?", raw)
            if m:
                new_line = int(m.group(1)) - 1
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            new_line += 1
            if changed_only:
                yield _Line(cur_file, new_line, raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
            # Removal — do not emit.
            continue
        else:
            new_line += 1


def _make_finding(rule: _SecretRule, line: _Line, snippet: str, description: str) -> Finding:
    return Finding(
        category=FindingCategory.SECURITY,
        severity=rule.severity,
        title=f"[hardcoded] {rule.label} detected",
        description=description,
        file_path=line.file_path or None,
        line_start=line.line_no,
        evidence=snippet,
        recommendation=(
            "Remove the hardcoded credential from source. Load it from an "
            "environment variable or a secrets manager, and rotate the leaked "
            "value immediately."
        ),
        confidence=0.95 if not rule.require_entropy else 0.8,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_hardcoded_secrets(diff: str, *, file_sources: dict[str, str] | None = None) -> list[Finding]:
    """Scan *diff* (and optionally full file sources) for hardcoded secrets.

    Returns a de-duplicated list of :class:`Finding` objects. The scan is
    purely deterministic; results include verbatim ``evidence`` copied from
    the diff line. No findings are emitted for *removed* lines (a secret
    being deleted is good news).
    """
    if not diff and not file_sources:
        return []

    findings: list[Finding] = []
    # Keyed by (value, file, line) → keep the most severe finding for that
    # exact secret value on that exact line.  This is what prevents e.g.
    # `API_KEY = "sk-..."` from being reported twice (once by the OpenAI
    # prefix rule, once by the generic API-key keyword rule).
    by_value: dict[tuple[str, str, int], Finding] = {}
    seen_keys: set[tuple[str, str, int]] = set()

    def _maybe_keep(key: tuple[str, str, int], finding: Finding) -> None:
        existing = by_value.get(key)
        if existing is None or _severity_rank(finding.severity) > _severity_rank(existing.severity):
            by_value[key] = finding

    if diff:
        for line in _iter_diff_lines(diff, changed_only=True):
            for rule in (*_PREFIX_RULES, *_KEYWORD_RULES):
                m = rule.pattern.search(line.text)
                if not m:
                    continue

                # Extract the value for keyword rules; prefix rules use group 0.
                value = ""
                gd = m.groupdict()
                if "val" in gd and gd["val"]:
                    value = gd["val"]
                elif m.groups():
                    # Last non-empty capture group is the secret value.
                    value = next((g for g in reversed(m.groups()) if g), "") or ""
                else:
                    value = m.group(0)

                if not _passes_entropy(value, rule):
                    continue

                snippet = _truncate(line.text)
                dedup_key = (rule.id, snippet, line.line_no)
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                description = (
                    f"A {rule.label} appears to be hardcoded directly in the diff. "
                    f"This is a confidentiality risk: anyone with repository access "
                    f"can read and abuse the credential. Even if the value is a stub "
                    f"in a fixture or test, leaking plausible-looking credentials can "
                    f"still be dangerous and should be avoided."
                )
                finding = _make_finding(rule, line, snippet, description)
                _maybe_keep((value, line.file_path or "", line.line_no), finding)

    if file_sources:
        for path, source in file_sources.items():
            # Only scan full-source for PEM blocks (rare, expensive to regex
            # line-by-line on huge files). Prefix rules already covered by
            # scanning the diff; only catch private keys not present in diff.
            for rule in _KEYWORD_RULES:
                if rule.id != "private_key_literal":
                    continue
                for m in rule.pattern.finditer(source):
                    snippet = _truncate(m.group(0))
                    dedup_key = (rule.id, snippet, 0)
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)
                    line_no = source.count("\n", 0, m.start()) + 1
                    finding = _make_finding(
                        rule,
                        _Line(path, line_no, snippet),
                        snippet,
                        "A PEM private key block is embedded in this file. "
                        "Private keys must never live in source code.",
                    )
                    _maybe_keep((snippet, path, line_no), finding)

    findings = list(by_value.values())
    findings.sort(key=lambda f: (f.file_path or "", f.line_start or 0, f.title))
    return findings
