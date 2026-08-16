"""Deterministic hardcoded-secret detector (zero dependencies, high precision)."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from codey.agents.schemas import Finding, FindingCategory, Severity

__all__ = ["detect_hardcoded_secrets", "shannon_entropy"]

_MIN_SECRET_LEN = 8  # shorter tokens are almost never secrets
_MIN_SECRET_ENTROPY = 2.0  # only rejects degenerate repetition ("aaaa…")
_MAX_EVIDENCE_LEN = 160

_SEVERITY_RANK = {
    Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1, Severity.INFO: 0,
}


def _severity_rank(s: Severity) -> int:
    return _SEVERITY_RANK.get(s, 0)


@dataclass(frozen=True)
class _SecretRule:
    id: str
    label: str
    severity: Severity
    pattern: re.Pattern[str]
    require_entropy: bool


# Well-known secret prefixes/formats. Prefix rules skip the entropy check.
_PREFIX_RULES: tuple[_SecretRule, ...] = (
    _SecretRule("openai_api_key", "OpenAI API key", Severity.CRITICAL,
                re.compile(r"\b(sk-[A-Za-z0-9_\-]{20,})\b"), False),
    _SecretRule("github_pat", "GitHub personal access token", Severity.CRITICAL,
                re.compile(r"(gh[pousr]_[A-Za-z0-9]{20,})\b|(github_pat_[A-Za-z0-9_]{22,})\b"), False),
    _SecretRule("aws_access_key_id", "AWS access key ID", Severity.CRITICAL,
                re.compile(r"\b((AKIA|ASIA|AGPA|AROA|AIDA|ANPA|ANVA|APKA)[0-9A-Z]{16})\b"), False),
    _SecretRule("aws_secret_access_key", "AWS secret access key", Severity.CRITICAL,
                re.compile(r'''(?<=["'`])([A-Za-z0-9/+=]{40})(?=["'`])'''), True),
    _SecretRule("slack_token", "Slack token", Severity.HIGH,
                re.compile(r"\b(xox[baprs]-[A-Za-z0-9\-]{10,})\b"), False),
    _SecretRule("google_api_key", "Google API key", Severity.HIGH,
                re.compile(r"\b(AIza[0-9A-Za-z_\-]{35})\b"), False),
    _SecretRule("google_oauth_id", "Google OAuth client ID", Severity.MEDIUM,
                re.compile(r"\b([0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com)\b"), False),
    _SecretRule("stripe_live_key", "Stripe live secret key", Severity.CRITICAL,
                re.compile(r"\b(sk_live_[A-Za-z0-9]{24,})|\b(rk_live_[A-Za-z0-9]{24,})\b"), False),
    _SecretRule("twilio_api_key", "Twilio API key", Severity.HIGH,
                re.compile(r"\b(SK[0-9a-fA-F]{32})\b"), False),
    _SecretRule("jwt", "JWT token", Severity.HIGH,
                re.compile(r"\b(eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})\b"), False),
)

# Generic keyword-based confidential-value rules (labelled assignments). Entropy
# + placeholder filtering is mandatory so `password = ""` isn't flagged.
def _kw(rule_id: str, label: str, sev: Severity, keywords: str) -> _SecretRule:
    return _SecretRule(
        rule_id, label, sev,
        re.compile(rf"(?i)\b(?:{keywords})\b\s*[:=]\s*([\"'`])(?P<val>[^\"'`]+)\1"),
        True,
    )


_KEYWORD_RULES: tuple[_SecretRule, ...] = (
    _kw("password_literal", "Hardcoded password", Severity.HIGH, "password|passwd|pwd"),
    _kw("secret_literal", "Hardcoded secret", Severity.HIGH, "secret|secret[_-]?key|client[_-]?secret"),
    _kw("token_literal", "Hardcoded auth token", Severity.HIGH, "[a-z0-9_]*token|bearer"),
    _kw("api_key_literal", "Hardcoded API key", Severity.HIGH, "api[_-]?key|apikey|access[_-]?key"),
    _SecretRule("private_key_literal", "Hardcoded private key", Severity.CRITICAL,
                re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
                           r"[\s\S]{1,8000}?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"), False),
)

_PLACEHOLDER_RE = re.compile(
    r"""(?x)
      ^(?:
        [\s<>{}\[\]()'"`,;:]* |
        your[_\-\s]?[a-z0-9_\-\s]* |
        (?:example|sample|test|demo|dummy|placeholder|fake|none|null|
            undefined|changeme|todo|fixme|default|xxx+|yyy+|abc+|123+) |
        x+|\.+|\?+
      )$
      """,
    re.IGNORECASE,
)


def shannon_entropy(s: str) -> float:
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
    return bool(_PLACEHOLDER_RE.match(stripped))


def _passes_entropy(value: str, rule: _SecretRule) -> bool:
    if not rule.require_entropy:
        return True
    if _is_placeholder(value):
        return False
    return shannon_entropy(value) >= _MIN_SECRET_ENTROPY


def _truncate(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _MAX_EVIDENCE_LEN else text[: _MAX_EVIDENCE_LEN - 1] + "…"


@dataclass
class _Line:
    file_path: str
    line_no: int
    text: str


def _iter_diff_lines(diff: str, *, changed_only: bool = True) -> Iterable[_Line]:
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
            m = re.search(r"\+(\d+)(?:,(\d+))?", raw)
            if m:
                new_line = int(m.group(1)) - 1
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            new_line += 1
            if changed_only:
                yield _Line(cur_file, new_line, raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
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


def detect_hardcoded_secrets(diff: str, *, file_sources: dict[str, str] | None = None) -> list[Finding]:
    if not diff and not file_sources:
        return []

    findings: list[Finding] = []
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
                gd = m.groupdict()
                if "val" in gd and gd["val"]:
                    value = gd["val"]
                elif m.groups():
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
                    f"can read and abuse the credential. Even a stub in a fixture or "
                    f"test is dangerous and should be avoided."
                )
                _maybe_keep(
                    (value, line.file_path or "", line.line_no),
                    _make_finding(rule, line, snippet, description),
                )

    if file_sources:
        for path, source in file_sources.items():
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
                    _maybe_keep((snippet, path, line_no), _make_finding(
                        rule, _Line(path, line_no, snippet), snippet,
                        "A PEM private key block is embedded in this file. "
                        "Private keys must never live in source code.",
                    ))

    findings = list(by_value.values())
    findings.sort(key=lambda f: (f.file_path or "", f.line_start or 0, f.title))
    return findings
