"""SecurityAgent — multi-tool security analysis with LLM synthesis.

Runs (in order):
1. A deterministic hardcoded-secret detector (no third-party dependency;
   prefix rules + Shannon-entropy placeholder filtering). This is the
   primary source of truth for *credential* leaks and the authoritative
   fallback when the LLM is unavailable.
2. Bandit (Python), semgrep (multi-language), gitleaks (secrets) — if
   installed.
3. An LLM pass that synthesises the tool outputs *and* specifically judges
   *non-secret confidentiality leaks* that have no easy regex anchor —
   sensitive PII, internal URLs/hostnames, account/record IDs, business-
   confidential strings, debug/telemetry payloads, etc.

Hardcoded-secret findings survive the LLM step: the LLM is told not to drop
them and to add its own judgement on top.  If the LLM is absent, the
hardcoded-secret findings are returned as-is.

Skips file types that obviously don't affect security (css, md, images,
fonts, etc.) for the external-tool stage; the diff is always reviewed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from codey.agents.context import ReviewContext
from codey.agents.evidence import attach_evidence
from codey.agents.schemas import AgentReport, Finding, FindingCategory, Severity
from codey.agents.secrets import detect_hardcoded_secrets
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry
from codey.process import scrubbed_env

__all__ = ["run_security_agent"]

# Files that obviously don't affect security.
_SKIP_SUFFIXES = {
    ".css", ".scss", ".sass", ".less",
    ".md", ".rst", ".txt", ".txt",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".bmp", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".zip", ".tar", ".gz", ".bz2",
    ".lock", ".toml", ".cfg", ".ini",
    ".json", ".yaml", ".yml",
    ".html",
}
_BANDIT_LANG = ".py"


def _should_skip(path: str) -> bool:
    return Path(path).suffix.lower() in _SKIP_SUFFIXES


_SECURITY_SYSTEM = (
    "You are a senior security analyst. You review code changes for both "
    "Classic application-security vulnerabilities AND confidentiality leaks.\n\n"
    "Your scope is explicitly MORE than just hardcoded secrets — the "
    "deterministic detector already handles credentials. You must also "
    "JUDGE confidentiality leaks that have no easy regex anchor, such as:\n"
    "  • PII — emails, phone numbers, addresses, national IDs, SSNs,\n"
    "    account/record IDs, user/client names embedded in source.\n"
    "  • Internal-only endpoints — internal hostnames, private IP ranges,\n"
    "    staging/admin URLs, hidden feature flags, internal API paths.\n"
    "  • Provider/account references — cloud account numbers, project IDs,\n"
    "    resource ARNs, org/workspace IDs, customer IDs.\n"
    "  • Crypto / auth mistakes — weak hashes (MD5/SHA1 for passwords),\n"
    "    reused IVs, disabled TLS verification, hardcoded JWT secrets,\n"
    "    insecure randomness, SQL string concatenation, command injection.\n"
    "  • Logging/telemetry — secrets or PII written to logs/metrics, verbose\n"
    "    error messages leaking stack/state, debug endpoints left enabled.\n"
    "  • Any other disclosure that could harm users or the organisation.\n\n"
    "For each genuine issue produce a structured finding with:\n"
    "- category: security\n"
    "- severity: critical/high/medium/low/info (calibrate honestly — "
    "real secret EMERGENCY = critical; leaked PII = high/medium depending "
    "on volume & sensitivity; risky pattern = medium/low)\n"
    "- title: short description\n"
    "- description: what the vulnerability is and why it matters\n"
    "- file_path and line_start (from the tool output when available)\n"
    "- evidence: VERBATIM copy of the relevant code line / tool output. "
    "NEVER fabricate or paraphrase evidence. If you cannot quote the exact "
    "text, do NOT emit the finding.\n"
    "- recommendation: how to fix it\n"
    "- confidence: 0.0-1.0\n\n"
    "IMPORTANT rules:\n"
    "  - You will be given a list of findings already produced by the "
    "deterministic detector. Do NOT duplicate or downgrade them — they are "
    "high-precision. You MAY keep them as-is in your output and add your own.\n"
    "  - Only report genuine issues grounded in the provided data. Deduplicate.\n"
    "  - If there are no issues, output an empty array [].\n"
    "  - Output ONLY a JSON array of finding objects — no commentary, no "
    "markdown fences."
)


def run_security_agent(
    ctx: ReviewContext,
    db=None,
    llm: object | None = None,
) -> AgentReport:
    """Run security scanners and synthesise results into an AgentReport.

    Pipeline:
      1. ``detect_hardcoded_secrets`` (deterministic, no external deps) on
         the diff + optional full file sources — the authoritative source
         for credential leaks AND the guaranteed fallback when the LLM is
         unavailable.
      2. bandit / semgrep / gitleaks — if installed.
      3. LLM synthesis that ingests the hardcoded-secret findings AND the
         tool outputs AND the diff, and is explicitly asked to add its own
         *confidentiality-judgement* findings beyond secrets.
    """
    repo = ctx.repo_path
    py_files = [p for p in ctx.changed_files if Path(p).suffix == _BANDIT_LANG]
    all_relevant = [p for p in ctx.changed_files if not _should_skip(p)]

    # 1. Deterministic hardcoded-secret detection (always runs). Scans the
    #    RAW diff (pre-summarisation) so LLM summaries can't hide secrets.
    hardcoded = detect_hardcoded_secrets(ctx.raw_full_diff or ctx.full_diff, file_sources=ctx.file_sources)
    hardcoded_titles = {f.title for f in hardcoded}

    # 2. External scanners.
    raw_results: dict[str, str] = {}
    bandit_findings = _run_bandit(repo, py_files)
    if bandit_findings:
        raw_results["bandit"] = bandit_findings
    semgrep_findings = _run_semgrep(repo, all_relevant)
    if semgrep_findings is not None:
        raw_results["semgrep"] = semgrep_findings
    gitleaks_findings = _run_gitleaks(repo, commit=ctx.git_hash)
    if gitleaks_findings is not None:
        raw_results["gitleaks"] = gitleaks_findings

    tool_findings: list[Finding] = []
    for src in raw_results:
        tool_findings.extend(_parse_tool_results(src, raw_results[src]))

    token_usage = 0
    report_error: str | None = None

    # 3. LLM judgement. The LLM always reviews the diff (when available) —
    #    even if no tool flagged anything — so non-secret confidentiality
    #    leaks get judged.  Hardcoded-secret findings are merged *after* and
    #    survive LLM output (only an LLM finding that exactly matches a
    #    hardcoded one is dropped, to avoid duplicates).
    llm_findings: list[Finding] = []
    if llm is not None:
        synthesised, _usage, synth_error = _llm_synthesise(
            llm,
            raw_results,
            ctx.full_diff,
            ctx.changed_files,
            hardcoded_findings=hardcoded,
        )
        token_usage = _usage
        if synth_error is not None:
            report_error = synth_error
        else:
            llm_findings = _parse_llm_findings(synthesised)
            if not _llm_output_is_parseable(synthesised):
                report_error = "LLM returned unparseable output (expected a JSON array of findings)"

    # Merge: LLM findings first (broader confidentiality judgement), then
    # tool findings, then hardcoded-secret findings — dropping any LLM/tool
    # finding whose title collides with a hardcoded finding.
    findings: list[Finding] = []
    for f in llm_findings:
        if f.title in hardcoded_titles:
            continue
        findings.append(f)
    for f in tool_findings:
        if f.title in hardcoded_titles:
            continue
        findings.append(f)
    findings.extend(hardcoded)

    if not findings and report_error is None:
        tool_evidence = "; ".join(f"{k}: 0 issues" for k in raw_results) or "no external tools ran"
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=Severity.INFO,
            title="No security issues found",
            description=(
                f"No security vulnerabilities detected by the hardcoded-secret "
                f"detector, automated tools, or LLM review. "
                f"Scanned {len(all_relevant)} relevant file(s); hardcode check "
                f"examined the full diff."
            ),
            evidence=tool_evidence,
            confidence=0.8,
        ))

    # Attach verbatim diff evidence to any LLM findings that lack it.
    attach_evidence(findings, ctx)

    tools_used = [k for k in raw_results] or ["no external security tools"]
    finding_details = "; ".join(
        f"{f.title} [{f.severity.value}]"
        + (f" at {f.file_path}:{f.line_start}" if f.file_path else "")
        for f in findings
        if f.severity != Severity.INFO
    )
    summary = (
        f"Security scan using {', '.join(tools_used)} + hardcoded-secret detector"
        + (", LLM confidentiality review" if llm is not None else "")
        + f". Scanned {len(all_relevant)} relevant file(s) and the full diff. "
        f"Found {len(findings)} finding(s)."
        + (f" Issues: {finding_details}." if finding_details else "")
    )
    if report_error is not None:
        summary += f" Warning: LLM review failed ({report_error})."
    metadata = {k: v[:2000] for k, v in raw_results.items()}
    if hardcoded:
        metadata["hardcoded_secrets"] = f"{len(hardcoded)} finding(s) from deterministic detector"
    return AgentReport(
        agent="security",
        status="error" if report_error is not None else "completed",
        summary=summary,
        findings=findings,
        metadata=metadata,
        token_usage=token_usage,
        error=report_error,
    )


# --- Bandit ----


def _run_bandit(repo: Path, py_files: list[str]) -> str:
    if not py_files:
        return ""
    if not shutil.which("bandit"):
        return ""
    try:
        # '--' separates options from file paths so repo filenames that begin
        # with '-' can't be interpreted as flags.
        cmd = ["bandit", "-f", "json", "-q", "--"] + py_files
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, timeout=60, check=False,
            env=scrubbed_env(),
        )
        if proc.returncode in (0, 1):  # 1 = issues found, still valid output
            return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _parse_bandit_json(raw: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings
    for r in data.get("results", []):
        sev_map = {"LOW": Severity.LOW, "MEDIUM": Severity.MEDIUM, "HIGH": Severity.HIGH}
        conf_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=sev_map.get(r.get("issue_severity", ""), Severity.MEDIUM),
            title=f"[bandit] {r.get('test_id', '')}: {r.get('issue_text', '')[:200]}",
            description=r.get("issue_text", ""),
            file_path=r.get("filename"),
            line_start=r.get("line_number"),
            evidence=r.get("code", ""),
            recommendation=r.get("issue_cwe", {}).get("link", "") if isinstance(r.get("issue_cwe"), dict) else "",
            confidence=conf_map.get(r.get("issue_confidence", ""), 0.5),
        ))
    return findings


# --- Semgrep ----


def _run_semgrep(repo: Path, files: list[str]) -> str | None:
    if not shutil.which("semgrep"):
        return None
    if not files:
        return None
    try:
        # '--' separates options from file paths so repo filenames that begin
        # with '-' can't be interpreted as flags.
        cmd = ["semgrep", "--json", "--quiet", "--no-rewrite-rule-ids", "--"]
        # scan changed paths
        cmd.extend(files[:50])
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, timeout=90, check=False,
            env=scrubbed_env(),
        )
        if proc.returncode in (0, 1):
            return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _parse_semgrep_json(raw: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings
    for r in data.get("results", []):
        sev = Severity.MEDIUM
        extra = r.get("extra", {})
        sev_str = extra.get("severity", "").upper()
        sev_map = {"INFO": Severity.INFO, "WARNING": Severity.MEDIUM, "ERROR": Severity.HIGH}
        sev = sev_map.get(sev_str, sev)
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=sev,
            title=f"[semgrep] {r.get('check_id', '')}",
            description=extra.get("message", ""),
            file_path=r.get("path"),
            line_start=r.get("start", {}).get("line"),
            line_end=r.get("end", {}).get("line"),
            evidence=extra.get("lines", ""),
            recommendation=extra.get("fix", ""),
            confidence=0.7,
        ))
    return findings


# --- Gitleaks ----


def _run_gitleaks(repo: Path, *, commit: str = "HEAD") -> str | None:
    if not shutil.which("gitleaks"):
        return None
    try:
        # Scan only the commit under review (diff vs its parent), not the
        # entire working tree — pre-existing secrets in unrelated files must
        # not be attributed to this review. Falls back to a plain tree scan
        # if log-opts isn't supported by the installed gitleaks.
        log_opts = f"{commit}~1..{commit}"
        if commit == "HEAD":
            import subprocess as _sp

            head = _sp.run(
                ["git", "rev-parse", "--verify", "--quiet", "HEAD~1"],
                cwd=str(repo), capture_output=True, text=True, timeout=15, check=False,
            )
            if head.returncode != 0:
                log_opts = commit  # initial commit has no parent
        proc = subprocess.run(
            ["gitleaks", "detect", "--source", str(repo),
             "--log-opts", log_opts,
             "--report-format", "json", "--report-path", "-"],
            cwd=str(repo), capture_output=True, text=True, timeout=60, check=False,
            env=scrubbed_env(),
        )
        if proc.returncode in (0, 1):
            return proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _parse_gitleaks_json(raw: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings
    for r in data:
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=Severity.CRITICAL,
            title=f"[gitleaks] {r.get('RuleID', 'secret')}: potential secret detected",
            description=r.get("Description", "Potential secret or credential in code."),
            file_path=r.get("File"),
            line_start=r.get("StartLine"),
            line_end=r.get("EndLine"),
            evidence=r.get("Secret", "")[:100] + "..." if r.get("Secret") else "",
            recommendation="Remove the secret and rotate it immediately via env vars or a secrets manager.",
            confidence=0.9,
        ))
    return findings


# --- Tool result dispatcher ----


def _parse_tool_results(source: str, raw: str) -> list[Finding]:
    if source == "bandit":
        return _parse_bandit_json(raw)
    elif source == "semgrep":
        return _parse_semgrep_json(raw)
    elif source == "gitleaks":
        return _parse_gitleaks_json(raw)
    return []


# --- LLM synthesis ----


def _hardcoded_summary(hardcoded: list[Finding]) -> str:
    """Render the deterministic-secret findings as plain text for the LLM.

    The LLM is instructed (via the system prompt) not to drop or downgrade
    these — they are high-precision.  We surface them so the LLM doesn't
    have to re-derive them from the raw diff, which both saves tokens and
    keeps the final list consistent.
    """
    if not hardcoded:
        return "(none)"
    out = []
    for i, f in enumerate(hardcoded, 1):
        loc = f"{f.file_path}:{f.line_start}" if f.file_path else "unknown"
        out.append(
            f"  {i}. [{f.severity.value.upper()}] {f.title} @ {loc}\n"
            f"     evidence: {f.evidence}"
        )
    return "\n".join(out)


def _llm_synthesise(
    llm: object,
    raw_results: dict[str, str],
    diff: str,
    changed_files: list[str],
    *,
    hardcoded_findings: list[Finding] | None = None,
) -> tuple[str, int, str | None]:
    """Ask the LLM to merge tool outputs + hardcoded-secret findings and
    ADD its own confidentiality-judgement findings for issues that have no
    easy regex anchor (PII, internal endpoints, crypto/auth mistakes, etc.).

    The LLM always reviews the diff — regardless of whether bandit/semgrep/
    gitleaks produced output — so confidential-info leaks are judged, not
    just secrets.

    Returns ``(text, token_usage, error)`` where ``error`` is set when the
    LLM call itself failed (rate limit exhausted, provider error, timeout).
    Unparseable output is *not* treated as an error here — the caller
    distinguishes it after parsing.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    raw_text = "\n\n".join(f"=== {k} ===\n{v[:5000]}" for k, v in raw_results.items()) or "(no external tool output)"
    diff_excerpt = diff[:8000] if diff else "(no diff)"
    files_str = ", ".join(changed_files[:30])
    hardcoded_text = _hardcoded_summary(hardcoded_findings or [])
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_SECURITY_SYSTEM),
            HumanMessage(content=(
                f"Changed files: {files_str}\n\n"
                f"Deterministic hardcoded-secret findings (already confirmed — "
                f"keep them in your output, do not re-derive or downgrade):\n"
                f"{hardcoded_text}\n\n"
                f"Raw tool results:\n{raw_text}\n\n"
                f"Diff:\n```diff\n{diff_excerpt}\n```\n\n"
                "Output the findings as a single JSON array. Include the "
                "hardcoded-secret findings above verbatim AND add your own "
                "for any other confidentiality or security issue you find."
            )),
        ])
        text = extract_text(response)
        return text, response_tokens(response, fallback_text=text), None
    except Exception as e:
        return "", 0, str(e)


def _parse_llm_findings(text: str) -> list[Finding]:
    """Parse LLM JSON output into Finding objects.

    Returns ``[]`` both for a valid empty array and for unparseable output.
    Callers that need to distinguish the two should check
    :func:`_llm_output_is_parseable` first.
    """
    stripped = _strip_code_fences(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    findings: list[Finding] = []
    for item in data:
        try:
            findings.append(Finding(
                category=FindingCategory.SECURITY,
                severity=Severity(item.get("severity", "info").lower()),
                title=item.get("title", ""),
                description=item.get("description", ""),
                file_path=item.get("file_path"),
                line_start=item.get("line_start"),
                line_end=item.get("line_end"),
                evidence=item.get("evidence", ""),
                recommendation=item.get("recommendation", ""),
                confidence=float(item.get("confidence", 0.7)),
            ))
        except (ValueError, TypeError):
            continue
    return findings


def _llm_output_is_parseable(text: str) -> bool:
    """True when *text* parses as a JSON array (possibly empty)."""
    try:
        json.loads(_strip_code_fences(text))
        return True
    except json.JSONDecodeError:
        return False


def _strip_code_fences(text: str) -> str:
    """Remove an enclosing markdown ``` code fence, if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    return stripped
