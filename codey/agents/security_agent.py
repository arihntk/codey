"""SecurityAgent — deterministic secret detection + external tools + LLM synthesis."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from codey.agents.context import ReviewContext
from codey.agents.evidence import attach_evidence
from codey.agents.schemas import (
    AgentReport,
    Finding,
    FindingCategory,
    Severity,
    llm_output_is_parseable,
    parse_llm_findings,
    severity_weight,
    strip_code_fences,
)
from codey.agents.secrets import detect_hardcoded_secrets
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry
from codey.process import allowlist_env, scrubbed_env

__all__ = ["run_security_agent"]

_llm_output_is_parseable = llm_output_is_parseable
_strip_code_fences = strip_code_fences

# Files that obviously don't affect security. Config/data files stay in scope
# (.json/.yaml/.toml/.ini/.cfg/.env, HTML) — that's where credentials get committed.
_SKIP_SUFFIXES = {
    ".css", ".scss", ".sass", ".less", ".md", ".rst",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".bmp", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".lock",
}
_BANDIT_LANG = ".py"

# Low-signal bandit rules that fire on idiomatic test code (asserts, subprocess
# fixtures, fake credentials). They flood reviews with per-line noise when a test
# suite is committed, so they are dropped for test files only.
_BANDIT_TEST_NOISE = {"B101", "B105", "B404", "B603", "B607"}
_BANDIT_ID_PREFIX = "[bandit] "


def _is_test_path(path: str | None) -> bool:
    if not path:
        return False
    norm = path.lstrip("./")
    name = norm.rsplit("/", 1)[-1]
    return (
        norm.startswith("tests/")
        or norm.startswith("test/")
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _should_skip(path: str) -> bool:
    return Path(path).suffix.lower() in _SKIP_SUFFIXES


def _parse_llm_findings(text: str) -> tuple[list[Finding], str | None]:
    return parse_llm_findings(text, FindingCategory.SECURITY)


_SECURITY_SYSTEM = (
    "You are a senior security analyst. You review code changes for both "
    "application-security vulnerabilities AND confidentiality leaks.\n\n"
    "The deterministic detector already handles credentials; you must also "
    "judge confidentiality leaks that have no easy regex anchor:\n"
    "  • PII — emails, phones, addresses, national IDs, SSNs, account/record IDs,\n"
    "    user/client names embedded in source.\n"
    "  • Internal-only endpoints — hostnames, private IPs, staging/admin URLs,\n"
    "    hidden flags, internal API paths.\n"
    "  • Provider/account references — cloud account numbers, project IDs, ARNs.\n"
    "  • Crypto/auth mistakes — weak hashes, reused IVs, disabled TLS verify,\n"
    "    hardcoded JWT secrets, insecure randomness, SQL concat, command injection.\n"
    "  • Logging/telemetry — secrets or PII in logs, verbose errors, debug endpoints.\n\n"
    "For each genuine issue output a structured finding with: category=security,\n"
    "severity (calibrate honestly), title, description, file_path, line_start,\n"
    "evidence (VERBATIM copy of the code/tool line — never fabricate; if you cannot\n"
    "quote it, do NOT emit the finding), recommendation, confidence (0.0-1.0).\n\n"
    "Rules:\n"
    "  - You are given findings already produced by the deterministic detector.\n"
    "    Do NOT duplicate or downgrade them — they are high-precision. Keep them\n"
    "    as-is in your output and add your own.\n"
    "  - Only report genuine issues grounded in the provided data. Deduplicate.\n"
    "  - If there are no issues, output an empty array [].\n"
    "  - Output ONLY a JSON array of finding objects — no commentary, no fences."
)


def run_security_agent(ctx: ReviewContext, db=None, llm: object | None = None) -> AgentReport:
    repo = ctx.repo_path
    py_files = [p for p in ctx.changed_files if Path(p).suffix == _BANDIT_LANG]
    all_relevant = [p for p in ctx.changed_files if not _should_skip(p)]

    # 1. Deterministic hardcoded-secret detection (always runs, on the RAW diff).
    hardcoded = detect_hardcoded_secrets(ctx.raw_full_diff or ctx.full_diff, file_sources=ctx.file_sources)
    hardcoded_titles = {f.title for f in hardcoded}

    # 2. External scanners.
    raw_results: dict[str, str] = {}
    tool_errors: list[str] = []
    bandit_out, bandit_err = _run_bandit(repo, py_files)
    if bandit_out:
        raw_results["bandit"] = _filter_bandit_raw(bandit_out)
    if bandit_err:
        tool_errors.append(f"bandit: {bandit_err}")
    semgrep_out, semgrep_err = _run_semgrep(repo, all_relevant)
    if semgrep_out is not None:
        raw_results["semgrep"] = semgrep_out
    if semgrep_err:
        tool_errors.append(f"semgrep: {semgrep_err}")
    gitleaks_findings = _run_gitleaks(repo, commit=ctx.git_hash)
    if gitleaks_findings is not None:
        raw_results["gitleaks"] = gitleaks_findings

    tool_findings: list[Finding] = []
    for src in raw_results:
        tool_findings.extend(_parse_tool_results(src, raw_results[src]))

    token_usage = 0
    report_error: str | None = "; ".join(tool_errors) or None

    # 3. LLM judgement (always reviews the diff). Hardcoded findings merge after
    #    and survive LLM output — only an LLM finding whose title collides is dropped.
    llm_kept: list[Finding] = []
    if llm is not None:
        synthesised, _usage, synth_error = _llm_synthesise(
            llm, raw_results, ctx.full_diff, ctx.changed_files, hardcoded_findings=hardcoded,
        )
        token_usage = _usage
        if synth_error is not None:
            report_error = (report_error + "; " if report_error else "") + synth_error
        else:
            llm_findings, parse_error = _parse_llm_findings(synthesised)
            if parse_error:
                report_error = (report_error + "; " if report_error else "") + parse_error
            elif not _llm_output_is_parseable(synthesised):
                report_error = (report_error + "; " if report_error else "") + (
                    "LLM returned unparseable output (expected a JSON array of findings)"
                )
            llm_kept = [f for f in llm_findings if f.title not in hardcoded_titles]

    findings: list[Finding] = list(llm_kept)
    findings.extend(f for f in tool_findings if f.title not in hardcoded_titles)
    findings.extend(hardcoded)
    findings = _dedupe_findings(findings)

    if not findings and report_error is None:
        tool_evidence = "; ".join(f"{k}: 0 issues" for k in raw_results) or "no external tools ran"
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=Severity.INFO,
            title="No security issues found",
            description=(
                f"No vulnerabilities detected by the hardcoded-secret detector, "
                f"automated tools, or LLM review. Scanned {len(all_relevant)} relevant "
                f"file(s); hardcode check examined the full diff."
            ),
            evidence=tool_evidence,
            confidence=0.8,
        ))

    # Attach verbatim diff evidence, then ENFORCE the verbatim rule: LLM findings
    # that still have no evidence are discarded (the system prompt promises this).
    attach_evidence(findings, ctx)
    if llm_kept:
        kept_ids = {id(f) for f in llm_kept if f.evidence.strip()}
        findings = [f for f in findings if id(f) not in {id(x) for x in llm_kept} or id(f) in kept_ids]

    tools_used = list(raw_results) or ["no external security tools"]
    notable = [f for f in findings if f.severity != Severity.INFO]
    notable.sort(key=lambda f: (-severity_weight(f.severity), f.file_path or "", f.line_start or 0))
    finding_details = "; ".join(
        f"{f.title} [{f.severity.value}]"
        + (f" at {f.file_path}:{f.line_start}" if f.file_path else "")
        for f in notable[:8]
    )
    if len(notable) > 8:
        finding_details += f"; +{len(notable) - 8} more finding(s)"
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
    trunc_note = _diff_truncated_note(ctx.raw_full_diff or ctx.full_diff)
    if trunc_note:
        metadata["diff_truncated"] = trunc_note
        summary += f" Note: {trunc_note}"
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


def _run_bandit(repo: Path, py_files: list[str]) -> tuple[str, str | None]:
    if not py_files or not shutil.which("bandit"):
        return "", None
    try:
        cmd = ["bandit", "-f", "json", "-q", "--"] + py_files
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, timeout=60, check=False,
            env=scrubbed_env(),
        )
        if proc.returncode in (0, 1):
            return proc.stdout, None
        if proc.returncode in (2, 3):
            return "", (proc.stderr.strip() or "bandit exited fatally")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "", None


def _parse_bandit_json(raw: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings
    sev_map = {"LOW": Severity.LOW, "MEDIUM": Severity.MEDIUM, "HIGH": Severity.HIGH}
    conf_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}
    for r in data.get("results", []):
        cwe = r.get("issue_cwe", {})
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=sev_map.get(r.get("issue_severity", ""), Severity.MEDIUM),
            title=f"[bandit] {r.get('test_id', '')}: {r.get('issue_text', '')[:200]}",
            description=r.get("issue_text", ""),
            file_path=r.get("filename"),
            line_start=r.get("line_number"),
            evidence=r.get("code", ""),
            recommendation=cwe.get("link", "") if isinstance(cwe, dict) else "",
            confidence=conf_map.get(r.get("issue_confidence", ""), 0.5),
        ))
    return findings


# --- Semgrep ----


def _run_semgrep(repo: Path, files: list[str]) -> tuple[str | None, str | None]:
    if not shutil.which("semgrep") or not files:
        return None, None
    try:
        cmd = ["semgrep", "--json", "--quiet", "--no-rewrite-rule-ids", "--"] + files
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, timeout=90, check=False,
            env=scrubbed_env(),
        )
        if proc.returncode in (0, 1):
            return proc.stdout, None
        if proc.returncode > 1:
            return None, (proc.stderr.strip() or "semgrep exited abnormally")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None, None


def _parse_semgrep_json(raw: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return findings
    sev_map = {"INFO": Severity.INFO, "WARNING": Severity.MEDIUM, "ERROR": Severity.HIGH}
    for r in data.get("results", []):
        extra = r.get("extra", {})
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=sev_map.get(extra.get("severity", "").upper(), Severity.MEDIUM),
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
        parent_check = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{commit}~1"],
            cwd=str(repo), capture_output=True, text=True, timeout=15, check=False,
            env=allowlist_env(),
        )
        log_opts = f"{commit}~1..{commit}" if parent_check.returncode == 0 else commit
        proc = subprocess.run(
            ["gitleaks", "detect", "--source", str(repo), "--log-opts", log_opts,
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
        secret = r.get("Secret", "")
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=Severity.CRITICAL,
            title=f"[gitleaks] {r.get('RuleID', 'secret')}: potential secret detected",
            description=r.get("Description", "Potential secret or credential in code."),
            file_path=r.get("File"),
            line_start=r.get("StartLine"),
            line_end=r.get("EndLine"),
            evidence=secret[:100] + "..." if secret else "",
            recommendation="Remove the secret and rotate it immediately via env vars or a secrets manager.",
            confidence=0.9,
        ))
    return findings


def _parse_tool_results(source: str, raw: str) -> list[Finding]:
    parsers = {"bandit": _parse_bandit_json, "semgrep": _parse_semgrep_json, "gitleaks": _parse_gitleaks_json}
    return parsers[source](raw) if source in parsers else []


def _filter_bandit_raw(raw: str) -> str:
    """Drop low-signal bandit findings (asserts/subprocess/fake creds) in test files."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    results = data.get("results", [])
    data["results"] = [
        r for r in results
        if not (r.get("test_id") in _BANDIT_TEST_NOISE and _is_test_path(r.get("filename", "")))
    ]
    return json.dumps(data)


def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Collapse findings that target the same (category, file, line).

    Keeps the highest-severity (then highest-confidence) representative, so a
    secret reported by the detector, an external tool and the LLM surfaces once.
    """
    best: dict[tuple[str, str, int], Finding] = {}
    unlocated: list[Finding] = []
    for f in findings:
        if not f.file_path or f.line_start is None:
            unlocated.append(f)
            continue
        key = (f.category.value, f.file_path, f.line_start)
        prev = best.get(key)
        if prev is None or (severity_weight(f.severity), f.confidence) >= (
            severity_weight(prev.severity), prev.confidence
        ):
            best[key] = f
    return unlocated + list(best.values())


# --- LLM synthesis ----


def _hardcoded_summary(hardcoded: list[Finding]) -> str:
    if not hardcoded:
        return "(none)"
    out = []
    for i, f in enumerate(hardcoded, 1):
        loc = f"{f.file_path}:{f.line_start}" if f.file_path else "unknown"
        out.append(f"  {i}. [{f.severity.value.upper()}] {f.title} @ {loc}\n     evidence: {f.evidence}")
    return "\n".join(out)


def _llm_synthesise(
    llm: object,
    raw_results: dict[str, str],
    diff: str,
    changed_files: list[str],
    *,
    hardcoded_findings: list[Finding] | None = None,
) -> tuple[str, int, str | None]:
    """Returns ``(text, token_usage, error)``. Unparseable output is *not* an
    error here — the caller distinguishes it after parsing."""
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


def _diff_truncated_note(diff: str) -> str:
    if diff and len(diff) > 8000:
        return (
            f"LLM diff excerpt truncated to first 8000 chars of {len(diff)} "
            "(full raw diff scanned by the deterministic detector)."
        )
    return ""
