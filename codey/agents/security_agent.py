"""SecurityAgent — multi-tool security analysis with LLM synthesis.

Runs bandit (Python), semgrep (multi-language), and gitleaks (secrets) against
the changed files. Skips file types that obviously don't affect security (css,
md, images, fonts, etc.). Feeds raw results to the LLM which synthesises
structured findings with evidence and recommendations.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from codey.agents.context import ReviewContext
from codey.agents.schemas import AgentReport, Finding, FindingCategory, Severity

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
    "You are a senior security analyst. Review the following outputs from "
    "automated security tools (bandit, semgrep, gitleaks) for a code change. "
    "For each genuine issue, produce a structured finding with:\n"
    "- category: security\n"
    "- severity: critical/high/medium/low/info\n"
    "- title: short description\n"
    "- description: what the vulnerability is\n"
    "- file_path and line_start\n"
    "- evidence: the relevant code or tool output\n"
    "- recommendation: how to fix it\n"
    "- confidence: 0.0-1.0\n"
    "Only report genuine issues. Deduplicate. If no issues, say 'No security issues found.' "
    "Output as a JSON array of finding objects."
)


def run_security_agent(
    ctx: ReviewContext,
    db=None,
    llm: object | None = None,
) -> AgentReport:
    """Run security scanners and synthesise results into an AgentReport."""
    repo = ctx.repo_path
    # Filter to security-relevant files.
    py_files = [p for p in ctx.changed_files if Path(p).suffix == _BANDIT_LANG]
    all_relevant = [p for p in ctx.changed_files if not _should_skip(p)]

    raw_results: dict[str, str] = {}
    bandit_findings = _run_bandit(repo, py_files)
    if bandit_findings:
        raw_results["bandit"] = bandit_findings

    semgrep_findings = _run_semgrep(repo, all_relevant)
    if semgrep_findings is not None:
        raw_results["semgrep"] = semgrep_findings

    gitleaks_findings = _run_gitleaks(repo)
    if gitleaks_findings is not None:
        raw_results["gitleaks"] = gitleaks_findings

    findings: list[Finding] = []

    # Parse raw tool results into initial findings.
    for src in raw_results:
        tool_findings = _parse_tool_results(src, raw_results[src])
        findings.extend(tool_findings)

    token_usage = 0

    # LLM synthesis: refine raw findings into structured findings.
    if llm is not None and findings:
        synthesised = _llm_synthesise(llm, raw_results, ctx.full_diff, ctx.changed_files)
        token_usage = len(synthesised) // 4
        refined = _parse_llm_findings(synthesised)
        if refined:
            findings = refined
    elif llm is not None and not findings:
        # No raw findings; quick LLM scan of diff itself.
        synthesised = _llm_scan_diff(llm, ctx.full_diff, ctx.changed_files)
        token_usage = len(synthesised) // 4
        refined = _parse_llm_findings(synthesised)
        if refined:
            findings = refined

    if not findings:
        findings.append(Finding(
            category=FindingCategory.SECURITY,
            severity=Severity.INFO,
            title="No security issues found",
            description="No security vulnerabilities detected by automated tools or LLM review.",
            confidence=0.8,
        ))

    tools_used = [k for k in raw_results] or ["none (no security tools available)"]
    return AgentReport(
        agent="security",
        status="completed",
        summary=f"Security scan using {', '.join(tools_used)}. Found {len(findings)} finding(s).",
        findings=findings,
        metadata={k: v[:2000] for k, v in raw_results.items()},
        token_usage=token_usage,
    )


# --- Bandit ----


def _run_bandit(repo: Path, py_files: list[str]) -> str:
    if not py_files:
        return ""
    binary = shutil.which("bandit") or shutil.which("python") and "python -m bandit"
    if not shutil.which("bandit"):
        try:
            import bandit
            binary = None
        except ImportError:
            return ""
    try:
        cmd = ["bandit", "-f", "json", "-q"] + py_files
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, timeout=60, check=False,
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
        cmd = ["semgrep", "--json", "--quiet", "--no-rewrite-rule-ids"]
        # scan changed paths
        cmd.extend(files[:50])
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, timeout=90, check=False,
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


def _run_gitleaks(repo: Path) -> str | None:
    if not shutil.which("gitleaks"):
        return None
    try:
        # Scan unstaged + staged changes (diff).
        proc = subprocess.run(
            ["gitleaks", "detect", "--source", str(repo), "--no-git", "--report-format", "json", "--report-path", "-"],
            cwd=str(repo), capture_output=True, text=True, timeout=60, check=False,
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
            recommendation="Remove the secret and rotate it immediately. Use environment variables or a secrets manager.",
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


def _llm_synthesise(llm: object, raw_results: dict[str, str], diff: str, changed_files: list[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    raw_text = "\n\n".join(f"=== {k} ===\n{v[:5000]}" for k, v in raw_results.items())
    diff_excerpt = diff[:8000] if diff else "(no diff)"
    files_str = ", ".join(changed_files[:30])
    try:
        response = llm.invoke([
            SystemMessage(content=_SECURITY_SYSTEM),
            HumanMessage(content=(
                f"Changed files: {files_str}\n\n"
                f"Raw tool results:\n{raw_text}\n\n"
                f"Diff:\n```diff\n{diff_excerpt}\n```\n\n"
                "Output the findings as a JSON array:"
            )),
        ])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        return f"[] // LLM synthesis failed: {e}"


def _llm_scan_diff(llm: object, diff: str, changed_files: list[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    diff_excerpt = diff[:12000] if diff else "(no diff)"
    files_str = ", ".join(changed_files[:30])
    prompt = (
        "No automated security tool findings were reported. "
        "Review the diff below for security issues and output findings as a JSON array. "
        "If truly no issues, output []."
    )
    try:
        response = llm.invoke([
            SystemMessage(content=_SECURITY_SYSTEM + "\n\n" + prompt),
            HumanMessage(content=f"Changed files: {files_str}\n\nDiff:\n```diff\n{diff_excerpt}\n```"),
        ])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        return f"[] // LLM scan failed: {e}"


def _parse_llm_findings(text: str) -> list[Finding]:
    """Parse LLM JSON output into Finding objects."""
    # Strip markdown code fences if present.
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        data = json.loads(text)
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