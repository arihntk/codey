"""TestAgent — identifies and executes the test suite for changed code.

Detects test framework via config files (pytest.ini, pyproject.toml, package.json,
etc.) and runs the relevant commands. Skips if no concrete test command can be
identified. Optionally asks the LLM to determine which specific tests to run
based on the changed files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from codey.agents.context import ReviewContext
from codey.agents.schemas import AgentReport, Finding, FindingCategory, Severity
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry
from codey.process import scrubbed_env

__all__ = ["run_test_agent"]

_TEST_RUN_TIMEOUT = 120  # seconds

_TEST_CONFIGS = {
    "pytest": [
        ("pytest.ini", ["pytest", "-x", "--tb=short"]),
        ("pyproject.toml", ["pytest", "-x", "--tb=short"]),
        ("setup.cfg", ["pytest", "-x", "--tb=short"]),
        ("tox.ini", ["pytest", "-x", "--tb=short"]),
    ],
    "unittest": [
        ("conftest.py", ["python", "-m", "pytest", "-x", "--tb=short"]),
    ],
    "npm": [
        ("package.json", ["npm", "test"]),
    ],
    "go": [
        ("go.mod", ["go", "test", "./..."]),
    ],
    "cargo": [
        ("Cargo.toml", ["cargo", "test"]),
    ],
    "rake": [
        ("Rakefile", ["rake", "test"]),
    ],
}


def _detect_frameworks(repo: Path) -> list[tuple[str, list[str]]]:
    """Return a list of (framework_name, test_command) pairs detected in the repo."""

    detected: list[tuple[str, list[str]]] = []
    for framework, configs in _TEST_CONFIGS.items():
        for config_file, cmd in configs:
            target = repo / config_file
            if target.is_file():
                # For pyproject.toml, verify pytest is actually configured.
                if config_file == "pyproject.toml":
                    content = target.read_text(encoding="utf-8", errors="replace")
                    if "[tool.pytest" not in content and "pytest" not in content:
                        continue
                detected.append((framework, cmd))
                break

    # Fallback: if there's a tests/ directory and pytest is available.
    if not detected and any((repo / "tests").glob("test_*.py")) or any((repo / "test").glob("test_*.py")):
        if shutil.which("pytest"):
            detected.append(("pytest", ["pytest", "-x", "--tb=short"]))

    return detected


def _binary_available(cmd: list[str]) -> bool:
    return shutil.which(cmd[0]) is not None


def _run_tests(repo: Path, command: list[str]) -> tuple[bool, str, str]:
    """Run a test command and return (success, stdout, stderr)."""

    if not _binary_available(command):
        return False, "", f"'{command[0]}' not found in PATH"
    try:
        proc = subprocess.run(
            command,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=_TEST_RUN_TIMEOUT,
            check=False,
            env=scrubbed_env(),
        )
        return proc.returncode == 0, proc.stdout[:20_000], proc.stderr[:20_000]
    except subprocess.TimeoutExpired:
        return False, "", f"Test command timed out after {_TEST_RUN_TIMEOUT}s"
    except (OSError, FileNotFoundError) as e:
        return False, "", str(e)


def run_test_agent(
    ctx: ReviewContext,
    db=None,
    llm: object | None = None,
) -> AgentReport:
    """Identify and run tests for the changed code.

    IMPORTANT: test commands are executed ONLY when ``ctx.run_tests`` is
    True (set via ``codey review --run-tests`` after an explicit
    confirmation). Running ``npm test`` / ``go test ./...`` on a repository
    being reviewed executes code from that repository — reviewing untrusted
    code must never trigger build/test script execution implicitly.
    """

    if not ctx.run_tests:
        return AgentReport(
            agent="test",
            status="skipped",
            summary=(
                "Test execution disabled. Re-run with `codey review --run-tests` "
                "(and confirm the prompt) to execute the detected test commands."
            ),
            findings=[
                Finding(
                    category=FindingCategory.TESTING,
                    severity=Severity.INFO,
                    title="Test execution skipped (opt-in)",
                    description=(
                        "Test commands are not executed by default because they "
                        "run code from the repository under review. Enable with "
                        "`codey review --run-tests` after confirming the command "
                        "list is safe."
                    ),
                    confidence=1.0,
                )
            ],
        )

    repo = ctx.repo_path
    frameworks = _detect_frameworks(repo)

    findings: list[Finding] = []
    metadata: dict[str, str] = {}
    token_usage = 0

    if not frameworks:
        findings.append(Finding(
            category=FindingCategory.TESTING,
            severity=Severity.INFO,
            title="No test suite detected",
            description=(
                "No concrete test commands could be identified for this repository "
                "(no pytest.ini, pyproject.toml [tool.pytest], package.json test script, "
                "go.mod, Cargo.toml, or tests/ directory with test_*.py)."
            ),
            confidence=0.9,
        ))
        return AgentReport(
            agent="test",
            status="skipped",
            summary="No test suite detected — test execution skipped.",
            findings=findings,
            metadata={},
        )

    # Use LLM to narrow test scope if available.
    test_targets: list[str] = []
    report_error: str | None = None
    if llm is not None and ctx.changed_files:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            changed = ", ".join(ctx.changed_files[:20])
            response = invoke_with_retry(llm, [
                SystemMessage(content=(
                    "You are a test expert. Given the files changed in a commit, "
                    "determine which specific test files or directories should be run. "
                    "Output a JSON array of test path strings relative to repo root. "
                    "If unsure, output an empty array to run the full suite."
                )),
                HumanMessage(content=f"Changed files: {changed}\n\nRepository root: {repo.name}"),
            ])
            raw = extract_text(response)
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            try:
                test_targets = json.loads(text)
            except json.JSONDecodeError:
                report_error = "LLM returned unparseable test-target output (expected a JSON array)"
            token_usage = response_tokens(response, fallback_text=raw)
        except Exception as e:
            report_error = f"LLM test-target selection failed: {e}"

    for framework, cmd in frameworks:
        full_cmd = list(cmd)
        if test_targets and framework == "pytest":
            full_cmd.extend(test_targets[:20])

        ok, stdout, stderr = _run_tests(repo, full_cmd)
        metadata[f"{framework}_exit"] = "ok" if ok else "fail"
        metadata[f"{framework}_stdout"] = stdout[:4000]
        metadata[f"{framework}_stderr"] = stderr[:4000]

        if ok:
            findings.append(Finding(
                category=FindingCategory.TESTING,
                severity=Severity.INFO,
                title=f"Tests passed ({framework})",
                description=f"All tests for '{' '.join(full_cmd)}' passed.",
                evidence=stdout[:2000] if stdout else f"Command '{' '.join(full_cmd)}' exited 0",
                confidence=1.0,
            ))
        else:
            error_snippet = stderr or stdout or "No output captured."
            findings.append(Finding(
                category=FindingCategory.TESTING,
                severity=Severity.HIGH,
                title=f"Tests failed ({framework})",
                description=f"Test command '{' '.join(full_cmd)}' returned non-zero exit.",
                evidence=error_snippet[:2000],
                recommendation="Fix the failing tests before merging.",
                confidence=0.9,
            ))

    passed = sum(1 for f in findings if f.severity == Severity.INFO)
    failed = len(findings) - passed
    framework_names = ", ".join(fw for fw, _ in frameworks)
    summary = f"Ran {len(frameworks)} test framework(s) [{framework_names}]. {passed} passed, {failed} failed."
    if failed:
        failed_details = "; ".join(
            f.title for f in findings if f.severity == Severity.HIGH
        )
        if failed_details:
            summary += f" Failures: {failed_details}."
    if report_error is not None:
        summary += f" Warning: {report_error}."

    return AgentReport(
        agent="test",
        status="error" if report_error is not None else "completed",
        summary=summary,
        findings=findings,
        metadata=metadata,
        token_usage=token_usage,
        error=report_error,
    )
