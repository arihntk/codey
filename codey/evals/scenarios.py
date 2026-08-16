"""Golden scenario dataset for ``codey eval``.

Each scenario is a synthetic git repository (one or more commits) with
hand-annotated ground truth: which injected issues a correct review must
find (and, via ``expect_absent``, which it must NOT report). The runner
materializes these, runs the full review pipeline, and scores the output.
"""

from __future__ import annotations

from dataclasses import dataclass

from codey.agents.schemas import FindingCategory, Severity

__all__ = ["EvalFile", "ExpectedIssue", "EvalScenario", "SCENARIOS"]


@dataclass(frozen=True)
class EvalFile:
    path: str
    content: str


@dataclass(frozen=True)
class ExpectedIssue:
    """A labelled ground-truth issue (or, with ``expect_absent``, an expected silence)."""

    agent: str  # the agent report that must surface it: security | code_quality | test | index
    category: FindingCategory
    severity: Severity
    file_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    keywords: tuple[str, ...] = ()  # any must appear in title/evidence/description
    expect_absent: bool = False  # True: a matching finding is a false positive


@dataclass(frozen=True)
class EvalScenario:
    id: str
    description: str
    tags: tuple[str, ...]
    commits: tuple[tuple[EvalFile, ...], ...]  # each commit is a set of file writes; last is the reviewed one
    expected_issues: tuple[ExpectedIssue, ...] = ()
    expected_recommendation: str | None = None  # approve | request_changes | block
    review_commit: str = "HEAD"
    run_tests: bool = False
    expect_pruned_chunks: bool = False
    expected_dependent_files: tuple[str, ...] = ()
    expect_clean: bool = False  # True: any non-INFO finding counts as a false positive


_PY_CLEAN = EvalFile(
    "app.py",
    "def add(a, b):\n"
    "    return a + b\n",
)

_OPENAI_KEY_FILE = EvalFile(
    "config.py",
    'import os\n'
    '\n'
    'API_KEY = "sk-abcd1234efgh5678ijkl9012mnop3456"\n'
    '\n'
    'def get_api_key():\n'
    '    return os.environ.get("OPENAI_API_KEY", API_KEY)\n',
)


def _big_module(index: int, funcs: int = 16, statements: int = 48) -> str:
    """A large generated module (~22KB) that never trips a secret/keyword rule.

    Sized so a single file's diff stays under the large-diff summarization
    threshold (24 000 chars) while 20 files together exceed the context budget
    (100 000 tokens), exercising budget pruning.
    """
    lines: list[str] = []
    for k in range(funcs):
        lines.append(f"def fn_{index:02d}_{k:02d}(x: int) -> int:")
        for _ in range(statements):
            lines.append("    x = x + 1")
            lines.append("    x = x * 2")
        lines.append("    return x")
        lines.append("")
    return "\n".join(lines)


def _build_large_diff_scenario() -> EvalScenario:
    files = tuple(EvalFile(f"mod_{i:02d}.py", _big_module(i)) for i in range(20))
    return EvalScenario(
        id="large-diff-budget",
        description=(
            "A diff that exceeds the context budget is pruned, and every "
            "pruned chunk is recorded on the review (never silently dropped)."
        ),
        tags=("pipeline",),
        commits=((_PY_CLEAN,), files),
        expected_recommendation="approve",
        expect_pruned_chunks=True,
    )


SCENARIOS: tuple[EvalScenario, ...] = (
    EvalScenario(
        id="secret-openai-key",
        description="A new file commits a hardcoded OpenAI API key (deterministic detector).",
        tags=("deterministic",),
        commits=(
            (_PY_CLEAN,),
            (_OPENAI_KEY_FILE,),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.CRITICAL,
                "config.py", 3, keywords=("OpenAI API key",),
            ),
        ),
        expected_recommendation="block",
    ),
    EvalScenario(
        id="secret-aws-credential",
        description="AWS access key ID and secret access key committed together.",
        tags=("deterministic",),
        commits=(
            (EvalFile(
                "aws_creds.py",
                'aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"\n'
                'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n',
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.CRITICAL,
                "aws_creds.py", 1, keywords=("AWS access key ID",),
            ),
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.CRITICAL,
                "aws_creds.py", 2, keywords=("AWS secret access key",),
            ),
        ),
        expected_recommendation="block",
    ),
    EvalScenario(
        id="secret-pem-key",
        description="A PEM private key block embedded in source (full-file scan).",
        tags=("deterministic",),
        commits=(
            (_PY_CLEAN,),
            (EvalFile(
                "server.py",
                "-----BEGIN RSA PRIVATE KEY-----\n"
                "MIIEpAIBAAKCAQEA0123456789ABCDEF0123456789ABCDEF0123456789ABCD\n"
                "-----END RSA PRIVATE KEY-----\n"
                "\n"
                "def load_key():\n"
                "    return open('server.pem').read()\n",
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.CRITICAL,
                "server.py", 1, keywords=("private key",),
            ),
        ),
        expected_recommendation="block",
    ),
    EvalScenario(
        id="secret-placeholder-guard",
        description="Placeholder and low-entropy credential values must NOT be flagged.",
        tags=("deterministic",),
        commits=(
            (_PY_CLEAN,),
            (EvalFile(
                "creds.py",
                "# configuration\n"
                "\n"
                'PASSWORD = "changeme"\n'
                'TOKEN = "aaaaaaaaaaaaaaaa"\n'
                'pwd = ""\n',
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.HIGH,
                "creds.py", 3, keywords=("password",), expect_absent=True,
            ),
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.HIGH,
                "creds.py", 4, keywords=("token",), expect_absent=True,
            ),
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.HIGH,
                "creds.py", 5, keywords=("password",), expect_absent=True,
            ),
        ),
        expected_recommendation="approve",
    ),
    EvalScenario(
        id="secret-removed",
        description="A secret removed in the diff must not be reported (only added lines are scanned).",
        tags=("deterministic",),
        commits=(
            (EvalFile(
                "app.py",
                'API_KEY = "sk-abcd1234efgh5678ijkl9012mnop3456"\n',
            ),),
            (EvalFile(
                "app.py",
                'API_KEY = os.environ["API_KEY"]\n',
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.CRITICAL,
                "app.py", 1, keywords=("API key",), expect_absent=True,
            ),
        ),
        expected_recommendation="approve",
    ),
    EvalScenario(
        id="non-head-commit",
        description="Reviewing HEAD~1 materializes a worktree and still catches the injected secret.",
        tags=("deterministic",),
        commits=(
            (_PY_CLEAN,),
            (_OPENAI_KEY_FILE,),
            (EvalFile(
                "app.py",
                "def add(a, b):\n"
                "    return a + b\n"
                "\n"
                "def mul(a, b):\n"
                "    return a * b\n",
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.CRITICAL,
                "config.py", 3, keywords=("OpenAI API key",),
            ),
        ),
        expected_recommendation="block",
        review_commit="HEAD~1",
    ),
    _build_large_diff_scenario(),
    EvalScenario(
        id="injection-sql",
        description="F-string SQL concatenation with user input (LLM security judgement).",
        tags=("llm-only",),
        commits=(
            (EvalFile(
                "db.py",
                'import sqlite3\n'
                '\n'
                '\n'
                'def query_user(name: str) -> list:\n'
                '    conn = sqlite3.connect("users.db")\n'
                "    cursor = conn.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n"
                '    return cursor.fetchall()\n',
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.HIGH,
                "db.py", 6, keywords=("sql", "injection", "inject"),
            ),
        ),
        expected_recommendation="request_changes",
    ),
    EvalScenario(
        id="pii-leak-logging",
        description="Passwords and emails written to logs (LLM confidentiality judgement).",
        tags=("llm-only",),
        commits=(
            (EvalFile(
                "users.py",
                'import logging\n'
                '\n'
                'logger = logging.getLogger(__name__)\n'
                '\n'
                '\n'
                'def signup(email: str, password: str) -> dict:\n'
                '    logger.info("signup attempt: email=%s password=%s", email, password)\n'
                '    return {"email": email}\n',
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "security", FindingCategory.SECURITY, Severity.HIGH,
                "users.py", 7, keywords=("password", "log", "pii"),
            ),
        ),
        expected_recommendation="request_changes",
    ),
    EvalScenario(
        id="quality-regression",
        description="Duplicated block and debug prints in a new function (code quality).",
        tags=("llm-only",),
        commits=(
            (EvalFile(
                "service.py",
                "\n"
                "def process_items(items):\n"
                "    for item in items:\n"
                "        result = item * 2\n"
                '        print("processing", result)\n'
                "        result = item * 2\n"
                '        print("processing", result)\n'
                "        result = item * 2\n"
                '        print("processing", result)\n'
                "    return result\n",
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "code_quality", FindingCategory.CODE_QUALITY, Severity.HIGH,
                "service.py", 4, keywords=("duplicat", "redundant", "unnecessary", "print"),
            ),
        ),
        expected_recommendation="request_changes",
    ),
    EvalScenario(
        id="quality-clean",
        description="A clean, typed, documented function must not draw code-quality findings.",
        tags=("llm-only",),
        commits=(
            (EvalFile(
                "utils.py",
                'def greet(name: str, greeting: str = "hello") -> str:\n'
                '    """Return a greeting for *name*."""\n'
                '    return f"{greeting}, {name}!"\n',
            ),),
        ),
        expected_issues=(
            ExpectedIssue(
                "code_quality", FindingCategory.CODE_QUALITY, Severity.HIGH,
                "utils.py", 1, keywords=("naming", "type", "docstring"), expect_absent=True,
            ),
        ),
        expected_recommendation="approve",
    ),
    EvalScenario(
        id="test-failure",
        description="A committed test suite that fails must surface a HIGH test finding.",
        tags=("tests-enabled",),
        commits=(
            (
                EvalFile("pyproject.toml", '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'),
                EvalFile("tests/test_math.py", "def test_always_fails():\n    assert 1 + 1 == 3\n"),
            ),
        ),
        expected_issues=(
            ExpectedIssue(
                "test", FindingCategory.TESTING, Severity.HIGH,
                "tests/test_math.py", keywords=("failed",),
            ),
        ),
        expected_recommendation="request_changes",
        run_tests=True,
    ),
    EvalScenario(
        id="test-passing",
        description="A passing test suite must not report a failure.",
        tags=("tests-enabled",),
        commits=(
            (
                EvalFile("pyproject.toml", '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'),
                EvalFile("tests/test_math.py", "def test_add():\n    assert 1 + 1 == 2\n"),
            ),
        ),
        expected_issues=(
            ExpectedIssue(
                "test", FindingCategory.TESTING, Severity.HIGH,
                "tests/test_math.py", keywords=("failed",), expect_absent=True,
            ),
        ),
        expected_recommendation="approve",
        run_tests=True,
    ),
    EvalScenario(
        id="dependent-files-affected",
        description=(
            "Changing a public function's signature must surface the affected-but-"
            "unchanged caller as a dependent file (callgraph reverse lookup)."
        ),
        tags=("llm-only", "integration"),
        commits=(
            (
                EvalFile("lib/utils.py", "def transform(value: int) -> int:\n    return value * 2\n"),
                EvalFile(
                    "app/main.py",
                    "from lib.utils import transform\n\n\ndef run() -> int:\n    return transform(1)\n",
                ),
            ),
            (
                EvalFile("lib/utils.py", "def apply(value: int, factor: int) -> int:\n    return value * factor\n"),
                EvalFile(
                    "app/main.py",
                    "from lib.utils import transform\n\n\ndef run() -> int:\n    return transform(1)\n",
                ),
            ),
        ),
        expected_issues=(
            ExpectedIssue(
                "code_quality", FindingCategory.CODE_QUALITY, Severity.MEDIUM,
                "lib/utils.py", 1, keywords=("transform", "apply", "signature", "rename", "caller"),
            ),
        ),
        expected_recommendation=None,
        expected_dependent_files=("app/main.py",),
    ),
    EvalScenario(
        id="clean-commit",
        description="A clean, typed, documented change must yield no findings and an approve.",
        tags=("integration",),
        commits=(
            (_PY_CLEAN,),
            (EvalFile(
                "feature.py",
                "def compute(x: int) -> int:\n"
                '    """Double the input."""\n'
                "    return x * 2\n",
            ),),
        ),
        expected_recommendation="approve",
        expect_clean=True,
    ),
)
