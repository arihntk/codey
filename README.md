# codey

[![PyPI](https://img.shields.io/pypi/v/codey-review.svg)](https://pypi.org/project/codey-review/)
[![Python](https://img.shields.io/pypi/pyversions/codey-review.svg)](https://pypi.org/project/codey-review/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-grade multi-agent AI code review system.**

Codey orchestrates specialized agents — security, code quality, testing, and indexing — over your local git diff, with tree-sitter AST caching and jedi-based reverse-dependency lookup. Built on [LangGraph](https://github.com/langchain-ai/langgraph).

```
codey set      # configure provider & API key (OpenAI / Anthropic / DeepSeek / Google / custom)
codey model    # view or switch the active model
codey config   # show current configuration
codey review   # review the latest commit in the local repo
```

## Features

- **Multi-agent review**: LangGraph supervisor fans out to 4 specialist agents in parallel, each emitting a structured standalone report
- **AST-aware caching**: tree-sitter parses files once; subsequent runs diff against the last git-hash and re-parse only changed files (sqlite at `~/.cache/codey/codey.db`)
- **Reverse-dependency lookup** (Python v1): jedi call graph finds files that import or call into changed code, sending affected-but-unmodified chunks to the LLM for verification
- **AST-aware diff chunking**: large diffs are broken into function/class-level chunks mapped via the cached symbol table
- **Context window budgeting**: cheap/fast summarizer model condenses very large diffs; chunks are pruned to fit the model's token budget
- **Security analysis**: runs `bandit` (Python), `semgrep` (multi-language), `gitleaks` (secrets) — skips files that obviously don't affect security (css, images, lock files); LLM synthesizes raw tool output into structured findings
- **Code quality benchmarking**: compares the diff against the indexed codebase's architecture/design conventions
- **Test execution**: auto-detects the test framework (pytest, npm, go, cargo, rake); skips cleanly if no test suite is identifiable
- **Live progress**: rich terminal updates as each agent starts and finishes
- **Rich rendering**: final summary as markdown with severity-coloured findings table; interactive standalone report viewer (`--report`)

## Install

```bash
pip install codey-review
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install codey-review
```

> **Note:** The PyPI package name is `codey-review`, but the CLI command and Python import are both `codey`.

## Quickstart

```bash
codey set       # pick a provider, enter your API key, choose models
codey review   # review the latest commit (HEAD~1..HEAD)
```

## How it works

```
┌──────────────────────────────────────────────────────────┐
│                     Codey Review Pipeline                │
│                                                          │
│  git diff ──► AST chunker ──► diff chunks (per-symbol)   │
│                  │                                       │
│     reverse-dep lookup ──► dependent file sources         │
│                  │                                       │
│         ┌────────▼─────────┐                             │
│         │   LangGraph supervisor                         │
│         │  ┌─────────────┐ │                             │
│         │  │ IndexAgent  │─┼──► architecture summary     │
│         │  └──────┬──────┘ │                             │
│         │         │        │  (parallel fan-out)         │
│         │  ┌──────▼──────┐ │                             │
│         │  │SecurityAgnt │ │ ─► bandit/semgrep/gitleaks  │
│         │  │CodeQualitAgt│ │ ─► convention benchmarks    │
│         │  │ TestAgent   │ │ ─► pytest/npm/go/cargo      │
│         │  └──────┬──────┘ │                             │
│         │         │        │                             │
│         │  ┌──────▼──────┐  │                             │
│         │  │ CodeyAgent  │─┼──► final ReviewSummary      │
│         │  └─────────────┘  │  (markdown + recommendation)│
│         └──────────────────┘                              │
│                  │                                       │
│          rich terminal render                             │
└──────────────────────────────────────────────────────────┘
```

### Agents

| Agent | Role | LLM | Tools | Skips |
|-------|------|-----|-------|-------|
| **index** | Repo indexer + architecture summary | ✓ | tree-sitter, jedi | — |
| **security** | Vulnerability & secret detection | ✓ | bandit, semgrep, gitleaks | css, md, images, fonts, lock files |
| **code_quality** | Convention & pattern benchmarks | ✓ | — | when no diff provided |
| **test** | Test suite identification & execution | ✓ | pytest, npm, go, cargo, rake | when no framework detectable |
| **codey** (supervisor) | Executive synthesis + retrieval | ✓ | grep, cat, ls, git (via react-agent) | — |

### Structured findings

Every agent emits a pydantic `AgentReport` containing `Finding` objects with `severity`, `category`, `title`, `file_path`, `line_start`, `evidence`, `recommendation`, and `confidence` (0-1). The orchestrator aggregates these into a `ReviewSummary` with an overall severity and a recommendation of `approve`, `request_changes`, or `block`.

### Cache

The AST/symbol cache lives at `~/.cache/codey/codey.db` (sqlite, WAL mode), keyed by absolute repo path. On each review:

1. Compute current HEAD hash
2. If already indexed → no-op (cache hit)
3. Otherwise reuse file entries from the last indexed hash whose content hash hasn't changed; re-parse only changed files
4. Fresh symbols + call edges + import edges stored for the new hash

## Development

```bash
git clone https://github.com/arihant/codey
cd codey
uv sync
uv run pytest tests/ -v
uv run ruff check codey/
```

## Package

- **PyPI:** https://pypi.org/project/codey-review/
- **Install:** `pip install codey-review`
- **Python import:** `import codey`
- **CLI command:** `codey --help`

## License

MIT