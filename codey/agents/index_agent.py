"""IndexAgent — indexes the repo and extracts architecture/design principles.

Responsibilities:
1. Run the tree-sitter indexer + jedi call graph builder.
2. Use the LLM to synthesise an architecture/design summary from the
   indexed symbol table (entry points, module structure, conventions).
3. Emit an AgentReport describing the codebase.
"""

from __future__ import annotations

from codey.agents.context import ReviewContext
from codey.agents.schemas import AgentReport, Finding, FindingCategory, Severity
from codey.cache.ast_cache import CacheDB
from codey.index.callgraph import build_call_graph
from codey.index.indexer import index_repository
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry

__all__ = ["run_index_agent"]

_INDEX_SYSTEM = (
    "You are a code architecture analyst. Given a repository's file list and "
    "symbol table, produce a concise architecture summary:\n"
    "1. Entry points and module structure\n"
    "2. Design patterns and conventions observed\n"
    "3. Code quality benchmarks (naming, typing, docstring coverage)\n\n"
    "CRITICAL: Cite specific file paths and symbol names from the provided "
    "symbol table as evidence for each observation. Do not fabricate file "
    "paths or symbols that are not in the input.\n"
    "Output 5-10 bullet points. Be specific, cite file paths."
)


def run_index_agent(
    ctx: ReviewContext,
    db: CacheDB,
    llm: object | None = None,
) -> tuple[AgentReport, str]:
    """Index the repo and produce an architecture summary.

    Returns (AgentReport, index_summary_string).
    """
    repo = ctx.repo_path
    index_result = index_repository(repo, db)
    build_call_graph(repo, index_result.git_hash, db)

    # Build symbol overview for the LLM.
    overview = _build_symbol_overview(db, str(repo), index_result.git_hash)

    index_summary = overview
    findings: list[Finding] = []
    token_usage = 0

    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = invoke_with_retry(llm, [
                SystemMessage(content=_INDEX_SYSTEM),
                HumanMessage(content=f"Repository: {repo.name}\n\n{overview}"),
            ])
            index_summary = extract_text(response)
            token_usage = response_tokens(response, fallback_text=index_summary)
        except Exception as e:
            findings.append(Finding(
                category=FindingCategory.ARCHITECTURE,
                severity=Severity.LOW,
                title="Index LLM summarisation failed",
                description=str(e),
                confidence=0.5,
            ))

    findings.insert(0, Finding(
        category=FindingCategory.ARCHITECTURE,
        severity=Severity.INFO,
        title="Repository indexed",
        description=(
            f"Indexed {index_result.total_files} files "
            f"({index_result.parsed_files} parsed, {index_result.reused_files} cached). "
            f"Extracted {index_result.symbols_extracted} symbols across "
            f"{len(set(s.rel_path for s in db.all_symbols(str(repo), index_result.git_hash)))} files."
        ),
        evidence=overview[:1000] if overview else "",
        file_path=None,
        confidence=1.0,
    ))

    report = AgentReport(
        agent="index",
        status="completed",
        summary=index_summary[:500] if index_summary else (
            f"Indexed {index_result.total_files} files, "
            f"{index_result.symbols_extracted} symbols."
        ),
        findings=findings,
        metadata={
            "git_hash": index_result.git_hash,
            "total_files": str(index_result.total_files),
            "parsed_files": str(index_result.parsed_files),
            "reused_files": str(index_result.reused_files),
            "symbols_extracted": str(index_result.symbols_extracted),
        },
        token_usage=token_usage,
    )
    return report, index_summary


def _build_symbol_overview(db: CacheDB, repo: str, git_hash: str, *, max_files: int = 60) -> str:
    """Build a compact text overview of the indexed symbols for LLM consumption."""
    lines: list[str] = []
    symbols = db.all_symbols(repo, git_hash)
    by_file: dict[str, list[str]] = {}
    for s in symbols:
        by_file.setdefault(s.rel_path, []).append(
            f"  L{s.line_start}-{s.line_end} {s.kind} {s.qualified_name}"
        )
    for i, (path, syms) in enumerate(sorted(by_file.items())):
        if i >= max_files:
            lines.append(f"... and {len(by_file) - max_files} more files")
            break
        lines.append(f"{path}:")
        lines.extend(syms[:20])
        if len(syms) > 20:
            lines.append(f"  ... +{len(syms) - 20} more symbols")
    return "\n".join(lines) if lines else "No symbols found."
