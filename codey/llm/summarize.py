"""Cheap/fast diff summarizer — used for large diffs before agent review.

Summarizes a raw diff into a compact natural-language description suitable for
inclusion in agent prompts, respecting the model's context budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from codey.llm.factory import ResolvedLLM, estimate_tokens

__all__ = ["DiffSummary", "summarize_diff", "summarize_diffs"]

_SUMMARY_SYSTEM = (
    "You are a concise code-review assistant. Summarize the following git diff "
    "in 5-10 bullet points: what changed, what functions/classes are affected, "
    "and any risky patterns. Be terse. Output only bullets, no preamble."
)


@dataclass
class DiffSummary:
    path: str
    summary: str
    token_estimate: int


def summarize_diff(
    summarizer: ResolvedLLM,
    path: str,
    diff_text: str,
    *,
    max_chars: int = 24_000,
) -> DiffSummary:
    """Summarize a single file diff using the cheap/fast model.

    If the diff is small enough it is left as-is (no LLM call).
    """
    if len(diff_text) <= max_chars:
        return DiffSummary(
            path=path,
            summary=diff_text,
            token_estimate=estimate_tokens(diff_text),
        )
    truncated = diff_text[: max_chars * 2]  # leave room for long truncation
    from langchain_core.messages import HumanMessage, SystemMessage

    response = summarizer.model.invoke([
        SystemMessage(content=_SUMMARY_SYSTEM),
        HumanMessage(content=f"File: {path}\n\n```diff\n{truncated}\n```"),
    ])
    summary = response.content if isinstance(response.content, str) else str(response.content)
    return DiffSummary(path=path, summary=summary, token_estimate=estimate_tokens(summary))


def summarize_diffs(
    summarizer: ResolvedLLM,
    diffs: dict[str, str],
    *,
    max_chars: int = 24_000,
) -> list[DiffSummary]:
    """Summarize a mapping of {path -> diff_text} into DiffSummary objects."""
    return [summarize_diff(summarizer, path, text, max_chars=max_chars) for path, text in diffs.items()]
