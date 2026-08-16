"""LLM-as-judge — rubric scoring of the synthesized review summary.

The judge grades the final review for groundedness, completeness,
actionability and precision on a 0-5 scale. It is advisory only: it never
changes the review's verdict, and a judge failure degrades to ``None`` rather
than failing the eval.
"""

from __future__ import annotations

import json

from codey.agents.schemas import ReviewSummary, Severity, strip_code_fences
from codey.evals.report import JudgeScore
from codey.llm.response import extract_text, response_tokens
from codey.llm.retry import invoke_with_retry

__all__ = ["judge_review"]

_JUDGE_SYSTEM = (
    "You are an impartial evaluator of AI code reviews. You are given a change "
    "description, the raw diff, and the review a code-review system produced. "
    "Score the review on four dimensions, each 0-5 (0 = worst, 5 = best):\n"
    "  groundedness  — every claim in the summary is backed by the agent reports/diff; no invented evidence\n"
    "  completeness  — the review catches the issues the change actually introduces\n"
    "  actionability — recommendations are concrete, specific, and useful\n"
    "  precision     — the review stays on-topic; no misleading, redundant or irrelevant claims\n"
    "Output ONLY a JSON object: "
    '{"groundedness": 0-5, "completeness": 0-5, "actionability": 0-5, '
    '"precision": 0-5, "overall": 0-5, "comment": "one short sentence"}. '
    "Do not pad the comment; be honest."
)


def _finding_lines(review: ReviewSummary, *, limit: int = 25) -> str:
    notable = [f for f in review.all_findings() if f.severity != Severity.INFO][:limit]
    if not notable:
        return "(no non-informational findings)"
    return "\n".join(
        f"- [{f.severity.value}] {f.title}" + (f" @ {f.file_path}:{f.line_start}" if f.file_path else "")
        for f in notable
    )


def judge_review(
    llm: object, *, description: str, diff: str, review: ReviewSummary
) -> tuple[JudgeScore | None, int, str | None]:
    """Returns ``(score, token_usage, error)``."""
    from langchain_core.messages import HumanMessage, SystemMessage

    diff_excerpt = diff[:12_000] if diff else "(no diff)"
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=(
                f"Change under review: {description}\n\n"
                f"Diff:\n```diff\n{diff_excerpt}\n```\n\n"
                f"Final recommendation: {review.recommendation}\n"
                f"Overall severity: {review.overall_severity.value}\n\n"
                f"Executive summary:\n{review.summary[:4000]}\n\n"
                f"Findings:\n{_finding_lines(review)}\n\n"
                "Score the review with the JSON rubric."
            )),
        ])
        raw = extract_text(response)
        text = strip_code_fences(raw).strip()
        usage = response_tokens(response, fallback_text=raw)
        obj = json.loads(text)
        score = JudgeScore(
            groundedness=float(obj.get("groundedness", 0)),
            completeness=float(obj.get("completeness", 0)),
            actionability=float(obj.get("actionability", 0)),
            precision=float(obj.get("precision", 0)),
            overall=float(obj.get("overall", 0)),
            comment=str(obj.get("comment", "")),
        )
        return score, usage, None
    except json.JSONDecodeError:
        return None, 0, "judge output was not valid JSON"
    except Exception as e:
        return None, 0, str(e)
