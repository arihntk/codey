"""Normalise provider response objects to plain text + token usage."""

from __future__ import annotations

__all__ = ["extract_text", "extract_usage", "response_tokens"]


def extract_text(response: object) -> str:
    """Extract plain text from an LLM response (str, Gemini list-of-parts, etc.)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif hasattr(item, "content"):
                parts.append(str(item.content))
        return "\n".join(parts) if parts else str(content)
    return str(content)


def extract_usage(response: object) -> int | None:
    metadata = getattr(response, "usage_metadata", None)
    if not isinstance(metadata, dict):
        return None
    input_tokens = metadata.get("input_tokens")
    output_tokens = metadata.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return input_tokens + output_tokens


def response_tokens(response: object, *, fallback_text: str = "") -> int:
    real = extract_usage(response)
    if real is not None:
        return real
    return max(1, len(fallback_text) // 4)
