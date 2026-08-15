"""Utilities for extracting text and usage from LLM response objects.

Different providers return ``response.content`` in different shapes:
  - OpenAI / Anthropic / DeepSeek: a plain ``str``
  - Google Gemini: a ``list[dict]`` of parts, e.g. ``[{'type': 'text', 'text': '...'}]``

``extract_text()`` normalises both to a plain string so downstream code
doesn't need to care about the provider.
"""

from __future__ import annotations

__all__ = ["extract_text", "extract_usage", "response_tokens"]


def extract_text(response: object) -> str:
    """Extract a plain-text string from an LLM response object.

    Handles:
      * ``response.content`` is a ``str`` → returned as-is.
      * ``response.content`` is a ``list`` of parts (Gemini) → parts concatenated.
      * ``response.content`` is any other object → ``str()`` fallback.

    Args:
        response: A langchain ``BaseMessage`` (AIMessage) returned by ``llm.invoke()``.

    Returns:
        The text content of the response as a string.
    """
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
    """Extract the total token usage from an LLM response, if the provider
    reported it.

    LangChain chat models surface usage on the ``AIMessage`` via
    ``usage_metadata`` (``{"input_tokens": ..., "output_tokens": ...}``) when
    the provider returns it. This returns the total when available, otherwise
    ``None`` so callers can fall back to an estimate.

    Args:
        response: A langchain ``BaseMessage`` (AIMessage) returned by ``llm.invoke()``.

    Returns:
        Total token count (input + output) or ``None`` if the provider did
        not report usage.
    """
    metadata = getattr(response, "usage_metadata", None)
    if not isinstance(metadata, dict):
        return None
    input_tokens = metadata.get("input_tokens")
    output_tokens = metadata.get("output_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return input_tokens + output_tokens


def response_tokens(response: object, *, fallback_text: str = "") -> int:
    """Total tokens for a response, using real provider usage when available.

    Prefers ``usage_metadata`` (input + output). When the provider did not
    report usage, falls back to a rough estimate (~4 chars per token) of
    *fallback_text*. The estimate is intentionally only a fallback so the
    number displayed is honest: real usage when known, estimated otherwise.
    """
    real = extract_usage(response)
    if real is not None:
        return real
    return max(1, len(fallback_text) // 4)
