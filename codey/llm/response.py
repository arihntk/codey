"""Utilities for extracting text from LLM response objects.

Different providers return ``response.content`` in different shapes:
  - OpenAI / Anthropic / DeepSeek: a plain ``str``
  - Google Gemini: a ``list[dict]`` of parts, e.g. ``[{'type': 'text', 'text': '...'}]``

``extract_text()`` normalises both to a plain string so downstream code
doesn't need to care about the provider.
"""

from __future__ import annotations

__all__ = ["extract_text"]


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
