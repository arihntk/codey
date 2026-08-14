"""Retry utilities for LLM calls — exponential backoff with Retry-After support.

Wraps ``llm.invoke()`` with ``tenacity`` to retry on rate-limit (429),
server (5xx), timeout, and connection errors.  The wait between attempts
respects the provider's ``Retry-After`` response header when present and
falls back to exponential backoff with jitter otherwise.
"""

from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

__all__ = ["invoke_with_retry"]

_logger = logging.getLogger("codey.llm.retry")

_MAX_ATTEMPTS = 5
_INITIAL_WAIT = 1.0
_MAX_WAIT = 60.0

_RETRYABLE_NAME_KEYWORDS = (
    "ratelimit",
    "rateLimit",
    "ratelimiterror",
    "apierror",
    "apitimeout",
    "apiconnection",
    "apistatus",
    "internalserver",
    "serviceunavailable",
    "servererror",
    "resourceexhausted",
    "deadlineexceeded",
)


def _is_retryable(exc: Exception) -> bool:
    """Return True when *exc* represents a retryable transient LLM error."""

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None
    if status is not None:
        return status == 429 or status >= 500

    name = type(exc).__name__.lower()
    return any(kw.lower() in name for kw in _RETRYABLE_NAME_KEYWORDS)


def _extract_retry_after(exc: Exception) -> float | None:
    """Best-effort extraction of the ``Retry-After`` header value (seconds)."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After", "RETRY-AFTER"):
        raw = headers.get(key) if hasattr(headers, "get") else None
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return None


def _retry_wait(retry_state) -> float:
    """Wait function: prefer Retry-After, else exponential backoff with jitter."""
    if retry_state.outcome and retry_state.outcome.failed:
        delay = _extract_retry_after(retry_state.outcome.exception())
        if delay is not None:
            return min(delay, _MAX_WAIT)
    return wait_exponential_jitter(initial=_INITIAL_WAIT, max=_MAX_WAIT)(retry_state)


def invoke_with_retry(llm: object, messages: list, **kwargs: object) -> object:
    """Call ``llm.invoke(messages)`` with automatic retry on transient errors.

    Retries up to ``_MAX_ATTEMPTS`` times (5 total).  The wait between
    attempts respects the ``Retry-After`` response header when the provider
    includes one, otherwise it uses exponential backoff with jitter
    (1s initial, 60s max).

    Args:
        llm: A langchain chat model (ChatOpenAI, ChatAnthropic, etc.).
        messages: A list of langchain message objects (SystemMessage, HumanMessage, ...).
        **kwargs: Forwarded to ``llm.invoke()``.

    Returns:
        The response from ``llm.invoke()``.

    Raises:
        The last exception if all retry attempts are exhausted.
    """
    from tenacity import Retrying

    retrying = Retrying(
        retry=retry_if_exception(_is_retryable),
        wait=_retry_wait,
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        reraise=True,
        before_sleep=before_sleep_log(_logger, logging.WARNING),
    )
    for attempt in retrying:
        with attempt:
            return llm.invoke(messages, **kwargs)

    raise RuntimeError("unreachable")
