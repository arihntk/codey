"""Retry LLM calls with exponential backoff + Retry-After support."""

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
    "ratelimit", "ratelimiterror", "apierror", "apitimeout", "apiconnection",
    "apistatus", "internalserver", "serviceunavailable", "servererror",
    "resourceexhausted", "deadlineexceeded",
)


def _is_retryable(exc: Exception) -> bool:
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
    return any(kw in name for kw in _RETRYABLE_NAME_KEYWORDS)


def _extract_retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
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
    if retry_state.outcome and retry_state.outcome.failed:
        delay = _extract_retry_after(retry_state.outcome.exception())
        if delay is not None:
            return min(delay, _MAX_WAIT)
    return wait_exponential_jitter(initial=_INITIAL_WAIT, max=_MAX_WAIT)(retry_state)


def invoke_with_retry(llm: object, messages: list, **kwargs: object) -> object:
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
