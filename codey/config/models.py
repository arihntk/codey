"""Dynamic model listing for the CLI.

Fetches the latest available models from each provider's public models API so
the ``codey set`` / ``codey model`` prompts don't show a stale hardcoded list.
Every fetch is best-effort with a short timeout: on any failure (offline,
auth error, unsupported endpoint) the caller falls back to the bundled
``recommended_models`` preset list.

Endpoints used (all public, OpenAI-compatible where possible):
  - openai / custom / local: ``GET {base}/models`` (Bearer key; local servers
    like Ollama/LM Studio/vLLM accept it without a real key)
  - anthropic: ``GET https://api.anthropic.com/v1/models``
  - google: ``GET https://generativelanguage.googleapis.com/v1beta/models``
  - deepseek: ``GET https://api.deepseek.com/models``
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from codey.config.providers import ProviderPreset

__all__ = ["fetch_available_models", "model_fetch_help"]

_TIMEOUT = 6  # seconds; keep the interactive prompt snappy
_MAX_MODELS = 40  # cap the list presented to the user

# Response-cache: fetch once per (provider, base_url) within a few minutes so
# repeated prompts (primary + summarizer) don't re-hit the network.
_CACHE: dict[tuple[str, str | None], tuple[float, list[str]]] = {}
_CACHE_TTL = 300


def _get(url: str, *, headers: dict[str, str] | None = None, timeout: int = _TIMEOUT) -> Any:
    """GET *url* and parse JSON, raising on any failure."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _openai_compatible(url: str, *, api_key: str | None, timeout: int = _TIMEOUT) -> list[str]:
    """Fetch from an OpenAI-compatible ``/models`` endpoint (cloud or local)."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = _get(url, headers=headers, timeout=timeout)
    out: list[str] = []
    for item in data.get("data", []):
        mid = str(item.get("id", "")).strip()
        if mid:
            out.append(mid)
    return out


def _anthropic(url: str, *, api_key: str, timeout: int = _TIMEOUT) -> list[str]:
    data = _get(url, headers={
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }, timeout=timeout)
    return [str(item.get("id", "")).strip() for item in data.get("data", []) if item.get("id")]


def _google(url: str, *, api_key: str, timeout: int = _TIMEOUT) -> list[str]:
    data = _get(f"{url}?key={api_key}", timeout=timeout)
    out: list[str] = []
    for item in data.get("models", []):
        name = str(item.get("name", ""))
        # names look like "models/gemini-2.5-pro" — strip the prefix.
        if name.startswith("models/"):
            name = name[len("models/"):]
        if name:
            out.append(name)
    return out


def _fetch_raw(preset: ProviderPreset, *, api_key: str, base_url: str | None) -> list[str]:
    """Query the provider's models API. Raises on any failure."""
    key = preset.key
    if key == "openai":
        return _openai_compatible("https://api.openai.com/v1/models", api_key=api_key)
    if key == "anthropic":
        return _anthropic("https://api.anthropic.com/v1/models", api_key=api_key)
    if key == "google":
        return _google("https://generativelanguage.googleapis.com/v1beta/models", api_key=api_key)
    if key == "deepseek":
        return _openai_compatible("https://api.deepseek.com/models", api_key=api_key)
    if key == "custom":
        base = (base_url or "").rstrip("/")
        if not base:
            raise ValueError("no base URL configured")
        return _openai_compatible(f"{base}/models", api_key=api_key)
    if key == "local":
        base = (base_url or "").rstrip("/")
        if not base:
            raise ValueError("no base URL configured")
        return _openai_compatible(f"{base}/models", api_key=None)
    raise ValueError(f"no models endpoint for provider '{key}'")


def fetch_available_models(
    preset: ProviderPreset,
    *,
    api_key: str = "",
    base_url: str | None = None,
    max_models: int = _MAX_MODELS,
) -> list[str]:
    """Return the latest available model IDs for *preset*, or ``[]`` on failure.

    Best-effort: a network error, auth failure, or unsupported endpoint
    returns an empty list so the caller can fall back to the bundled list.
    Results are cached briefly per (provider, base_url).
    """
    cache_key = (preset.key, base_url)
    now = time.monotonic()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    try:
        models = _fetch_raw(preset, api_key=api_key, base_url=base_url)
        # Dedupe, drop empties, keep the primary ordering.
        seen: set[str] = set()
        ordered: list[str] = []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)
        models = ordered[:max_models]
    except Exception:
        models = []
    _CACHE[cache_key] = (now, models)
    return models


def model_fetch_help(preset: ProviderPreset) -> str:
    """Human-readable note about where the dynamic list comes from."""
    if preset.key == "local":
        return "listing models from your local server's /models endpoint"
    if preset.key == "custom":
        return "listing models from your custom endpoint's /models endpoint"
    if preset.key == "anthropic":
        return "listing models from the Anthropic API"
    if preset.key == "google":
        return "listing models from the Google Generative Language API"
    return f"listing models from the {preset.label} API"
