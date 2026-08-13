"""LLM factory — build a langchain chat model from the stored config.

Supports the built-in providers (OpenAI, Anthropic, DeepSeek, Google) and
arbitrary OpenAI-compatible endpoints (Custom).  The summarizer uses the
same provider but a smaller/faster model tier.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

from codey.config.providers import ProviderPreset
from codey.config.store import Config, ConfigError, load_config, resolve_provider

__all__ = [
    "ResolvedLLM",
    "build_llm",
    "build_summarizer",
    "resolve_llms",
]


@dataclass
class ResolvedLLM:
    """Materialized chat model + metadata."""

    model: object  # BaseChatModel
    preset: ProviderPreset
    model_name: str
    api_key: str
    base_url: str | None


def _instantiate(
    preset: ProviderPreset,
    *,
    model_name: str,
    api_key: str,
    base_url: str | None,
) -> object:
    """Import the langchain chat class lazily and instantiate it."""
    module = importlib.import_module(preset.langchain_module)
    cls = getattr(module, preset.langchain_class)

    # DeepSeek uses ChatOpenAI with a fixed base_url.
    kwargs: dict[str, object] = {
        "model": model_name,
        "api_key": api_key,
    }
    if preset.key == "deepseek":
        kwargs["base_url"] = base_url or "https://api.deepseek.com"
    elif base_url:
        kwargs["base_url"] = base_url

    return cls(**kwargs)


def _ensure_api_key_in_env(preset: ProviderPreset, api_key: str) -> None:
    """Some langchain providers read the key from env; set it for this process."""
    os.environ[preset.env_key_var] = api_key


def build_llm(cfg: Config | None = None) -> ResolvedLLM:
    """Build the primary review LLM from stored config."""
    cfg = cfg or load_config()
    preset, api_key, base_url = resolve_provider(cfg)
    model_name = cfg.model or preset.default_model
    _ensure_api_key_in_env(preset, api_key)
    model = _instantiate(preset, model_name=model_name, api_key=api_key, base_url=base_url)
    return ResolvedLLM(
        model=model,
        preset=preset,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def build_summarizer(cfg: Config | None = None) -> ResolvedLLM:
    """Build the cheap/fast summarizer LLM (smaller model from same provider)."""
    cfg = cfg or load_config()
    preset, api_key, base_url = resolve_provider(cfg)
    model_name = cfg.summarizer_model or preset.summarizer_model or preset.default_model
    _ensure_api_key_in_env(preset, api_key)
    model = _instantiate(preset, model_name=model_name, api_key=api_key, base_url=base_url)
    return ResolvedLLM(
        model=model,
        preset=preset,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def resolve_llms() -> tuple[ResolvedLLM, ResolvedLLM]:
    """Convenience: build both primary and summarizer LLMs.

    Raises ``ConfigError`` if not configured.
    """
    cfg = load_config()
    if not cfg.is_complete():
        raise ConfigError("No provider configured. Run `codey set` first.")
    return build_llm(cfg), build_summarizer(cfg)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Used for context budgeting."""
    return max(1, len(text) // 4)