"""LLM factory — build langchain chat models from the stored config."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from codey.config.providers import ProviderPreset
from codey.config.store import Config, ConfigError, load_config, resolve_provider

__all__ = ["ResolvedLLM", "build_llm", "build_summarizer", "resolve_llms", "estimate_tokens"]


@dataclass
class ResolvedLLM:
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
    """Import the langchain chat class lazily and instantiate it.

    The API key is passed to the constructor directly, NEVER written into
    ``os.environ`` — the security/test agents spawn subprocesses that execute
    repo code, which could exfiltrate an env-injected key.
    """
    module = importlib.import_module(preset.langchain_module)
    cls = getattr(module, preset.langchain_class)

    kwargs: dict[str, object] = {"model": model_name, "api_key": api_key, "max_retries": 3}
    if preset.key == "deepseek":
        kwargs["base_url"] = base_url or "https://api.deepseek.com"
    elif base_url:
        kwargs["base_url"] = base_url

    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop("max_retries", None)
        return cls(**kwargs)


def build_llm(cfg: Config | None = None) -> ResolvedLLM:
    cfg = cfg or load_config()
    preset, api_key, base_url = resolve_provider(cfg)
    model_name = cfg.model or preset.default_model
    return ResolvedLLM(
        model=_instantiate(preset, model_name=model_name, api_key=api_key, base_url=base_url),
        preset=preset, model_name=model_name, api_key=api_key, base_url=base_url,
    )


def build_summarizer(cfg: Config | None = None) -> ResolvedLLM:
    cfg = cfg or load_config()
    preset, api_key, base_url = resolve_provider(cfg)
    model_name = cfg.summarizer_model or preset.summarizer_model or preset.default_model
    return ResolvedLLM(
        model=_instantiate(preset, model_name=model_name, api_key=api_key, base_url=base_url),
        preset=preset, model_name=model_name, api_key=api_key, base_url=base_url,
    )


def resolve_llms() -> tuple[ResolvedLLM, ResolvedLLM]:
    cfg = load_config()
    if not cfg.is_complete():
        raise ConfigError("No provider configured. Run `codey set` first.")
    return build_llm(cfg), build_summarizer(cfg)


def estimate_tokens(text: str) -> int:
    return len(text) // 4
