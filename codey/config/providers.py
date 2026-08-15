"""Provider preset table — built-in models per supported provider."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderPreset:
    """Static metadata for a supported LLM provider."""

    key: str
    label: str
    langchain_module: str
    langchain_class: str
    env_key_var: str
    env_base_url_var: str | None
    default_model: str
    summarizer_model: str
    recommended_models: list[str] = field(default_factory=list)
    requires_base_url: bool = False
    requires_api_key: bool = True

    @property
    def keyring_service(self) -> str:
        return f"codey::{self.key}"


OPENAI = ProviderPreset(
    key="openai",
    label="OpenAI",
    langchain_module="langchain_openai",
    langchain_class="ChatOpenAI",
    env_key_var="OPENAI_API_KEY",
    env_base_url_var="OPENAI_BASE_URL",
    default_model="gpt-4.1",
    summarizer_model="gpt-4.1-mini",
    recommended_models=["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "o3-mini"],
)

ANTHROPIC = ProviderPreset(
    key="anthropic",
    label="Anthropic",
    langchain_module="langchain_anthropic",
    langchain_class="ChatAnthropic",
    env_key_var="ANTHROPIC_API_KEY",
    env_base_url_var=None,
    default_model="claude-sonnet-4-5-20250929",
    summarizer_model="claude-haiku-4-5-20251001",
    recommended_models=[
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-1-20250805",
    ],
)

DEEPSEEK = ProviderPreset(
    key="deepseek",
    label="DeepSeek",
    langchain_module="langchain_openai",
    langchain_class="ChatOpenAI",
    env_key_var="DEEPSEEK_API_KEY",
    env_base_url_var="DEEPSEEK_BASE_URL",
    default_model="deepseek-chat",
    summarizer_model="deepseek-chat",
    recommended_models=["deepseek-chat", "deepseek-reasoner"],
)

GOOGLE = ProviderPreset(
    key="google",
    label="Google",
    langchain_module="langchain_google_genai",
    langchain_class="ChatGoogleGenerativeAI",
    env_key_var="GOOGLE_API_KEY",
    env_base_url_var=None,
    default_model="gemini-2.5-pro",
    summarizer_model="gemini-2.0-flash",
    recommended_models=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
)

CUSTOM = ProviderPreset(
    key="custom",
    label="Custom (OpenAI-compatible)",
    langchain_module="langchain_openai",
    langchain_class="ChatOpenAI",
    env_key_var="CODEY_CUSTOM_API_KEY",
    env_base_url_var="CODEY_CUSTOM_BASE_URL",
    default_model="",
    summarizer_model="",
    recommended_models=[],
    requires_base_url=True,
)

LOCAL = ProviderPreset(
    key="local",
    label="Local (Ollama / LM Studio / llama.cpp)",
    langchain_module="langchain_openai",
    langchain_class="ChatOpenAI",
    env_key_var="",
    env_base_url_var="LOCAL_MODEL_BASE_URL",
    default_model="llama3.2",
    summarizer_model="llama3.2",
    recommended_models=["llama3.2", "llama3.1", "qwen2.5", "mistral", "phi3", "gemma2"],
    requires_base_url=True,
    requires_api_key=False,
)

PRESETS: list[ProviderPreset] = [OPENAI, ANTHROPIC, DEEPSEEK, GOOGLE, CUSTOM, LOCAL]

_PRESET_MAP: dict[str, ProviderPreset] = {p.key: p for p in PRESETS}


def get_preset(key: str) -> ProviderPreset | None:
    return _PRESET_MAP.get(key)


def all_presets() -> list[ProviderPreset]:
    return list(PRESETS)
