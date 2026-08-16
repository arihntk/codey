"""Secure credential store (OS keyring) + non-secret JSON config."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import keyring

from codey.config.providers import ProviderPreset, get_preset

__all__ = [
    "Config", "ConfigError", "load_config", "save_config",
    "get_api_key", "set_api_key", "delete_api_key",
    "is_provider_configured", "resolve_provider",
]

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "codey"
CONFIG_FILE = CONFIG_DIR / "config.json"


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass
class Config:
    provider: str = ""
    model: str = ""
    summarizer_model: str = ""
    base_url: str | None = None
    append_summary_to_commit: bool | None = None

    def is_complete(self) -> bool:
        return bool(self.provider) and bool(self.model)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Config:
        return cls(
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            summarizer_model=d.get("summarizer_model", ""),
            base_url=d.get("base_url"),
            append_summary_to_commit=d.get("append_summary_to_commit"),
        )


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return Config()
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Config file at {CONFIG_FILE} is corrupt: {e}") from e
    return Config.from_dict(raw)


def save_config(cfg: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except OSError:
        pass


def get_api_key(preset: ProviderPreset, *, account: str = "default") -> str | None:
    return keyring.get_password(preset.keyring_service, account)


def set_api_key(preset: ProviderPreset, key: str, *, account: str = "default") -> None:
    keyring.set_password(preset.keyring_service, account, key)


def delete_api_key(preset: ProviderPreset, *, account: str = "default") -> None:
    try:
        keyring.delete_password(preset.keyring_service, account)
    except keyring.errors.PasswordDeleteError:
        pass


def is_provider_configured(preset: ProviderPreset) -> bool:
    if not preset.requires_api_key:
        return True
    if get_api_key(preset):
        return True
    return bool(preset.env_key_var and os.environ.get(preset.env_key_var))


def resolve_provider(cfg: Config) -> tuple[ProviderPreset, str, str | None]:
    if not cfg.is_complete():
        raise ConfigError("No provider configured. Run `codey set` first.")
    preset = get_preset(cfg.provider)
    if preset is None:
        raise ConfigError(f"Unknown provider '{cfg.provider}'. Run `codey set`.")
    api_key = ""
    if preset.requires_api_key:
        api_key = get_api_key(preset)
        if not api_key and preset.env_key_var:
            api_key = os.environ.get(preset.env_key_var)
        if not api_key:
            raise ConfigError(f"No API key found for {preset.label}. Run `codey set` to configure it.")
    else:
        # Local endpoints accept any non-empty key; a placeholder keeps the client happy.
        api_key = "local"
    base_url = cfg.base_url
    if preset.env_base_url_var and not base_url:
        base_url = os.environ.get(preset.env_base_url_var)
    if preset.requires_base_url and not base_url:
        raise ConfigError(f"Base URL required for {preset.label}. Run `codey set`.")
    return preset, api_key, base_url
