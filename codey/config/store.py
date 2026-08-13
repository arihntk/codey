"""Secure credential store and configuration management.

Uses the OS keyring (macOS Keychain, libsecret, Windows Credential Manager)
for API keys and a small JSON file at ``~/.config/codey/config.json`` for
non-secret metadata (active provider, model selection, base URLs).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import keyring

from codey.config.providers import ProviderPreset, get_preset

if TYPE_CHECKING:
    pass

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "save_config",
    "get_api_key",
    "set_api_key",
    "delete_api_key",
    "is_provider_configured",
    "resolve_provider",
    "active_or_prompt_provider",
]

# --- Paths ---------------------------------------------------------------

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "codey"
CONFIG_FILE = CONFIG_DIR / "config.json"


# --- Exceptions ----------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


# --- Data model ----------------------------------------------------------


@dataclass
class Config:
    """Non-secret configuration persisted to disk."""

    provider: str = ""
    model: str = ""
    summarizer_model: str = ""
    base_url: str | None = None

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
        )


# --- Persistent config I/O ----------------------------------------------


def load_config() -> Config:
    """Load config from disk, returning an empty Config if absent."""
    if not CONFIG_FILE.exists():
        return Config()
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Config file at {CONFIG_FILE} is corrupt: {e}") from e
    return Config.from_dict(raw)


def save_config(cfg: Config) -> None:
    """Persist non-secret config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _chmod_600(CONFIG_FILE)


def _chmod_600(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# --- API key storage (OS keyring) ---------------------------------------


def _keyring_service(preset: ProviderPreset, *, account: str = "default") -> tuple[str, str]:
    return preset.keyring_service, account


def get_api_key(preset: ProviderPreset, *, account: str = "default") -> str | None:
    """Retrieve an API key from the OS keyring."""
    service, acct = _keyring_service(preset, account=account)
    return keyring.get_password(service, acct)


def set_api_key(preset: ProviderPreset, key: str, *, account: str = "default") -> None:
    """Store an API key in the OS keyring."""
    service, acct = _keyring_service(preset, account=account)
    keyring.set_password(service, acct, key)


def delete_api_key(preset: ProviderPreset, *, account: str = "default") -> None:
    """Remove an API key from the OS keyring (no error if absent)."""
    service, acct = _keyring_service(preset, account=account)
    try:
        keyring.delete_password(service, acct)
    except keyring.errors.PasswordDeleteError:
        pass


# --- Convenience ---------------------------------------------------------


def is_provider_configured(preset: ProviderPreset) -> bool:
    """True when an API key (and base_url if required) are available."""
    if get_api_key(preset):
        return True
    return bool(preset.env_key_var and os.environ.get(preset.env_key_var))


def resolve_provider(cfg: Config) -> tuple[ProviderPreset, str, str | None]:
    """Resolve the active provider preset, API key, and base URL.

    Raises ``ConfigError`` if incomplete.
    """
    if not cfg.is_complete():
        raise ConfigError("No provider configured. Run `codey set` first.")
    preset = get_preset(cfg.provider)
    if preset is None:
        raise ConfigError(f"Unknown provider '{cfg.provider}'. Run `codey set`.")
    api_key = get_api_key(preset)
    if not api_key and preset.env_key_var:
        api_key = os.environ.get(preset.env_key_var)
    if not api_key:
        raise ConfigError(
            f"No API key found for {preset.label}. Run `codey set` to configure it."
        )
    base_url = cfg.base_url
    if preset.env_base_url_var and not base_url:
        env_base = os.environ.get(preset.env_base_url_var)
        if env_base:
            base_url = env_base
    if preset.requires_base_url and not base_url:
        raise ConfigError(f"Base URL required for {preset.label}. Run `codey set`.")
    return preset, api_key, base_url


def active_or_prompt_provider() -> str:
    """Return the active provider key or raise, telling the user to run `codey set`."""
    cfg = load_config()
    if not cfg.is_complete():
        raise ConfigError("No provider configured. Run `codey set` first.")
    return cfg.provider