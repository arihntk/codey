"""Tests for codey.config — providers, store, and dynamic model listing."""

from __future__ import annotations

import pytest

from codey.config import store
from codey.config.providers import (
    PRESETS,
    all_presets,
    get_preset,
)


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """The models module caches fetch results; clear it between tests."""
    from codey.config import models

    models._CACHE.clear()
    yield


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

def test_all_presets_and_lookup():
    assert len(PRESETS) == 6
    assert [p.key for p in all_presets()] == [
        "openai", "anthropic", "deepseek", "google", "custom", "local",
    ]
    assert get_preset("openai") is not None
    assert get_preset("bogus") is None


def test_keyring_service():
    assert get_preset("openai").keyring_service == "codey::openai"


def test_local_provider_is_keyless():
    p = get_preset("local")
    assert p.requires_api_key is False
    assert p.requires_base_url is True


def test_custom_requires_base_url():
    p = get_preset("custom")
    assert p.requires_base_url is True
    assert p.requires_api_key is True


def test_preset_is_immutable():
    with pytest.raises(Exception):
        get_preset("openai").key = "nope"  # frozen dataclass


# ---------------------------------------------------------------------------
# store: Config
# ---------------------------------------------------------------------------

def test_config_defaults_and_completeness():
    c = store.Config()
    assert c.provider == ""
    assert c.is_complete() is False


def test_config_round_trip(tmp_path):
    c = store.Config(provider="openai", model="gpt-4.1", summarizer_model="gpt-4.1-mini")
    assert c.is_complete() is True
    d = c.to_dict()
    assert store.Config.from_dict(d) == c


def test_config_from_dict_ignores_unknown_keys():
    c = store.Config.from_dict({"provider": "openai", "model": "m", "extra": 1})
    assert c.provider == "openai"
    assert not hasattr(c, "extra")


def test_load_config_missing_returns_empty(tmp_path):
    # XDG_CONFIG_HOME is isolated per-test via the _isolate fixture.
    assert store.load_config() == store.Config()


def test_save_and_load_config(tmp_path):
    store.save_config(store.Config(provider="openai", model="gpt-4.1"))
    loaded = store.load_config()
    assert loaded.provider == "openai"
    assert loaded.model == "gpt-4.1"


def test_load_config_corrupt_raises():
    store.save_config(store.Config(provider="x", model="y"))
    store.CONFIG_FILE.write_text("{not json", encoding="utf-8")
    with pytest.raises(store.ConfigError):
        store.load_config()


def test_save_config_chmods_600(tmp_path):
    import os

    store.save_config(store.Config(provider="x", model="y"))
    mode = os.stat(store.CONFIG_FILE).st_mode & 0o777
    assert mode == 0o600


# ---------------------------------------------------------------------------
# store: resolve_provider / is_provider_configured
# ---------------------------------------------------------------------------

def test_resolve_provider_incomplete():
    with pytest.raises(store.ConfigError):
        store.resolve_provider(store.Config())


def test_resolve_provider_unknown():
    with pytest.raises(store.ConfigError):
        store.resolve_provider(store.Config(provider="bogus", model="m"))


def test_resolve_provider_local_keyless(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
    cfg = store.Config(provider="local", model="llama3.2", base_url="http://localhost:11434/v1")
    preset, api_key, base_url = store.resolve_provider(cfg)
    assert preset.key == "local"
    assert api_key == "local"  # placeholder
    assert base_url == "http://localhost:11434/v1"


def test_resolve_provider_custom_requires_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEY_CUSTOM_BASE_URL", raising=False)
    store.set_api_key(get_preset("custom"), "sk-test")
    cfg = store.Config(provider="custom", model="m")  # no base_url
    with pytest.raises(store.ConfigError):
        store.resolve_provider(cfg)


def test_resolve_provider_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(store, "get_api_key", lambda preset, **kw: None)
    with pytest.raises(store.ConfigError):
        store.resolve_provider(store.Config(provider="openai", model="gpt-4.1"))


def test_is_provider_configured_local_always_true(tmp_path):
    assert store.is_provider_configured(get_preset("local")) is True


# ---------------------------------------------------------------------------
# store: keyring-backed API keys (via a fake keyring)
# ---------------------------------------------------------------------------

class _FakeKeyring:
    class errors:
        class PasswordDeleteError(Exception):
            pass

    def __init__(self):
        self.data = {}

    def get_password(self, service, account):
        return self.data.get((service, account))

    def set_password(self, service, account, password):
        self.data[(service, account)] = password

    def delete_password(self, service, account):
        if (service, account) not in self.data:
            raise self.errors.PasswordDeleteError("nope")
        del self.data[(service, account)]


@pytest.fixture
def fake_keyring(monkeypatch):
    kr = _FakeKeyring()
    monkeypatch.setattr(store, "keyring", kr)
    return kr


def test_api_key_set_get_delete(fake_keyring):
    preset = get_preset("openai")
    assert store.get_api_key(preset) is None
    store.set_api_key(preset, "sk-abc")
    assert store.get_api_key(preset) == "sk-abc"
    store.delete_api_key(preset)
    assert store.get_api_key(preset) is None


def test_delete_api_key_missing_is_noop(fake_keyring):
    # must not raise when the key is absent
    store.delete_api_key(get_preset("anthropic"))


# ---------------------------------------------------------------------------
# models: dynamic model listing
# ---------------------------------------------------------------------------

def test_fetch_available_models_error_returns_empty(monkeypatch):
    from codey.config import models

    def boom(*a, **kw):
        raise OSError("offline")

    monkeypatch.setattr(models, "_get", boom)
    assert models.fetch_available_models(get_preset("openai"), api_key="k") == []


def test_fetch_available_models_openai_compatible(monkeypatch):
    from codey.config import models

    monkeypatch.setattr(
        models,
        "_get",
        lambda url, **kw: {"data": [{"id": "gpt-4.1"}, {"id": "o3-mini"}, {"id": ""}]},
    )
    out = models.fetch_available_models(get_preset("openai"), api_key="k")
    assert out == ["gpt-4.1", "o3-mini"]


def test_fetch_available_models_dedupes_and_caps(monkeypatch):
    from codey.config import models

    monkeypatch.setattr(
        models,
        "_get",
        lambda url, **kw: {"data": [{"id": f"m{i}"} for i in range(100)]},
    )
    out = models.fetch_available_models(get_preset("openai"), api_key="k", max_models=10)
    assert len(out) == 10


def test_fetch_available_models_caches(monkeypatch):
    from codey.config import models

    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        return {"data": [{"id": "m1"}]}

    monkeypatch.setattr(models, "_get", fake_get)
    p = get_preset("openai")
    assert models.fetch_available_models(p, api_key="k") == ["m1"]
    assert models.fetch_available_models(p, api_key="k") == ["m1"]
    assert calls["n"] == 1  # cached


def test_fetch_models_anthropic_google_deepseek_dispatch(monkeypatch):
    from codey.config import models

    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return {"data": [{"id": "x"}], "models": [{"name": "models/x"}]}

    monkeypatch.setattr(models, "_get", fake_get)

    models.fetch_available_models(get_preset("anthropic"), api_key="k")
    assert "anthropic" in seen["url"]

    models.fetch_available_models(get_preset("google"), api_key="k")
    assert "generativelanguage" in seen["url"]

    models.fetch_available_models(get_preset("deepseek"), api_key="k")
    assert "deepseek" in seen["url"]


def test_model_fetch_help():
    from codey.config.models import model_fetch_help

    assert "local" in model_fetch_help(get_preset("local")).lower()
    assert "anthropic" in model_fetch_help(get_preset("anthropic")).lower()
