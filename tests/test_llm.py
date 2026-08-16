"""Tests for codey.llm — response extraction, retry, factory, summarize."""

from __future__ import annotations

import pytest

from codey.llm import response as resp
from codey.llm import retry as retry_mod
from codey.llm.factory import estimate_tokens
from tests.conftest import FakeLLM, FakeResponse, RateLimitError, ServerError

# ---------------------------------------------------------------------------
# response
# ---------------------------------------------------------------------------

def test_extract_text_str():
    assert resp.extract_text(FakeResponse("hello")) == "hello"


def test_extract_text_gemini_list_of_dicts():
    content = [
        {"type": "text", "text": "part1"},
        {"type": "text", "text": "part2"},
    ]
    assert resp.extract_text(FakeResponse(content)) == "part1\npart2"


def test_extract_text_list_of_strings():
    assert resp.extract_text(FakeResponse(["a", "b"])) == "a\nb"


def test_extract_text_list_of_objects_with_text_attr():
    class Part:
        def __init__(self, text):
            self.text = text

    assert resp.extract_text(FakeResponse([Part("x")])) == "x"


def test_extract_text_falls_back_to_str():
    assert resp.extract_text(FakeResponse(123)) == "123"


def test_extract_text_whole_object_no_content():
    # response without a .content attr -> str() of the object
    class R:
        pass

    assert "R object" in resp.extract_text(R())


def test_extract_usage_valid():
    assert resp.extract_usage(FakeResponse("x", {"input_tokens": 7, "output_tokens": 3})) == 10


def test_extract_usage_missing_or_bad():
    assert resp.extract_usage(FakeResponse("x")) is None
    assert resp.extract_usage(FakeResponse("x", {"input_tokens": "a", "output_tokens": 1})) is None
    assert resp.extract_usage(FakeResponse("x", {"input_tokens": 1})) is None


def test_response_tokens_prefers_real_usage():
    r = FakeResponse("x", {"input_tokens": 7, "output_tokens": 3})
    assert resp.response_tokens(r, fallback_text="x" * 100) == 10


def test_response_tokens_falls_back_to_estimate():
    assert resp.response_tokens(FakeResponse("x"), fallback_text="y" * 100) == 25


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def test_is_retryable_timeouts():
    assert retry_mod._is_retryable(TimeoutError("t")) is True
    assert retry_mod._is_retryable(ConnectionError("c")) is True


def test_is_retryable_status_codes():
    assert retry_mod._is_retryable(RateLimitError()) is True
    assert retry_mod._is_retryable(ServerError()) is True
    e = Exception("client error")
    e.status_code = 400
    assert retry_mod._is_retryable(e) is False


def test_is_retryable_by_name():
    class APIError(Exception):
        pass

    assert retry_mod._is_retryable(APIError("x")) is True
    assert retry_mod._is_retryable(ValueError("nope")) is False


def test_extract_retry_after():
    assert retry_mod._extract_retry_after(RateLimitError("5")) == 5.0
    assert retry_mod._extract_retry_after(ValueError("no resp")) is None


def test_invoke_with_retry_succeeds_first_try():
    llm = FakeLLM(content="ok")
    out = retry_mod.invoke_with_retry(llm, ["m"])
    assert out.content == "ok"
    assert len(llm.calls) == 1


def test_invoke_with_retry_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(retry_mod, "_retry_wait", lambda st: 0)  # no sleep
    llm = FakeLLM(content="eventual", failures=2)
    out = retry_mod.invoke_with_retry(llm, ["m"])
    assert out.content == "eventual"
    assert len(llm.calls) == 1  # succeeded after 2 failures


def test_invoke_with_retry_raises_after_exhaustion(monkeypatch):
    monkeypatch.setattr(retry_mod, "_retry_wait", lambda st: 0)
    monkeypatch.setattr(retry_mod, "_MAX_ATTEMPTS", 3)
    llm = FakeLLM(content="x", failures=10)  # always fails
    with pytest.raises(RateLimitError):
        retry_mod.invoke_with_retry(llm, ["m"])


def test_invoke_with_retry_does_not_retry_non_retryable(monkeypatch):
    monkeypatch.setattr(retry_mod, "_retry_wait", lambda st: 0)
    llm = FakeLLM(content="x", failures=1, error_factory=ValueError)
    with pytest.raises(ValueError):
        retry_mod.invoke_with_retry(llm, ["m"])


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 2
    assert estimate_tokens("x" * 100) == 25


def test_instantiate_local_provider():
    from codey.config.providers import get_preset
    from codey.llm.factory import _instantiate

    model = _instantiate(
        get_preset("local"),
        model_name="llama3.2",
        api_key="local",
        base_url="http://localhost:11434/v1",
    )
    assert model.model_name == "llama3.2"
    assert model.openai_api_base == "http://localhost:11434/v1"


def test_instantiate_deepseek_defaults_base_url():
    from codey.config.providers import get_preset
    from codey.llm.factory import _instantiate

    model = _instantiate(get_preset("deepseek"), model_name="deepseek-chat", api_key="k", base_url=None)
    assert model.openai_api_base == "https://api.deepseek.com"


def test_build_llm_and_summarizer_local(tmp_path, monkeypatch):
    from codey.config import store
    from codey.llm.factory import build_llm, build_summarizer, resolve_llms

    store.save_config(store.Config(
        provider="local",
        model="llama3.2",
        summarizer_model="llama3.2",
        base_url="http://localhost:11434/v1",
    ))
    primary = build_llm()
    assert primary.model_name == "llama3.2"
    summarizer = build_summarizer()
    assert summarizer.model_name == "llama3.2"
    p, s = resolve_llms()
    assert p.model_name == "llama3.2"


def test_build_llm_raises_without_config(tmp_path):
    from codey.config.store import ConfigError
    from codey.llm.factory import build_llm

    with pytest.raises(ConfigError):
        build_llm()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_diff_small_is_passthrough():
    from codey.llm.factory import ResolvedLLM
    from codey.llm.summarize import summarize_diff

    llm = ResolvedLLM(FakeLLM(), object(), "m", "k", None)
    out = summarize_diff(llm, "a.py", "small diff", max_chars=1000)
    assert out.summary == "small diff"
    assert out.token_estimate == estimate_tokens("small diff")


def test_summarize_diff_large_calls_llm():
    from codey.llm.factory import ResolvedLLM
    from codey.llm.summarize import summarize_diff

    fake = FakeLLM(content="summary text")
    llm = ResolvedLLM(fake, object(), "m", "k", None)
    big = "x" * 2000
    out = summarize_diff(llm, "a.py", big, max_chars=1000)
    assert out.summary == "summary text"
    assert len(fake.calls) == 1


def test_summarize_diff_large_llm_error_falls_back_to_truncation():
    from codey.llm.factory import ResolvedLLM
    from codey.llm.summarize import summarize_diff

    fake = FakeLLM(content="summary", failures=1, error_factory=ValueError)
    llm = ResolvedLLM(fake, object(), "m", "k", None)
    big = "x" * 2000
    out = summarize_diff(llm, "a.py", big, max_chars=1000)
    # falls back to the truncated raw diff (first max_chars*2 chars)
    assert out.summary.startswith("x" * 1000)


def test_summarize_diffs_maps_all_paths():
    from codey.llm.factory import ResolvedLLM
    from codey.llm.summarize import summarize_diffs

    llm = ResolvedLLM(FakeLLM(), object(), "m", "k", None)
    out = summarize_diffs(llm, {"a.py": "one", "b.py": "two"})
    assert {s.path for s in out} == {"a.py", "b.py"}
