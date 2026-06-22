"""Config loading, ingest normalization, and LLM factory tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from thread_digest_bot.config import AppConfig, LLMConfig, load_config
from thread_digest_bot.ingest import thread_from_dict, thread_from_json
from thread_digest_bot.llm.factory import build_llm
from thread_digest_bot.llm.fake import FakeLLM

_THREAD: dict[str, object] = {
    "channel_id": "c1",
    "platform": "telegram",
    "messages": [
        {
            "id": "m1",
            "author": {"id": "u1", "display": "Ada"},
            "text": "hello",
            "ts_label": "t1",
            "permalink": None,
        }
    ],
}


# -- ingest ------------------------------------------------------------------


def test_thread_from_dict_preserves_order_and_fields() -> None:
    thread = thread_from_dict(_THREAD)
    assert thread.channel_id == "c1"
    assert thread.platform == "telegram"
    assert [m.id for m in thread.messages] == ["m1"]
    assert thread.messages[0].author.display == "Ada"


def test_thread_from_json_round_trip() -> None:
    import json

    thread = thread_from_json(json.dumps(_THREAD))
    assert thread.messages[0].text == "hello"


def test_thread_from_dict_rejects_unknown_field() -> None:
    bad = dict(_THREAD)
    bad["extra"] = 1
    with pytest.raises(ValidationError):
        thread_from_dict(bad)


def test_thread_from_dict_rejects_missing_field() -> None:
    bad = {"channel_id": "c1", "platform": "telegram", "messages": [{"id": "m1"}]}
    with pytest.raises(ValidationError):
        thread_from_dict(bad)


def test_thread_from_dict_rejects_bad_platform() -> None:
    bad = dict(_THREAD)
    bad["platform"] = "discord"
    with pytest.raises(ValidationError):
        thread_from_dict(bad)


# -- config ------------------------------------------------------------------


def test_load_config_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "fake"
fixture = "happy"

[storage]
decisions_dir = "docs/decisions"
orphan_policy = "raise"

[[platforms]]
name = "telegram"
token_env = "TG_TOKEN"

[[channels]]
channel_id = "team-eng"
platform = "telegram"
default_last_n = 100
rollup_period = "weekly"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.llm.provider == "fake"
    assert config.storage.orphan_policy == "raise"
    assert config.channel("team-eng") is not None
    assert config.channel("team-eng").default_last_n == 100  # type: ignore[union-attr]
    assert config.channel("missing") is None


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[llm]\nprovider = 'fake'\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(config_path)


def test_llm_api_key_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "secret")
    cfg = LLMConfig(provider="openai", api_key_env="MY_KEY")
    assert cfg.api_key() == "secret"
    assert LLMConfig(provider="fake").api_key() is None


def test_platform_token_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from thread_digest_bot.config import PlatformConfig

    monkeypatch.setenv("TG", "tok")
    assert PlatformConfig(name="telegram", token_env="TG").token() == "tok"
    assert PlatformConfig(name="telegram").token() is None


# -- factory -----------------------------------------------------------------


def test_factory_builds_fake() -> None:
    backend = build_llm(LLMConfig(provider="fake", fixture="empty"))
    assert isinstance(backend, FakeLLM)
    assert backend.fixture == "empty"


def test_factory_default_app_config() -> None:
    # The default AppConfig uses the fake provider so nothing requires extras.
    assert AppConfig().llm.provider == "fake"
    assert isinstance(build_llm(AppConfig().llm), FakeLLM)


def test_factory_builds_ollama() -> None:
    # Ollama needs no extra (stdlib HTTP), so construction succeeds offline.
    from thread_digest_bot.llm.ollama import OllamaBackend

    backend = build_llm(LLMConfig(provider="ollama", model="llama3.1", base_url="http://x:1"))
    assert isinstance(backend, OllamaBackend)
    assert backend.model == "llama3.1"
    assert backend.base_url == "http://x:1"


def test_factory_dispatches_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the backend so the dispatch branch is covered without a real SDK client.
    import thread_digest_bot.llm.openai as openai_mod

    captured: dict[str, object] = {}

    class _Stub:
        def __init__(self, model: str, *, api_key: str | None = None) -> None:
            captured["model"] = model
            captured["api_key"] = api_key

    monkeypatch.setattr(openai_mod, "OpenAIBackend", _Stub)
    monkeypatch.setenv("OAI", "k")
    backend = build_llm(LLMConfig(provider="openai", model="gpt-4o-mini", api_key_env="OAI"))
    assert isinstance(backend, _Stub)
    assert captured == {"model": "gpt-4o-mini", "api_key": "k"}


def test_factory_dispatches_to_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    import thread_digest_bot.llm.anthropic as anthropic_mod

    class _Stub:
        def __init__(self, model: str, *, api_key: str | None = None) -> None:
            self.model = model

    monkeypatch.setattr(anthropic_mod, "AnthropicBackend", _Stub)
    backend = build_llm(LLMConfig(provider="anthropic"))
    assert isinstance(backend, _Stub)
    assert backend.model == "claude-3-5-sonnet-latest"  # default model applied
