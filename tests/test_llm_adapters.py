"""LLM adapter tests — all offline via injected transports/clients (zero network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from thread_digest_bot.llm import LLMError, RawDecisionLog
from thread_digest_bot.llm.fake import run_with_retry
from thread_digest_bot.llm.ollama import OllamaBackend

_VALID = RawDecisionLog().model_dump_json()
_MALFORMED = '{"decisions": "not-a-list"}'


# -- run_with_retry (shared retry helper) ------------------------------------


def test_run_with_retry_succeeds_first_try() -> None:
    result = run_with_retry(lambda _p: _VALID, "prompt", RawDecisionLog)
    assert result == RawDecisionLog()


def test_run_with_retry_recovers_on_second_try() -> None:
    calls: list[str] = []

    def fetch(prompt: str) -> str:
        calls.append(prompt)
        return _MALFORMED if len(calls) == 1 else _VALID

    result = run_with_retry(fetch, "prompt", RawDecisionLog)
    assert result == RawDecisionLog()
    assert len(calls) == 2
    assert "did not match the required schema" in calls[1]


def test_run_with_retry_raises_after_two_failures() -> None:
    calls: list[str] = []

    def fetch(prompt: str) -> str:
        calls.append(prompt)
        return _MALFORMED

    with pytest.raises(LLMError) as excinfo:
        run_with_retry(fetch, "prompt", RawDecisionLog)
    assert len(calls) == 2  # bounded: no infinite loop
    assert "RawDecisionLog" in str(excinfo.value)


# -- Ollama (injected transport) ---------------------------------------------


def _ollama_response(content: str) -> dict[str, object]:
    return {"message": {"role": "assistant", "content": content}}


def test_ollama_happy_path() -> None:
    backend = OllamaBackend(transport=lambda _body: _ollama_response(_VALID))
    result = backend.complete_json("prompt", RawDecisionLog)
    assert result == RawDecisionLog()


def test_ollama_retries_then_succeeds() -> None:
    bodies: list[dict[str, object]] = []

    def transport(body: dict[str, object]) -> dict[str, object]:
        bodies.append(body)
        return _ollama_response(_MALFORMED if len(bodies) == 1 else _VALID)

    backend = OllamaBackend(transport=transport)
    result = backend.complete_json("prompt", RawDecisionLog)
    assert result == RawDecisionLog()
    assert len(bodies) == 2
    assert bodies[0]["format"] == "json"


def test_ollama_missing_content_raises() -> None:
    backend = OllamaBackend(transport=lambda _body: {"message": {}})
    with pytest.raises(LLMError):
        backend.complete_json("prompt", RawDecisionLog)


def test_ollama_http_transport_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the real stdlib HTTP transport with urlopen mocked (no network).
    import urllib.request

    payload = json.dumps(_ollama_response(_VALID)).encode("utf-8")

    class _FakeResp:
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self) -> bytes:
            return payload

    def fake_urlopen(_request: object, timeout: float = 0) -> _FakeResp:
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = OllamaBackend(base_url="http://localhost:11434/")
    result = backend.complete_json("prompt", RawDecisionLog)
    assert result == RawDecisionLog()


# -- OpenAI (injected client) ------------------------------------------------


class _FakeOpenAIClient:
    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self._i = 0
        self.chat = self
        self.completions = self

    def create(self, **_kwargs: Any) -> Any:
        content = self._contents[min(self._i, len(self._contents) - 1)]
        self._i += 1
        message = type("Msg", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


def test_openai_happy_path() -> None:
    from thread_digest_bot.llm.openai import OpenAIBackend

    backend = OpenAIBackend(client=_FakeOpenAIClient([_VALID]))
    result = backend.complete_json("prompt", RawDecisionLog)
    assert result == RawDecisionLog()


def test_openai_retries_then_raises() -> None:
    from thread_digest_bot.llm.openai import OpenAIBackend

    backend = OpenAIBackend(client=_FakeOpenAIClient([_MALFORMED, _MALFORMED]))
    with pytest.raises(LLMError):
        backend.complete_json("prompt", RawDecisionLog)


# -- Anthropic (injected client) ---------------------------------------------


class _Block:
    def __init__(self, payload: dict[str, object]) -> None:
        self.type = "tool_use"
        self.input = payload


class _FakeAnthropicMessages:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = payloads
        self._i = 0

    def create(self, **_kwargs: Any) -> Any:
        payload = self._payloads[min(self._i, len(self._payloads) - 1)]
        self._i += 1
        return type("Resp", (), {"content": [_Block(payload)]})()


class _FakeAnthropicClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.messages = _FakeAnthropicMessages(payloads)


def test_anthropic_happy_path() -> None:
    from thread_digest_bot.llm.anthropic import AnthropicBackend

    backend = AnthropicBackend(client=_FakeAnthropicClient([json.loads(_VALID)]))
    result = backend.complete_json("prompt", RawDecisionLog)
    assert result == RawDecisionLog()


def test_anthropic_retries_then_recovers() -> None:
    from thread_digest_bot.llm.anthropic import AnthropicBackend

    backend = AnthropicBackend(
        client=_FakeAnthropicClient([{"decisions": "bad"}, json.loads(_VALID)])
    )
    result = backend.complete_json("prompt", RawDecisionLog)
    assert result == RawDecisionLog()


def test_anthropic_empty_tool_use_yields_empty_then_retry() -> None:
    from thread_digest_bot.llm.anthropic import AnthropicBackend

    class _NoToolBlock:
        type = "text"

    class _Messages:
        def __init__(self) -> None:
            self.n = 0

        def create(self, **_kwargs: Any) -> Any:
            self.n += 1
            if self.n == 1:
                # No tool_use block -> empty string -> validation failure -> retry.
                return type("Resp", (), {"content": [_NoToolBlock()]})()
            return type("Resp", (), {"content": [_Block(json.loads(_VALID))]})()

    client = type("C", (), {"messages": _Messages()})()
    backend = AnthropicBackend(client=client)
    assert backend.complete_json("prompt", RawDecisionLog) == RawDecisionLog()


# -- Real-SDK construction (skipped unless the optional extra is installed) ---


def test_openai_constructs_real_client_with_key() -> None:
    pytest.importorskip("openai")
    from thread_digest_bot.llm.openai import OpenAIBackend

    backend = OpenAIBackend(model="gpt-4o-mini", api_key="sk-test-not-used")
    assert backend.model == "gpt-4o-mini"
    assert backend._client is not None


def test_anthropic_constructs_real_client_with_key() -> None:
    pytest.importorskip("anthropic")
    from thread_digest_bot.llm.anthropic import AnthropicBackend

    backend = AnthropicBackend(model="claude-3-5-sonnet-latest", api_key="sk-test-not-used")
    assert backend.model == "claude-3-5-sonnet-latest"
    assert backend._client is not None
