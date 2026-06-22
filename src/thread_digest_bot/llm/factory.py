"""Construct an :class:`~thread_digest_bot.llm.LLMBackend` from configuration.

Kept separate from the protocol module so importing the protocol never imports the
optional provider SDKs. The ``fake`` provider needs no extras and is the default.
"""

from __future__ import annotations

from thread_digest_bot.config import LLMConfig
from thread_digest_bot.llm import LLMBackend
from thread_digest_bot.llm.fake import FakeLLM


def build_llm(config: LLMConfig) -> LLMBackend:
    """Build a concrete LLM backend from an :class:`LLMConfig`.

    Args:
        config: The validated LLM configuration.

    Returns:
        A backend implementing :class:`~thread_digest_bot.llm.LLMBackend`.

    Raises:
        ValueError: For an unknown provider.
        ImportError: If a cloud provider's optional extra is not installed.
    """
    provider = config.provider
    if provider == "fake":
        return FakeLLM(config.fixture)
    if provider == "openai":
        from thread_digest_bot.llm.openai import OpenAIBackend

        return OpenAIBackend(model=config.model or "gpt-4o-mini", api_key=config.api_key())
    if provider == "anthropic":
        from thread_digest_bot.llm.anthropic import AnthropicBackend

        return AnthropicBackend(
            model=config.model or "claude-3-5-sonnet-latest",
            api_key=config.api_key(),
        )
    if provider == "ollama":
        from thread_digest_bot.llm.ollama import DEFAULT_BASE_URL, OllamaBackend

        return OllamaBackend(
            model=config.model or "llama3.1",
            base_url=config.base_url or DEFAULT_BASE_URL,
        )
    raise ValueError(f"Unknown LLM provider: {provider!r}")  # pragma: no cover - guarded by Literal
