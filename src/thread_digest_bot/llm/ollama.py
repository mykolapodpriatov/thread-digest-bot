"""Ollama local-model adapter.

Ollama lacks native schema enforcement, so this adapter asks for JSON mode
(``format="json"``) and relies on the validating parse + one bounded retry in
:func:`thread_digest_bot.llm.fake.run_with_retry`. It talks to the Ollama HTTP API via
the stdlib only (no extra dependency), so it works out of the box against a local
server.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable

from thread_digest_bot.llm import LLMError, ModelT
from thread_digest_bot.llm.fake import run_with_retry

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaBackend:
    """An :class:`~thread_digest_bot.llm.LLMBackend` backed by a local Ollama server.

    Args:
        model: The local model name (e.g. ``llama3.1``).
        base_url: The Ollama server base URL.
        transport: Optional injected callable mapping a JSON request body to a JSON
            response body (for testing without a server).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: Callable[[dict[str, object]], dict[str, object]] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = transport or self._http_transport

    def _http_transport(self, body: dict[str, object]) -> dict[str, object]:
        url = f"{self.base_url}/api/chat"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise LLMError(f"Ollama request failed: {exc}") from exc
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):  # pragma: no cover - defensive
            raise LLMError("Ollama returned a non-object response.")
        return parsed

    def complete_json(self, prompt: str, schema: type[ModelT]) -> ModelT:
        """Return a schema-valid instance from a local Ollama model."""

        def fetch(current_prompt: str) -> str:
            response = self._transport(
                {
                    "model": self.model,
                    "format": "json",
                    "stream": False,
                    "messages": [{"role": "user", "content": current_prompt}],
                }
            )
            message = response.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            return ""

        return run_with_retry(fetch, prompt, schema)
