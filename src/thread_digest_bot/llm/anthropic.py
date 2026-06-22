"""Anthropic structured-output adapter (optional extra ``[anthropic]``).

Anthropic models do not expose a JSON-schema response format directly; the idiomatic
way to force a shape is a single-tool ``tool_use`` whose ``input_schema`` is the
Pydantic schema. The tool input is the structured payload, validated (with one bounded
retry) via :func:`thread_digest_bot.llm.fake.run_with_retry`.

The ``anthropic`` package is imported lazily so this module imports cleanly without the
extra installed.
"""

from __future__ import annotations

import json
from typing import Any

from thread_digest_bot.llm import ModelT
from thread_digest_bot.llm.fake import run_with_retry

_TOOL_NAME = "emit_decision_log"


class AnthropicBackend:
    """An :class:`~thread_digest_bot.llm.LLMBackend` backed by the Anthropic API.

    Args:
        model: The model name (e.g. ``claude-3-5-sonnet-latest``).
        api_key: API key; falls back to the SDK's environment resolution if ``None``.
        max_tokens: Output token cap for the structured response.
        client: An injected client (for testing).
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - exercised via extras
                raise ImportError(
                    "The Anthropic backend requires the 'anthropic' extra: "
                    "pip install 'thread-digest-bot[anthropic]'."
                ) from exc
            self._client = Anthropic(api_key=api_key)

    def complete_json(self, prompt: str, schema: type[ModelT]) -> ModelT:
        """Return a schema-valid instance using an Anthropic single-tool call."""
        input_schema = schema.model_json_schema()
        tool = {
            "name": _TOOL_NAME,
            "description": "Return the structured decision log.",
            "input_schema": input_schema,
        }

        def fetch(current_prompt: str) -> str:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=[tool],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
                messages=[{"role": "user", "content": current_prompt}],
            )
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    return json.dumps(block.input)
            return ""

        return run_with_retry(fetch, prompt, schema)
