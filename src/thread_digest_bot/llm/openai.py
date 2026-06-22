"""OpenAI structured-output adapter (optional extra ``[openai]``).

Uses the Chat Completions JSON-schema response format derived from the Pydantic
schema's ``model_json_schema()``, with one bounded retry via
:func:`thread_digest_bot.llm.fake.run_with_retry` for robustness against drift.

The ``openai`` package is imported lazily so importing this module never fails when
the extra is not installed; construction raises a clear error instead.
"""

from __future__ import annotations

from typing import Any

from thread_digest_bot.llm import ModelT
from thread_digest_bot.llm.fake import run_with_retry


class OpenAIBackend:
    """An :class:`~thread_digest_bot.llm.LLMBackend` backed by the OpenAI API.

    Args:
        model: The model name (e.g. ``gpt-4o-mini``).
        api_key: API key; falls back to the SDK's environment resolution if ``None``.
        client: An injected client (for testing); when given, ``model``/``api_key``
            are still used for requests but no real client is constructed.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised via extras
                raise ImportError(
                    "The OpenAI backend requires the 'openai' extra: "
                    "pip install 'thread-digest-bot[openai]'."
                ) from exc
            self._client = OpenAI(api_key=api_key)

    def complete_json(self, prompt: str, schema: type[ModelT]) -> ModelT:
        """Return a schema-valid instance using OpenAI JSON-schema response format."""
        json_schema = schema.model_json_schema()

        def fetch(current_prompt: str) -> str:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": current_prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema,
                        "strict": False,
                    },
                },
            )
            content = response.choices[0].message.content
            return content or ""

        return run_with_retry(fetch, prompt, schema)
