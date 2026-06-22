"""Deterministic, offline ``FakeLLM`` with named fixtures.

The fake returns canned structured output so the *combined* digest -> ground -> drop
pipeline can be exercised with zero network. Named fixtures intentionally produce
problematic output (hallucinated ids, fabricated quotes) so the grounding drop paths
are tested end-to-end, not just the happy path.

Fixtures
--------
``happy``
    A clean log whose citations all reference real messages with real quotes.
``invalid_ids``
    Adds an item citing a non-existent message id; grounding must drop it, leaving
    the committed log with fewer items than this raw output.
``fabricated_quote``
    A citation with a valid ``message_id`` but a ``quote`` that does not appear in
    the real message text; grounding must null/replace the quote.
``empty``
    No items at all.
``schema_mismatch``
    Forces one schema-validation failure then succeeds, to exercise the bounded
    retry. (Used by the OpenAI/Anthropic/Ollama validating-parse tests via
    :func:`run_with_retry`.)
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from thread_digest_bot.llm import (
    RETRY_PROMPT_SUFFIX,
    LLMError,
    ModelT,
    RawActionItem,
    RawCitation,
    RawDecision,
    RawDecisionLog,
    RawOpenQuestion,
)

FixtureBuilder = Callable[[], RawDecisionLog]


def _happy() -> RawDecisionLog:
    return RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship the new onboarding flow on Friday.",
                rationale="QA passed and the rollback plan is ready.",
                citations=[
                    RawCitation(message_id="m1", quote="Let's ship the onboarding flow"),
                    RawCitation(message_id="m3", quote="rollback plan is ready"),
                ],
            )
        ],
        action_items=[
            RawActionItem(
                task="Write the release notes.",
                assignee="Bob",
                citations=[RawCitation(message_id="m2", quote="I'll write the release notes")],
            )
        ],
        open_questions=[
            RawOpenQuestion(
                question="Do we need a feature flag for the rollout?",
                citations=[RawCitation(message_id="m3", quote="should we gate it behind a flag")],
            )
        ],
    )


def _invalid_ids() -> RawDecisionLog:
    log = _happy()
    extra_decision = RawDecision(
        statement="Adopt a four-day work week.",
        rationale="Someone allegedly proposed it.",
        citations=[RawCitation(message_id="does-not-exist", quote="four-day week")],
    )
    return RawDecisionLog(
        decisions=[*log.decisions, extra_decision],
        action_items=log.action_items,
        open_questions=log.open_questions,
    )


def _fabricated_quote() -> RawDecisionLog:
    return RawDecisionLog(
        decisions=[
            RawDecision(
                statement="Ship the new onboarding flow on Friday.",
                rationale=None,
                citations=[
                    RawCitation(
                        message_id="m1",
                        quote="this exact sentence was never written by anyone",
                    )
                ],
            )
        ],
        action_items=[],
        open_questions=[],
    )


def _empty() -> RawDecisionLog:
    return RawDecisionLog()


#: Fixtures whose first "response" is malformed, so ``complete_json`` exercises the
#: real bounded-retry machinery (:func:`run_with_retry`). ``schema_mismatch`` recovers
#: on the retry; ``schema_mismatch_persistent`` stays malformed and raises ``LLMError``.
_RETRY_FIXTURES = frozenset({"schema_mismatch", "schema_mismatch_persistent"})

_FIXTURES: dict[str, FixtureBuilder] = {
    "happy": _happy,
    "invalid_ids": _invalid_ids,
    "fabricated_quote": _fabricated_quote,
    "empty": _empty,
    # The retry fixtures are handled via ``run_with_retry`` in ``complete_json``; their
    # eventual (recovered) payload is the happy log.
    "schema_mismatch": _happy,
    "schema_mismatch_persistent": _happy,
}

#: A JSON string that fails ``RawDecisionLog`` validation (wrong types), used to drive
#: the bounded retry deterministically.
_MALFORMED_JSON = '{"decisions": "not-a-list"}'


class FakeLLM:
    """A deterministic in-memory LLM backend for tests and demos.

    Args:
        fixture: One of the named fixtures (default ``"happy"``).
        raw: An explicit :class:`RawDecisionLog` to return, overriding ``fixture``.
            Lets a test craft a precise raw payload.
    """

    def __init__(self, fixture: str = "happy", *, raw: RawDecisionLog | None = None) -> None:
        if raw is None and fixture not in _FIXTURES:
            valid = ", ".join(sorted(_FIXTURES))
            raise ValueError(f"Unknown FakeLLM fixture {fixture!r}; valid fixtures: {valid}.")
        self.fixture = fixture
        self._raw = raw
        #: Number of times the underlying "model" was called (incl. retries).
        self.calls = 0
        #: Prompts seen, in order — lets tests assert the retry prompt carries the error.
        self.prompts: list[str] = []

    def _payload(self) -> RawDecisionLog:
        if self._raw is not None:
            return self._raw
        return _FIXTURES[self.fixture]()

    def complete_json(self, prompt: str, schema: type[ModelT]) -> ModelT:
        """Return the canned payload coerced into ``schema``.

        Most fixtures return immediately. The retry fixtures route through
        :func:`run_with_retry` so the *single bounded retry* contract real adapters
        implement is exercised end-to-end: the first "response" is malformed JSON, the
        retry prompt carries the validation error (observable via :attr:`prompts`), and
        ``schema_mismatch_persistent`` then raises :class:`LLMError` rather than looping.
        """
        if self.fixture in _RETRY_FIXTURES and self._raw is None:
            return self._complete_with_retry(prompt, schema)

        self.calls += 1
        self.prompts.append(prompt)
        payload = self._payload()
        return self._coerce(payload, schema)

    def _complete_with_retry(self, prompt: str, schema: type[ModelT]) -> ModelT:
        recovered = self.fixture == "schema_mismatch"

        def fetch(current_prompt: str) -> str:
            self.calls += 1
            self.prompts.append(current_prompt)
            # First call is always malformed; the recovering fixture returns valid
            # JSON on the retry (any call after the first).
            if self.calls == 1 or not recovered:
                return _MALFORMED_JSON
            return self._payload().model_dump_json()

        return run_with_retry(fetch, prompt, schema)

    @staticmethod
    def _coerce(payload: RawDecisionLog, schema: type[ModelT]) -> ModelT:
        if schema is RawDecisionLog:
            # Common fast path; payload already validated.
            return payload  # type: ignore[return-value]
        # Generic path: round-trip through the requested schema.
        return schema.model_validate(payload.model_dump())


def run_with_retry(
    fetch: Callable[[str], str],
    prompt: str,
    schema: type[ModelT],
) -> ModelT:
    """Validate a provider's raw text against ``schema`` with one bounded retry.

    Adapters whose providers lack native structured output (e.g. Ollama, or any
    JSON-mode call that can still drift) use this helper: it parses, and on a
    :class:`pydantic.ValidationError` re-invokes ``fetch`` once with the error
    appended to the prompt, then raises :class:`LLMError` on a second failure.

    Args:
        fetch: Callable mapping a prompt to the provider's raw JSON text.
        prompt: The initial prompt.
        schema: The expected Pydantic model.

    Returns:
        A validated instance of ``schema``.

    Raises:
        LLMError: If validation fails twice.
    """
    current_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(2):
        raw_text = fetch(current_prompt)
        try:
            return _parse_into(raw_text, schema)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            current_prompt = prompt + RETRY_PROMPT_SUFFIX.format(error=str(exc))
            if attempt == 1:
                break
    raise LLMError(
        f"LLM response did not match schema {schema.__name__} after one retry: {last_error}"
    )


def _parse_into(raw_text: str, schema: type[ModelT]) -> ModelT:
    """Parse JSON text into ``schema`` (raises on malformed or non-conforming input)."""
    return schema.model_validate_json(raw_text)
