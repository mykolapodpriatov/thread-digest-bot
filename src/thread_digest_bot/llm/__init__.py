"""LLM backend abstraction.

The whole digest pipeline depends only on the :class:`LLMBackend` protocol:
``complete_json(prompt, schema)`` returns an instance of the given Pydantic model.
Concrete adapters (OpenAI / Anthropic / Ollama / the deterministic Fake) live in
sibling modules and translate the schema to their provider-native mechanism.

The *raw* LLM output schema (:class:`RawDecisionLog`) deliberately models citations
with only ``message_id`` and an optional candidate ``quote`` — author and permalink
are never requested from the model; they are derived during grounding.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM backend cannot return a schema-valid response."""


class RawCitation(BaseModel):
    """A citation as proposed by the LLM.

    Only ``message_id`` and an optional candidate ``quote`` are modelled. Author and
    permalink are intentionally absent so the model cannot supply (and thus cannot
    fabricate) attribution — those are filled from the real message during grounding.
    """

    message_id: str
    quote: str | None = None


class RawDecision(BaseModel):
    """A decision as proposed by the LLM."""

    statement: str
    rationale: str | None = None
    citations: list[RawCitation] = Field(default_factory=list)


class RawActionItem(BaseModel):
    """An action item as proposed by the LLM.

    ``assignee`` is the *display* name the model believes is responsible; it is
    resolved to a real participant during grounding when possible.
    """

    task: str
    assignee: str | None = None
    citations: list[RawCitation] = Field(default_factory=list)


class RawOpenQuestion(BaseModel):
    """An open question as proposed by the LLM."""

    question: str
    citations: list[RawCitation] = Field(default_factory=list)


class RawDecisionLog(BaseModel):
    """The structured JSON the LLM is asked to produce for a thread."""

    decisions: list[RawDecision] = Field(default_factory=list)
    action_items: list[RawActionItem] = Field(default_factory=list)
    open_questions: list[RawOpenQuestion] = Field(default_factory=list)


RETRY_PROMPT_SUFFIX = (
    "\n\nYour previous response did not match the required schema. "
    "Error: {error}. Please return only valid JSON matching the schema."
)
"""Suffix appended to the prompt for the single bounded retry on schema failure."""


@runtime_checkable
class LLMBackend(Protocol):
    """A pluggable structured-output LLM backend.

    Implementations must perform **at most one** bounded retry on a schema-validation
    failure (re-prompting with the validation error appended) and then raise
    :class:`LLMError` rather than looping.
    """

    def complete_json(self, prompt: str, schema: type[ModelT]) -> ModelT:
        """Return an instance of ``schema`` parsed from the model's JSON output.

        Args:
            prompt: The fully-rendered prompt.
            schema: A Pydantic model subclass describing the expected JSON shape.

        Returns:
            A validated instance of ``schema``.

        Raises:
            LLMError: If the backend cannot produce schema-valid output after one
                bounded retry.
        """
        ...


__all__ = [
    "RETRY_PROMPT_SUFFIX",
    "LLMBackend",
    "LLMError",
    "ModelT",
    "RawActionItem",
    "RawCitation",
    "RawDecision",
    "RawDecisionLog",
    "RawOpenQuestion",
]
