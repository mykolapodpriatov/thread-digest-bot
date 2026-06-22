"""Digest tests — prompt building, the bounded retry, determinism, empty threads."""

from __future__ import annotations

import pytest

from thread_digest_bot.digest import build_prompt, digest
from thread_digest_bot.llm import LLMError, RawDecisionLog
from thread_digest_bot.llm.fake import FakeLLM
from thread_digest_bot.types import Author, Message, Thread


def _thread() -> Thread:
    ada = Author(id="u_ada", display="Ada")
    bob = Author(id="u_bob", display="Bob")
    cleo = Author(id="u_cleo", display="Cleo")
    return Thread(
        channel_id="team-eng",
        platform="telegram",
        messages=[
            Message(id="m1", author=ada, text="Let's ship the onboarding flow", ts_label="t1"),
            Message(id="m2", author=bob, text="I'll write the release notes", ts_label="t2"),
            Message(
                id="m3",
                author=cleo,
                text="rollback plan is ready; should we gate it behind a flag",
                ts_label="t3",
            ),
        ],
    )


def test_digest_happy_path_produces_expected_log() -> None:
    log = digest(_thread(), FakeLLM("happy"))
    assert [d.statement for d in log.decisions] == ["Ship the new onboarding flow on Friday."]
    assert [a.task for a in log.action_items] == ["Write the release notes."]
    assert [q.question for q in log.open_questions] == [
        "Do we need a feature flag for the rollout?"
    ]
    # Citations were grounded to real authors.
    assert log.decisions[0].citations[0].author.display == "Ada"
    assert log.action_items[0].assignee is not None
    assert log.action_items[0].assignee.display == "Bob"


def test_build_prompt_includes_ids_authors_and_text() -> None:
    prompt = build_prompt(_thread())
    assert "[id=m1] Ada: Let's ship the onboarding flow" in prompt
    assert "[id=m2] Bob: I'll write the release notes" in prompt
    assert "Return only the JSON object." in prompt


def test_build_prompt_notes_truncation() -> None:
    thread = Thread(
        channel_id="c", platform="telegram", messages=_thread().messages, truncated=True
    )
    prompt = build_prompt(thread)
    assert "truncated" in prompt.lower()


def test_empty_thread_yields_empty_log_without_calling_llm() -> None:
    llm = FakeLLM("happy")
    thread = Thread(channel_id="c", platform="telegram", messages=[])
    log = digest(thread, llm)
    assert log.is_empty()
    assert llm.calls == 0  # the model is never invoked for an empty thread
    assert log.range_label == "empty thread"


def test_schema_mismatch_retries_then_succeeds() -> None:
    llm = FakeLLM("schema_mismatch")
    log = digest(_thread(), llm)
    # One retry happened (two underlying calls) and the retry prompt carried the error.
    assert llm.calls == 2
    assert "did not match the required schema" in llm.prompts[1]
    assert not log.is_empty()


def test_schema_mismatch_persistent_raises_clear_error() -> None:
    llm = FakeLLM("schema_mismatch_persistent")
    with pytest.raises(LLMError) as excinfo:
        digest(_thread(), llm)
    # Bounded: exactly one retry then raise (no infinite loop), and the error is clear.
    assert llm.calls == 2
    assert "did not match" in str(excinfo.value)
    assert "RawDecisionLog" in str(excinfo.value)


def test_digest_key_is_deterministic_for_same_message_set() -> None:
    thread = _thread()
    key_a = digest(thread, FakeLLM("happy")).digest_key
    key_b = digest(thread, FakeLLM("happy")).digest_key
    assert key_a == key_b


def test_digest_key_independent_of_message_order() -> None:
    thread = _thread()
    reordered = Thread(
        channel_id=thread.channel_id,
        platform=thread.platform,
        messages=list(reversed(thread.messages)),
    )
    original_key = digest(thread, FakeLLM("happy")).digest_key
    reordered_key = digest(reordered, FakeLLM("happy")).digest_key
    assert original_key == reordered_key


def test_digest_key_changes_with_message_set() -> None:
    thread = _thread()
    fewer = Thread(
        channel_id=thread.channel_id,
        platform=thread.platform,
        messages=thread.messages[:2],
    )
    assert digest(thread, FakeLLM("happy")).digest_key != digest(fewer, FakeLLM("happy")).digest_key


def test_invalid_ids_fixture_drops_extra_item() -> None:
    # The raw fixture has 2 decisions; one cites a non-existent id and is dropped.
    raw = FakeLLM("invalid_ids")._payload()
    assert len(raw.decisions) == 2
    log = digest(_thread(), FakeLLM("invalid_ids"))
    assert len(log.decisions) == 1


def test_range_label_override() -> None:
    log = digest(_thread(), FakeLLM("happy"), range_label="last week")
    assert log.range_label == "last week"


def test_default_range_label_single_message() -> None:
    ada = Author(id="u_ada", display="Ada")
    thread = Thread(
        channel_id="c",
        platform="telegram",
        messages=[
            Message(id="m1", author=ada, text="Let's ship the onboarding flow", ts_label="t")
        ],
    )
    log = digest(thread, FakeLLM("happy"))
    assert log.range_label == "1 message"


def test_explicit_raw_payload_round_trips() -> None:
    raw = RawDecisionLog()
    llm = FakeLLM(raw=raw)
    log = digest(_thread(), llm)
    assert log.is_empty()
    assert llm.calls == 1
