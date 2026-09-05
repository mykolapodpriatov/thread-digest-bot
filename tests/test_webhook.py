"""Tests for exporting a committed decision.

Nothing here opens a socket: the transport and the sleep are injected, so
retries cost no wall-clock and the suite stays offline like the rest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from thread_digest_bot.store import DecisionStore, StoreConfig
from thread_digest_bot.types import Author, Citation, Decision, DecisionLog
from thread_digest_bot.webhook import (
    DELIVERY_ID_HEADER,
    SIGNATURE_HEADER,
    CollectingWebhookSink,
    HttpWebhookSink,
    delivery_for,
    sign_body,
)


def _log(key: str = "key-1", channel: str = "team-eng") -> DecisionLog:
    ada = Author(id="u_ada", display="Ada")
    return DecisionLog(
        channel_id=channel,
        range_label="last 3 messages",
        decisions=[
            Decision(
                statement="Ship Friday",
                citations=[
                    Citation(message_id="m1", author=ada, permalink="https://example.com/m1")
                ],
            )
        ],
        participants=[ada],
        digest_key=key,
    )


class RecordingTransport:
    """Captures requests and replays a scripted sequence of outcomes."""

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.requests: list[urllib.request.Request] = []
        self.outcomes = outcomes or []

    def __call__(self, request: urllib.request.Request, timeout: float) -> int:
        self.requests.append(request)
        if not self.outcomes:
            return 200
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return int(outcome)  # type: ignore[arg-type]


class RecordingSleep:
    def __init__(self) -> None:
        self.slept: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def test_the_payload_is_fields_not_rendered_markdown() -> None:
    # A consumer that wants Markdown can render it; one that wants fields
    # cannot un-render it.
    delivery = delivery_for(_log(), path="docs/decisions/team-eng.md", commit_sha="abc123")

    payload = json.loads(delivery.to_json())
    assert payload["decisions"][0]["statement"] == "Ship Friday"
    assert payload["decisions"][0]["citations"][0]["permalink"] == "https://example.com/m1"
    assert payload["commit_sha"] == "abc123"
    assert payload["path"] == "docs/decisions/team-eng.md"


def test_the_delivery_id_is_stable_for_the_same_digest() -> None:
    # A receiver has to be able to deduplicate a retry.
    first = delivery_for(_log(), path="p")
    second = delivery_for(_log(), path="p", commit_sha="different-run")

    assert first.delivery_id == second.delivery_id


def test_the_delivery_id_differs_for_a_different_digest() -> None:
    assert delivery_for(_log("key-1"), path="p").delivery_id != (
        delivery_for(_log("key-2"), path="p").delivery_id
    )
    assert delivery_for(_log(channel="a"), path="p").delivery_id != (
        delivery_for(_log(channel="b"), path="p").delivery_id
    )


def test_the_body_is_stable_across_runs() -> None:
    assert delivery_for(_log(), path="p").to_json() == delivery_for(_log(), path="p").to_json()


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------


def test_a_delivery_is_posted_as_json_with_its_id() -> None:
    transport = RecordingTransport()
    sink = HttpWebhookSink(urls=["https://example.com/hook"], transport=transport)
    delivery = delivery_for(_log(), path="p")

    sink.send(delivery)

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.method == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header(DELIVERY_ID_HEADER.capitalize()) == delivery.delivery_id
    assert json.loads(request.data.decode())["channel_id"] == "team-eng"


def test_the_signature_covers_the_exact_body_sent() -> None:
    transport = RecordingTransport()
    sink = HttpWebhookSink(urls=["https://example.com/hook"], secret="s3cret", transport=transport)

    sink.send(delivery_for(_log(), path="p"))

    request = transport.requests[0]
    expected = hmac.new(b"s3cret", request.data, hashlib.sha256).hexdigest()
    assert request.get_header(SIGNATURE_HEADER.capitalize()) == f"sha256={expected}"


def test_without_a_secret_no_signature_is_sent() -> None:
    transport = RecordingTransport()
    sink = HttpWebhookSink(urls=["https://example.com/hook"], transport=transport)

    sink.send(delivery_for(_log(), path="p"))

    assert transport.requests[0].get_header(SIGNATURE_HEADER.capitalize()) is None


def test_sign_body_is_the_documented_form() -> None:
    assert sign_body(b"body", "k").startswith("sha256=")


def test_a_failing_endpoint_is_retried_with_backoff() -> None:
    transport = RecordingTransport(
        [urllib.error.URLError("down"), urllib.error.URLError("still down"), 200]
    )
    sleep = RecordingSleep()
    sink = HttpWebhookSink(
        urls=["https://example.com/hook"],
        transport=transport,
        sleep=sleep,
        backoff=0.5,
        max_attempts=3,
    )

    sink.send(delivery_for(_log(), path="p"))

    assert len(transport.requests) == 3
    assert sleep.slept == [0.5, 1.0]


def test_retries_stop_at_the_bound_instead_of_looping() -> None:
    transport = RecordingTransport([urllib.error.URLError("down")] * 10)
    sink = HttpWebhookSink(
        urls=["https://example.com/hook"],
        transport=transport,
        sleep=RecordingSleep(),
        max_attempts=3,
    )

    sink.send(delivery_for(_log(), path="p"))

    assert len(transport.requests) == 3


def test_an_endpoint_that_never_answers_does_not_raise() -> None:
    # The digest is already committed; a flaky endpoint cannot undo that.
    sink = HttpWebhookSink(
        urls=["https://example.com/hook"],
        transport=RecordingTransport([urllib.error.URLError("down")] * 5),
        sleep=RecordingSleep(),
        max_attempts=2,
    )

    sink.send(delivery_for(_log(), path="p"))  # must not raise


def test_one_failing_endpoint_does_not_stop_the_others() -> None:
    transport = RecordingTransport([urllib.error.URLError("down"), 200])
    sink = HttpWebhookSink(
        urls=["https://bad.example/hook", "https://good.example/hook"],
        transport=transport,
        sleep=RecordingSleep(),
        max_attempts=1,
    )

    sink.send(delivery_for(_log(), path="p"))

    assert [r.full_url for r in transport.requests] == [
        "https://bad.example/hook",
        "https://good.example/hook",
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"max_attempts": 0}, "max_attempts"), ({"backoff": -1.0}, "backoff")],
)
def test_a_nonsense_retry_policy_is_rejected(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HttpWebhookSink(urls=["https://example.com/hook"], **kwargs)  # type: ignore[arg-type]


def test_no_secret_appears_in_the_payload() -> None:
    transport = RecordingTransport()
    sink = HttpWebhookSink(
        urls=["https://example.com/hook?token=supersecret"],
        secret="s3cret",
        transport=transport,
    )

    sink.send(delivery_for(_log(), path="p"))

    body = transport.requests[0].data.decode()
    assert "s3cret" not in body
    assert "supersecret" not in body


def test_a_failure_log_line_drops_the_query_string(caplog: pytest.LogCaptureFixture) -> None:
    # Incoming-webhook URLs routinely carry the token in the query.
    sink = HttpWebhookSink(
        urls=["https://example.com/hook?token=supersecret"],
        transport=RecordingTransport([urllib.error.URLError("down")]),
        sleep=RecordingSleep(),
        max_attempts=1,
    )

    with caplog.at_level("WARNING"):
        sink.send(delivery_for(_log(), path="p"))

    assert "supersecret" not in caplog.text
    assert "https://example.com/hook" in caplog.text


# ---------------------------------------------------------------------------
# ordering against the commit
# ---------------------------------------------------------------------------


def test_the_export_happens_after_the_commit(temp_git_repo: Path) -> None:
    """A delivery for a digest that then failed to commit would report a
    decision that is not recorded anywhere."""

    class ShaCapturingSink(CollectingWebhookSink):
        pass

    sink = ShaCapturingSink()
    store = DecisionStore(temp_git_repo, webhook=sink)
    result = store.append(_log())

    assert result.committed
    assert sink.sent[0].commit_sha == result.commit_sha
    assert sink.sent[0].commit_message == result.commit_message


def test_a_duplicate_append_exports_nothing(temp_git_repo: Path) -> None:
    sink = CollectingWebhookSink()
    store = DecisionStore(temp_git_repo, webhook=sink)
    store.append(_log())
    store.append(_log())

    assert len(sink.sent) == 1


def test_a_sink_that_raises_does_not_fail_the_digest(temp_git_repo: Path) -> None:
    class ExplodingSink:
        def send(self, delivery: object) -> None:
            raise RuntimeError("receiver on fire")

    store = DecisionStore(temp_git_repo, webhook=ExplodingSink())
    result = store.append(_log())

    assert result.committed
    assert result.path.read_text(encoding="utf-8")


def test_no_commit_mode_still_exports_without_a_sha(tmp_path: Path) -> None:
    sink = CollectingWebhookSink()
    store = DecisionStore(tmp_path, config=StoreConfig(commit=False), webhook=sink)
    store.append(_log())

    assert len(sink.sent) == 1
    assert sink.sent[0].commit_sha is None
