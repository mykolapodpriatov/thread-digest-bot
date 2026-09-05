"""Exporting a committed decision to something other than Git.

A digest lands as an append-only Markdown entry in a Git repository. That is
the right home for it, and until now it was the only one: nothing downstream
could react to a decision being recorded. For a tool whose pitch is that
decisions stop getting lost in a thread, that is one hop short.

Three rules shape this, and they are the reason it is a module rather than a
`requests.post` in the store:

* **Delivery happens after the commit, never before.** A webhook that fires for
  a digest that then fails to commit reports a decision that is not recorded
  anywhere, which is worse than no webhook at all.
* **A failed delivery must not fail the digest.** Git is the source of truth and
  a flaky endpoint cannot be allowed to take it down. Errors are collected and
  returned; nothing here raises into the append path.
* **Redelivery is safe.** Every delivery carries a stable id derived from the
  digest, so a receiver can deduplicate a retry rather than recording the same
  decision twice.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import DecisionLog

__all__ = [
    "CollectingWebhookSink",
    "HttpWebhookSink",
    "WebhookDelivery",
    "WebhookSink",
    "delivery_for",
    "sign_body",
]

logger = logging.getLogger(__name__)

#: Header carrying the HMAC signature, prefixed with its algorithm.
SIGNATURE_HEADER = "X-Digest-Signature"
#: Header carrying the stable delivery id, so a receiver can deduplicate.
DELIVERY_ID_HEADER = "X-Digest-Delivery"

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5


@dataclass(frozen=True)
class WebhookDelivery:
    """One committed digest, ready to hand to something that is not Git.

    The payload is the structured facts, not the rendered Markdown. A consumer
    that wants Markdown can render it; one that wants fields cannot un-render
    it.
    """

    delivery_id: str
    channel_id: str
    range_label: str
    commit_sha: str | None
    commit_message: str | None
    path: str
    decisions: list[dict[str, Any]]
    action_items: list[dict[str, Any]]
    open_questions: list[dict[str, Any]]
    participants: list[dict[str, Any]]
    truncated: bool

    def to_json(self) -> str:
        """Serialize to a stable, sorted JSON body."""
        return json.dumps(
            {
                "delivery_id": self.delivery_id,
                "channel_id": self.channel_id,
                "range_label": self.range_label,
                "commit_sha": self.commit_sha,
                "commit_message": self.commit_message,
                "path": self.path,
                "decisions": self.decisions,
                "action_items": self.action_items,
                "open_questions": self.open_questions,
                "participants": self.participants,
                "truncated": self.truncated,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )


def delivery_for(
    log: DecisionLog,
    *,
    path: str,
    commit_sha: str | None = None,
    commit_message: str | None = None,
) -> WebhookDelivery:
    """Build a delivery from a committed log.

    ``delivery_id`` is derived from the channel and the digest key, which is
    already the deterministic identity of the exact message set that produced
    this log. So a redelivery of the same digest carries the same id and a
    receiver can deduplicate it, while a different digest never collides.
    """
    identity = f"{log.channel_id}\x00{log.digest_key}".encode()
    return WebhookDelivery(
        delivery_id=hashlib.sha256(identity).hexdigest()[:32],
        channel_id=log.channel_id,
        range_label=log.range_label,
        commit_sha=commit_sha,
        commit_message=commit_message,
        path=path,
        decisions=[d.model_dump(mode="json") for d in log.decisions],
        action_items=[a.model_dump(mode="json") for a in log.action_items],
        open_questions=[q.model_dump(mode="json") for q in log.open_questions],
        participants=[p.model_dump(mode="json") for p in log.participants],
        truncated=log.truncated,
    )


def sign_body(body: bytes, secret: str) -> str:
    """HMAC-SHA256 of ``body``, in the ``sha256=<hex>`` form receivers expect."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class WebhookSink:
    """A sink for exporting a committed digest.

    Subclass or duck-type it. :meth:`send` must not raise: the digest is
    already committed by the time it is called, and an exception here would
    turn a delivered decision into a failed run.
    """

    def send(self, delivery: WebhookDelivery) -> None:  # pragma: no cover - interface
        """Deliver ``delivery``. Must not raise."""
        raise NotImplementedError


@dataclass
class CollectingWebhookSink(WebhookSink):
    """An in-memory sink capturing deliveries, for tests and demos."""

    sent: list[WebhookDelivery] = field(default_factory=list)

    def send(self, delivery: WebhookDelivery) -> None:
        """Record ``delivery``."""
        self.sent.append(delivery)


#: The transport a sink uses. Takes a urllib Request, returns the status code.
Transport = Callable[[urllib.request.Request, float], int]


def _urlopen_transport(request: urllib.request.Request, timeout: float) -> int:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status)


@dataclass
class HttpWebhookSink(WebhookSink):
    """Posts each delivery as JSON to one or more endpoints.

    Args:
        urls: Endpoints to POST to. A Slack incoming webhook and an internal
            service can both be fed without a second pipeline; one failing
            never stops the others.
        secret: Optional shared secret. When set, each request carries an
            HMAC-SHA256 of the exact body it sends, so a receiver can tell a
            real delivery from anything else that finds the URL. It comes from
            the environment, never a config file, matching how LLM keys are
            already handled here.
        timeout: Per-request timeout in seconds.
        max_attempts: Total attempts per endpoint, including the first.
        backoff: Base delay between attempts; doubles each retry.
        transport: Injectable for tests, so the suite never opens a socket.
        sleep: Injectable for tests, so retries do not cost wall-clock.
    """

    urls: Sequence[str]
    secret: str | None = None
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff: float = DEFAULT_BACKOFF_SECONDS
    transport: Transport = _urlopen_transport
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.backoff < 0:
            raise ValueError(f"backoff must not be negative, got {self.backoff}")

    def send(self, delivery: WebhookDelivery) -> None:
        """POST ``delivery`` to every endpoint. Never raises."""
        body = delivery.to_json().encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            DELIVERY_ID_HEADER: delivery.delivery_id,
        }
        if self.secret:
            headers[SIGNATURE_HEADER] = sign_body(body, self.secret)

        for url in self.urls:
            self._post_with_retries(url, body, headers, delivery.delivery_id)

    def _post_with_retries(
        self, url: str, body: bytes, headers: dict[str, str], delivery_id: str
    ) -> None:
        """Try one endpoint until it takes the delivery or the bound is reached."""
        for attempt in range(1, self.max_attempts + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                self.transport(request, self.timeout)
                return
            except (urllib.error.URLError, OSError, ValueError) as exc:
                if attempt == self.max_attempts:
                    # Give up loudly in the log and quietly to the caller: the
                    # digest is committed and a flaky endpoint must not undo it.
                    logger.warning(
                        "webhook delivery %s to %s failed after %d attempt(s): %s",
                        delivery_id,
                        _redact(url),
                        attempt,
                        exc,
                    )
                    return
                self.sleep(self.backoff * (2 ** (attempt - 1)))


def _redact(url: str) -> str:
    """Drop the query string from a URL before logging it.

    Incoming-webhook URLs routinely carry the token in the path or the query;
    the path is needed to tell two endpoints apart, the query never is.
    """
    return url.split("?", 1)[0]
