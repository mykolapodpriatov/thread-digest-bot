"""Application service wiring fetch -> digest -> ground -> post -> store.

:class:`DigestService` is the platform-agnostic core of the bot. It registers the
``/digest`` command on any :class:`~thread_digest_bot.platforms.ChatPlatform`, and on
invocation it fetches the requested thread, digests it (with hard grounding), posts the
chat reply, and appends the entry to the append-only Git store — surfacing typed fetch
errors as friendly messages instead of a silent empty digest.

It also schedules per-channel rollups via the platform's injected scheduler, deduping
re-fires on the rollup's period-scoped key through the store (so a new period is always
recorded while the same period re-firing is a no-op).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from thread_digest_bot.digest import digest
from thread_digest_bot.grounding import GroundingPolicy
from thread_digest_bot.llm import LLMBackend, LLMError
from thread_digest_bot.platforms import (
    ChatPlatform,
    CommandContext,
    FetchError,
    ThreadNotFoundError,
)
from thread_digest_bot.render import render_chat_reply
from thread_digest_bot.rollup import build_rollup
from thread_digest_bot.store import DecisionStore
from thread_digest_bot.types import DecisionLog

PeriodKeyProvider = Callable[[], str]
"""A clock-free provider of the current rollup period key."""


@dataclass(frozen=True)
class DigestOutcome:
    """The result of handling a ``/digest`` invocation.

    Attributes:
        log: The grounded log, if digesting succeeded.
        reply_message_id: The id of the posted reply, if one was posted.
        committed: Whether a new commit was created in the store.
        skipped_duplicate: Whether the store treated this as an idempotent no-op.
        error_message: A friendly error message, if the request failed.
    """

    log: DecisionLog | None = None
    reply_message_id: str | None = None
    committed: bool = False
    skipped_duplicate: bool = False
    error_message: str | None = None


class DigestService:
    """Wires a chat platform, an LLM backend, and the decision store together.

    Args:
        platform: The chat platform to operate on.
        llm: The structured-output LLM backend.
        store: The append-only decision store.
        default_last_n: Default message count for ``/digest`` with no argument.
        policy: Optional grounding policy override applied to all digests.
    """

    def __init__(
        self,
        platform: ChatPlatform,
        llm: LLMBackend,
        store: DecisionStore,
        *,
        default_last_n: int = 200,
        policy: GroundingPolicy | None = None,
    ) -> None:
        self.platform = platform
        self.llm = llm
        self.store = store
        self.default_last_n = default_last_n
        self.policy = policy
        #: Outcomes recorded per invocation, in order (useful for tests/observability).
        self.history: list[DigestOutcome] = []

    def register(self, command: str = "digest") -> None:
        """Register the ``/digest`` command handler on the platform."""
        self.platform.register_command(command, self._on_digest)

    # -- command handling ----------------------------------------------------

    def _on_digest(self, ctx: CommandContext) -> None:
        outcome = self.handle_digest(
            ctx.channel_id,
            args=list(ctx.args),
            reply_to=ctx.reply_to,
        )
        self.history.append(outcome)
        # The success reply is already posted inside ``handle_digest``; here we only
        # surface a friendly *error* to the channel, so a successful /digest posts
        # exactly one reply.
        if outcome.error_message is not None:
            self.platform.post_reply(ctx.channel_id, outcome.error_message, reply_to=ctx.reply_to)

    def handle_digest(
        self,
        channel_id: str,
        *,
        args: list[str] | None = None,
        reply_to: str | None = None,
    ) -> DigestOutcome:
        """Run a single ``/digest`` request and return its outcome.

        Resolves the fetch parameters from ``args``/``reply_to``, applies the fetch
        contract, digests, posts the reply, and appends to the store. Typed fetch
        errors become friendly messages; no exception escapes for the normal
        not-found / API-failure paths.

        Args:
            channel_id: The channel to digest.
            args: Command argument tokens (e.g. ``["200"]``).
            reply_to: A replied-to message id, if the command was a thread reply.

        Returns:
            A :class:`DigestOutcome`.
        """
        last_n = self._resolve_last_n(args)
        try:
            thread = self.platform.fetch_thread(channel_id, reply_to=reply_to, last_n=last_n)
        except ThreadNotFoundError:
            return DigestOutcome(
                error_message=(
                    "I couldn't find that thread — the replied-to message may have "
                    "been deleted or is out of range."
                )
            )
        except FetchError:
            return DigestOutcome(
                error_message=(
                    "I couldn't read the messages (permissions or API error). "
                    "Please check the bot's access and try again."
                )
            )

        try:
            log = digest(thread, self.llm, policy=self.policy)
        except LLMError:
            return DigestOutcome(
                error_message="The digest model returned an unusable response; please retry."
            )

        reply_text = render_chat_reply(log)
        reply_id = self.platform.post_reply(channel_id, reply_text, reply_to=reply_to)
        result = self.store.append(log)

        return DigestOutcome(
            log=log,
            reply_message_id=reply_id,
            committed=result.committed,
            skipped_duplicate=result.skipped_duplicate,
        )

    def _resolve_last_n(self, args: list[str] | None) -> int:
        if args:
            for token in args:
                if token.isdigit():
                    return int(token)
        return self.default_last_n

    # -- scheduling ----------------------------------------------------------

    def schedule_rollup(
        self,
        channel_id: str,
        *,
        period: str,
        period_key_provider: PeriodKeyProvider,
        interval_seconds: float,
        last_n: int | None = None,
    ) -> None:
        """Schedule a periodic rollup for a channel via the platform scheduler.

        The rollup runs the same digest/ground/append pipeline; its key is scoped to
        ``(channel, period_key)``, so re-firing the *same* period is a store no-op (no
        duplicate commit) while a new period over the same messages is still recorded.

        Args:
            channel_id: Channel to roll up.
            period: Cadence label.
            period_key_provider: Callable returning the current period key (e.g.
                ``"2026-W25"``); injected so pure logic never reads the clock.
            interval_seconds: How often the scheduler fires the rollup.
            last_n: Window size; defaults to ``default_last_n``.
        """
        window = last_n or self.default_last_n

        def job() -> None:
            self.run_rollup_once(
                channel_id,
                period=period,
                period_key=period_key_provider(),
                last_n=window,
            )

        self.platform.scheduler().every(interval_seconds, job, name=f"rollup:{channel_id}")

    def run_rollup_once(
        self,
        channel_id: str,
        *,
        period: str,
        period_key: str,
        last_n: int,
    ) -> DigestOutcome:
        """Build and persist a single rollup; idempotent on ``(channel, period_key)``."""
        log = build_rollup(
            self.platform,
            self.llm,
            channel_id,
            period=period,
            period_key=period_key,
            last_n=last_n,
            policy=self.policy,
        )
        result = self.store.append(log)
        outcome = DigestOutcome(
            log=log,
            committed=result.committed,
            skipped_duplicate=result.skipped_duplicate,
        )
        self.history.append(outcome)
        return outcome
