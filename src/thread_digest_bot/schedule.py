"""Scheduling abstraction for periodic rollups.

The :class:`Scheduler` protocol lets pure logic stay clock-free: a job is registered
and *triggered* externally. The default :class:`IntervalScheduler` is a minimal
in-process loop; tests use :class:`FakeScheduler` to fire jobs deterministically.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

Job = Callable[[], None]


@runtime_checkable
class Scheduler(Protocol):
    """Registers and runs periodic jobs.

    Implementations decide *when* jobs fire; pure logic never reads the clock.
    """

    def every(self, seconds: float, job: Job, *, name: str | None = None) -> None:
        """Register ``job`` to run every ``seconds`` seconds."""
        ...

    def start(self) -> None:
        """Begin firing registered jobs."""
        ...

    def stop(self) -> None:
        """Stop firing jobs."""
        ...


@dataclass
class _Registration:
    seconds: float
    job: Job
    name: str


@dataclass
class FakeScheduler:
    """A deterministic scheduler for tests.

    Jobs are registered but never auto-fire; tests call :meth:`fire_all` (or
    :meth:`fire`) to trigger them, making scheduled behavior fully reproducible.
    """

    registrations: list[_Registration] = field(default_factory=list)
    started: bool = False

    def every(self, seconds: float, job: Job, *, name: str | None = None) -> None:
        """Register a job; it will only run when explicitly fired."""
        self.registrations.append(
            _Registration(seconds=seconds, job=job, name=name or f"job-{len(self.registrations)}")
        )

    def start(self) -> None:
        """Mark the scheduler started (no background thread in the fake)."""
        self.started = True

    def stop(self) -> None:
        """Mark the scheduler stopped."""
        self.started = False

    def fire_all(self) -> None:
        """Synchronously run every registered job once, in registration order."""
        for registration in self.registrations:
            registration.job()

    def fire(self, name: str) -> None:
        """Run the named job once.

        Raises:
            KeyError: If no job with that name is registered.
        """
        for registration in self.registrations:
            if registration.name == name:
                registration.job()
                return
        raise KeyError(name)


class IntervalScheduler:
    """A minimal in-process interval scheduler using background timer threads.

    Each registered job runs on its own daemon thread on a fixed interval. This is the
    default production scheduler; it is intentionally simple (single process,
    best-effort) since rollups are coarse-grained.
    """

    def __init__(self) -> None:
        self._registrations: list[_Registration] = []
        self._timers: list[threading.Timer] = []
        self._running = False
        self._lock = threading.Lock()

    def every(self, seconds: float, job: Job, *, name: str | None = None) -> None:
        """Register ``job`` to run every ``seconds`` seconds."""
        if seconds <= 0:
            raise ValueError("interval seconds must be positive")
        self._registrations.append(
            _Registration(seconds=seconds, job=job, name=name or f"job-{len(self._registrations)}")
        )

    def _schedule(self, registration: _Registration) -> None:
        with self._lock:
            if not self._running:
                return

            def run_and_reschedule() -> None:
                try:
                    registration.job()
                finally:
                    self._schedule(registration)

            timer = threading.Timer(registration.seconds, run_and_reschedule)
            timer.daemon = True
            self._timers.append(timer)
            timer.start()

    def start(self) -> None:
        """Start all registered jobs."""
        with self._lock:
            self._running = True
        for registration in self._registrations:
            self._schedule(registration)

    def stop(self) -> None:
        """Stop all jobs and cancel pending timers."""
        with self._lock:
            self._running = False
            for timer in self._timers:
                timer.cancel()
            self._timers.clear()
