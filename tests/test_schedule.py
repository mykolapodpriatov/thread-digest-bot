"""Scheduler tests — FakeScheduler determinism and the IntervalScheduler lifecycle."""

from __future__ import annotations

import threading
import time

import pytest

from thread_digest_bot.schedule import FakeScheduler, IntervalScheduler


def test_fake_scheduler_fires_on_demand() -> None:
    counter = {"n": 0}
    sched = FakeScheduler()
    sched.every(60, lambda: counter.__setitem__("n", counter["n"] + 1), name="job")

    assert counter["n"] == 0  # nothing auto-fires
    sched.start()
    assert sched.started is True
    sched.fire("job")
    assert counter["n"] == 1
    sched.fire_all()
    assert counter["n"] == 2
    sched.stop()
    assert sched.started is False


def test_fake_scheduler_unknown_job_raises() -> None:
    sched = FakeScheduler()
    with pytest.raises(KeyError):
        sched.fire("missing")


def test_fake_scheduler_autonames_jobs() -> None:
    sched = FakeScheduler()
    sched.every(1, lambda: None)
    sched.every(1, lambda: None)
    names = {r.name for r in sched.registrations}
    assert names == {"job-0", "job-1"}


def test_interval_scheduler_rejects_nonpositive_interval() -> None:
    sched = IntervalScheduler()
    with pytest.raises(ValueError, match="positive"):
        sched.every(0, lambda: None)


def test_interval_scheduler_runs_and_stops() -> None:
    fired = threading.Event()
    sched = IntervalScheduler()
    sched.every(0.01, fired.set, name="tick")
    sched.start()
    try:
        assert fired.wait(timeout=2.0) is True
    finally:
        sched.stop()
    # After stop, no timers remain pending.
    assert sched._timers == []


def test_interval_scheduler_reschedules() -> None:
    count = {"n": 0}
    lock = threading.Lock()
    reached = threading.Event()

    def job() -> None:
        with lock:
            count["n"] += 1
            if count["n"] >= 3:
                reached.set()

    sched = IntervalScheduler()
    sched.every(0.01, job)
    sched.start()
    try:
        assert reached.wait(timeout=3.0) is True
    finally:
        sched.stop()
    # Give any in-flight timer a moment to settle, then assert it stopped growing.
    with lock:
        observed = count["n"]
    time.sleep(0.05)
    with lock:
        assert count["n"] - observed <= 1  # at most one more fire could be in flight
