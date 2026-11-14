"""A second stop must not return before the first one has finished.

A session stops itself when its source ends, on the capture thread. A caller
that asks to stop at the same moment used to get control back at once, with
the session still running or stopping, and its next call, finalize_session,
was refused with ``live.not_stopped``. With real speech recognition the
trailing utterance is transcribed inside that stop, so the window is seconds
wide: test_real_transcript_survives_finalisation_into_another_process failed
every run once speech started arriving close to the end of the source.
"""
from __future__ import annotations

import threading
import time

from watch_skill.live import db
from watch_skill.live.session import RunningSession
from watch_skill.live.types import (
    LiveSession,
    LiveSourceKind,
    LiveSourceSpec,
    LiveState,
)


class _SlowStoppingSource:
    """A source whose stop takes long enough to be overlapped."""

    def __init__(self, stop_seconds: float) -> None:
        self.stop_seconds = stop_seconds
        self.stop_entered = threading.Event()
        self.running = True

    def stop(self) -> None:
        self.stop_entered.set()
        time.sleep(self.stop_seconds)
        self.running = False


def _runner(source: _SlowStoppingSource) -> RunningSession:
    session = LiveSession(
        session_id="live_concurrentstop",
        spec=LiveSourceSpec(kind=LiveSourceKind.FILE_REPLAY, target="unused.mp4",
                            audio=False),
        state=LiveState.RUNNING,
    )
    db.insert_session(session)
    return RunningSession(session, source)


def test_a_second_stop_waits_for_the_stop_in_progress() -> None:
    source = _SlowStoppingSource(stop_seconds=1.0)
    runner = _runner(source)

    owner = threading.Thread(target=runner.stop, kwargs={"reason": "source ended"})
    owner.start()
    assert source.stop_entered.wait(5.0), "the first stop never reached the source"

    runner.stop(reason="stopped by request")

    try:
        stored = db.get_session(runner.session.session_id)
        assert stored is not None
        assert stored.state is LiveState.STOPPED, (
            f"stop returned while the session was still {stored.state.value}")
    finally:
        owner.join(timeout=10.0)


def test_the_owning_thread_does_not_wait_on_itself() -> None:
    """Stop re-entered from the thread already stopping must return at once."""
    source = _SlowStoppingSource(stop_seconds=0.0)
    runner = _runner(source)
    inner: dict[str, float] = {}

    def reenter() -> None:
        started = time.monotonic()
        runner.stop(reason="re-entered")
        inner["seconds"] = time.monotonic() - started

    source.stop = lambda: reenter()  # type: ignore[method-assign]
    runner.stop(reason="stopped by request")

    assert inner["seconds"] < 1.0
    stored = db.get_session(runner.session.session_id)
    assert stored is not None and stored.state is LiveState.STOPPED
