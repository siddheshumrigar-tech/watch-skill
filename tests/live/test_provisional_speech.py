"""Provisional speech: a first guess before the utterance is complete.

A final piece waits for a full 4 s utterance, so an agent watching live heard
nothing for four seconds after someone started talking. A provisional piece
(final=False) is published once 2 s have accumulated; the final piece for the
same utterance supersedes it. Only final pieces are evidence.
"""
from __future__ import annotations

import math
import struct
import threading
import time

import pytest

from watch_skill.live.asr import DeterministicASR
from watch_skill.live.audio import SAMPLE_RATE, AudioChunk, EnergyGate, UtteranceAssembler
from watch_skill.live.types import LiveEvent, LiveEventType


def _chunk(i: int) -> AudioChunk:
    pcm = b"".join(struct.pack("<h", int(8000 * math.sin(n * 0.05))) for n in range(SAMPLE_RATE))
    return AudioChunk(session_id="s", seq=i, pcm=pcm, media_ts=float(i), wall_ts=time.time())


class _FakeSource:
    running = True

    def __init__(self, seconds: int = 8) -> None:
        self.seconds = seconds

    def chunks(self, session_id: str):
        for i in range(self.seconds):
            yield _chunk(i)

    def stop(self) -> None:
        self.running = False


def test_the_assembler_offers_one_provisional_per_utterance() -> None:
    assembler = UtteranceAssembler(session_id="s")
    assembler.add(_chunk(0))
    assert assembler.peek_provisional(2.0) is None, "1 s is under the threshold"
    assembler.add(_chunk(1))
    first = assembler.peek_provisional(2.0)
    assert first is not None and first.provisional
    assert first.index == 1 and first.duration == pytest.approx(2.0)
    assembler.add(_chunk(2))
    assert assembler.peek_provisional(2.0) is None, "at most once per pending utterance"
    final = assembler.add(_chunk(3))
    assert final is not None and not final.provisional and final.index == 1
    assembler.add(_chunk(4))
    assembler.add(_chunk(5))
    assert assembler.peek_provisional(2.0).index == 2


def _run(runtime, want: int) -> list:
    published: list = []
    done = threading.Event()

    def on_speech(piece) -> None:
        published.append(piece)
        if len(published) >= want:
            done.set()

    runtime.on_speech = on_speech
    runtime.start()
    done.wait(timeout=20)
    time.sleep(0.3)
    runtime.stop()
    return published


def test_a_provisional_piece_arrives_before_the_final_one() -> None:
    from watch_skill.live.audio_pipeline import AudioRuntime

    backend = DeterministicASR([{"start": 0.5, "end": 3.0, "text": "hello there"}])
    runtime = AudioRuntime("s", _FakeSource(4), backend, None, speech_gate=EnergyGate())
    pieces = _run(runtime, 2)
    first = [(p.final, p.utterance_index, p.text) for p in pieces if p.utterance_index == 1]
    assert first[:2] == [(False, 1, "hello there"), (True, 1, "hello there")], first
    assert pieces[1].supersedes_provisional is True
    assert runtime.stats()["provisional_published"] == 1


def test_the_setting_turns_provisional_speech_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.config import reset_settings
    from watch_skill.live.audio_pipeline import AudioRuntime

    monkeypatch.setenv("WATCHSKILL_LIVE_PROVISIONAL_SPEECH", "false")
    reset_settings()
    backend = DeterministicASR([{"start": 0.5, "end": 3.0, "text": "hello there"}])
    runtime = AudioRuntime("s", _FakeSource(4), backend, None, speech_gate=EnergyGate())
    pieces = _run(runtime, 1)
    assert [p.final for p in pieces] == [True]
    assert pieces[0].supersedes_provisional is False


def _speech(ts: float, text: str, *, final: bool) -> LiveEvent:
    return LiveEvent(session_id="s", seq=0, media_ts=ts, wall_ts=time.time(),
                     type=LiveEventType.SPEECH, final=final, summary=text)


def test_finalize_keeps_only_final_speech() -> None:
    from watch_skill.live.finalize import _events_to_transcript

    transcript = _events_to_transcript([
        _speech(0.0, "hello th", final=False),
        _speech(0.0, "hello there", final=True),
    ])
    assert [s.text for s in transcript.segments] == ["[speech] hello there"]


def test_live_ask_ignores_provisional_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from watch_skill.live import ask as live_ask
    from watch_skill.live import session as live_session

    events = [_speech(1.0, "the invoice is wrong", final=False),
              _speech(5.0, "everything is fine", final=True)]
    monkeypatch.setattr(live_ask.db, "read_events", lambda *a, **k: events)
    monkeypatch.setattr(live_session, "get_session",
                        lambda sid: SimpleNamespace(state=SimpleNamespace(value="running")))
    answer = live_ask.ask_live("s", "what is wrong with the invoice?", scope="session")
    cited = " ".join(str(e) for e in answer["evidence"])
    assert "invoice is wrong" not in cited
