"""The 0.5 s overlap between live utterances must not publish words twice (A6).

Utterances overlap so a word on the boundary survives, which means the words
inside the overlap are recognised by both pieces. Before this, "the checkout
total" followed by "checkout total is wrong" reached the event log verbatim.
"""
from __future__ import annotations

import math
import struct
import threading
import time

import pytest

from watch_skill.live.asr import DeterministicASR
from watch_skill.live.audio import SAMPLE_RATE, AudioChunk, UtteranceAssembler
from watch_skill.transcribe.overlap import strip_overlap_echo


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ("the checkout total", "checkout total is wrong", ("is wrong", 2)),
        ("we ship on friday", "friday and then rest", ("and then rest", 1)),
        ("hello there", "general kenobi", ("general kenobi", 0)),
        ("The Checkout TOTAL.", "checkout, total is wrong", ("is wrong", 2)),
        ("checkout total", "checkout total", ("", 2)),
        ("", "anything", ("anything", 0)),
    ],
    ids=["exact", "partial", "none", "case-and-punctuation", "all-echo", "no-previous"],
)
def test_strip_overlap_echo(previous: str, current: str, expected: tuple[str, int]) -> None:
    assert strip_overlap_echo(previous, current) == expected


def test_only_the_tail_of_previous_counts() -> None:
    """A word repeated from the middle of the last piece is not an echo."""
    assert strip_overlap_echo("total is wrong today", "total again") == ("total again", 0)


def test_max_words_bounds_the_match() -> None:
    text = "one two three four five six seven eight nine ten"
    assert strip_overlap_echo(text, text, max_words=3) == (text, 0)


def test_overlap_metadata_on_utterances() -> None:
    assembler = UtteranceAssembler(session_id="s", utterance_seconds=2.0, overlap_seconds=0.5)
    first = second = None
    for i in range(4):
        out = assembler.add(_chunk(i))
        if out is not None and first is None:
            first = out
        elif out is not None:
            second = out
    assert first.leading_overlap_seconds == 0.0
    assert second.leading_overlap_seconds == pytest.approx(0.5)
    tail = UtteranceAssembler(session_id="s", utterance_seconds=10.0)
    tail.add(_chunk(0))
    assert tail.flush().leading_overlap_seconds == 0.0


def _chunk(i: int) -> AudioChunk:
    pcm = b"".join(struct.pack("<h", int(8000 * math.sin(n * 0.05))) for n in range(SAMPLE_RATE))
    return AudioChunk(session_id="s", seq=i, pcm=pcm, media_ts=float(i), wall_ts=time.time())


class _FakeSource:
    """Eight seconds of loud audio, as fast as the runtime will take it."""

    running = True

    def chunks(self, session_id: str):
        for i in range(8):
            yield _chunk(i)

    def stop(self) -> None:
        self.running = False


def test_the_runtime_publishes_an_overlapped_phrase_once() -> None:
    from watch_skill.live.audio_pipeline import AudioRuntime

    # 0-4 s is the first utterance; the second starts at 3.5 s. The cue at
    # 3.6-3.9 s sits inside the overlap, so both pieces hear it.
    backend = DeterministicASR([
        {"start": 0.5, "end": 3.0, "text": "the"},
        {"start": 3.6, "end": 3.9, "text": "checkout total"},
        {"start": 5.0, "end": 7.0, "text": "is wrong"},
    ])
    published: list = []
    done = threading.Event()

    def on_speech(piece) -> None:
        published.append(piece)
        if len(published) >= 2:
            done.set()

    runtime = AudioRuntime("s", _FakeSource(), backend, on_speech)
    runtime.start()
    done.wait(timeout=20)
    runtime.stop()

    texts = [p.text for p in published]
    assert texts[:2] == ["the checkout total", "is wrong"], texts
    assert published[1].overlap_trimmed_words == 2
    assert published[0].overlap_trimmed_words == 0


def test_an_all_echo_piece_is_dropped_and_counted() -> None:
    from watch_skill.live.audio_pipeline import AudioRuntime

    backend = DeterministicASR([{"start": 3.6, "end": 3.9, "text": "checkout total"}])
    published: list = []
    runtime = AudioRuntime("s", _FakeSource(), backend, published.append)
    runtime.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and runtime.stats()["utterances"] < 2:
        time.sleep(0.05)
    time.sleep(0.5)
    runtime.stop()
    assert [p.text for p in published] == ["checkout total"]
    assert runtime.stats()["overlap_echo_dropped"] == 1
