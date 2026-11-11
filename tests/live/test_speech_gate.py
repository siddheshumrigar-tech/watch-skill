"""The live speech gate: Silero VAD when available, energy otherwise.

The energy gate drops anything quieter than a mean amplitude of 350, which is
a quiet speaker on a laptop microphone. Silero VAD ships inside faster-whisper
(its ONNX file is in the package), so preferring it costs no download.
"""
from __future__ import annotations

import math
import struct
import sys
import threading
import time
import types

import pytest

from watch_skill.live.asr import DeterministicASR
from watch_skill.live.audio import SAMPLE_RATE, AudioChunk


def _quiet_chunk(i: int, amplitude: int = 314) -> AudioChunk:
    """Mean |sample| of about 200: under the energy gate's 350."""
    pcm = b"".join(
        struct.pack("<h", int(amplitude * math.sin(n * 0.05))) for n in range(SAMPLE_RATE)
    )
    return AudioChunk(session_id="s", seq=i, pcm=pcm, media_ts=float(i), wall_ts=time.time())


@pytest.fixture(autouse=True)
def _auto_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite pins the energy gate; these tests are about choosing."""
    from watch_skill.config import reset_settings

    monkeypatch.setenv("WATCHSKILL_LIVE_SPEECH_GATE", "auto")
    reset_settings()


def _fake_vad(monkeypatch: pytest.MonkeyPatch, *, speech: bool = True, explode: bool = False):
    calls: list[int] = []

    class VadOptions:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def get_speech_timestamps(audio, vad_options=None, sampling_rate=16000, **kwargs):
        calls.append(len(audio))
        assert str(audio.dtype) == "float32" and abs(float(audio.max())) <= 1.0
        if explode:
            raise RuntimeError("onnxruntime cannot load silero_vad_v6.onnx")
        return [{"start": 0, "end": len(audio)}] if speech else []

    package = types.ModuleType("faster_whisper")
    vad = types.ModuleType("faster_whisper.vad")
    vad.VadOptions, vad.get_speech_timestamps = VadOptions, get_speech_timestamps
    package.vad = vad
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    return calls


class _QuietSource:
    running = True

    def chunks(self, session_id: str):
        for i in range(5):
            yield _quiet_chunk(i)

    def stop(self) -> None:
        self.running = False


def _run(runtime) -> None:
    runtime.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and runtime.stats()["utterances"] < 1:
        time.sleep(0.05)
    time.sleep(0.3)
    runtime.stop()


def test_quiet_speech_gets_through_the_silero_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the energy gate dropped this as silence."""
    from watch_skill.live.audio import is_probably_silent
    from watch_skill.live.audio_pipeline import AudioRuntime

    assert is_probably_silent(_quiet_chunk(0).pcm), "fixture must be under the energy gate"
    calls = _fake_vad(monkeypatch)
    published: list = []
    runtime = AudioRuntime(
        "s", _QuietSource(),
        DeterministicASR([{"start": 0.0, "end": 5.0, "text": "a quiet voice"}]),
        published.append,
    )
    _run(runtime)
    assert [p.text for p in published][:1] == ["a quiet voice"]
    assert calls, "the Silero gate was never asked"
    assert runtime.stats()["speech_gate"] == "silero"
    assert runtime.stats()["speech_gate_fallback_reason"] is None


def test_silero_saying_no_speech_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.live.audio import build_speech_gate

    _fake_vad(monkeypatch, speech=False)
    gate = build_speech_gate()
    assert gate.name == "silero"
    assert gate.is_speech(_quiet_chunk(0, amplitude=20000).pcm) is False


def test_without_faster_whisper_the_energy_gate_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.live.audio import build_speech_gate

    package = types.ModuleType("faster_whisper")
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)
    gate = build_speech_gate()
    assert gate.name == "energy"
    assert gate.fallback_reason and "silero" in gate.fallback_reason


def test_a_failing_first_silero_call_falls_back_to_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from watch_skill.live.audio import build_speech_gate

    _fake_vad(monkeypatch, explode=True)
    gate = build_speech_gate()
    loud = _quiet_chunk(0, amplitude=20000).pcm
    assert gate.is_speech(loud) is True, "the energy fallback decides instead"
    assert gate.name == "energy"
    assert "silero_vad_v6.onnx" in gate.fallback_reason


def test_the_setting_forces_the_energy_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.config import reset_settings
    from watch_skill.live.audio import build_speech_gate

    _fake_vad(monkeypatch)
    monkeypatch.setenv("WATCHSKILL_LIVE_SPEECH_GATE", "energy")
    reset_settings()
    gate = build_speech_gate()
    assert gate.name == "energy"
    assert gate.fallback_reason is None


def test_stats_and_status_name_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.live.audio_pipeline import AudioRuntime

    _fake_vad(monkeypatch)
    runtime = AudioRuntime("s", _QuietSource(), None, lambda piece: None)
    assert runtime.stats()["speech_gate"] == "silero"
    status = runtime.status()
    assert status["speech_gate"] == "silero"
    assert "speech_gate_fallback_reason" in status


def test_the_gate_is_thread_safe_enough_for_one_assembler() -> None:
    """One assembler thread calls it; the stop() flush may call it once more."""
    from watch_skill.live.audio import EnergyGate

    gate = EnergyGate()
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(gate.is_speech(b"\x00\x00" * 100)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [False] * 4
