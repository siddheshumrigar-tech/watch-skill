"""Non-speech sound events (A10): detection, failure modes, index, surfaces.

No model is downloaded: a scripted FakeClassifier stands in for YAMNet and a
fake InferenceSession stands in for onnxruntime.
"""
from __future__ import annotations

import math
import shutil
import struct
import sys
import types
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from watch_skill.errors import DependencyError  # noqa: E402
from watch_skill.transcribe import audio_events as ae  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")

RATE = 16_000


def _wav(path: Path, blocks: list[tuple[int, float]]) -> Path:
    """Blocks of (tone amplitude, seconds) at 16 kHz mono; amplitude 0 is silence."""
    frames = bytearray()
    for amplitude, seconds in blocks:
        for n in range(int(seconds * RATE)):
            value = int(amplitude * math.sin(2 * math.pi * 440 * n / RATE))
            frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(bytes(frames))
    return path


class FakeClassifier:
    """The loud first block is music, the quieter last one is applause."""

    name = "fake"

    def classify(self, samples) -> list[tuple[str, float]]:
        if float(np.abs(samples).max()) > 0.3:
            return [("Music", 0.9), ("Speech", 0.1)]
        return [("Applause", 0.8), ("Clapping", 0.7), ("Vehicle", 0.95)]


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return _wav(tmp_path / "tone silence tone.wav", [(12000, 2.0), (0, 2.0), (6000, 2.0)])


def test_windows_merge_into_labelled_events(clip: Path) -> None:
    events = ae.detect_audio_events(clip, FakeClassifier(), start_offset=100.0)
    labels = [(e.label, e.backend) for e in events]
    assert labels == [("music", "fake"), ("silence", "rms"), ("applause", "fake")]
    music, silence, applause = events
    # 0.96 s windows every 0.48 s: the last window touching the tone ends at 2.88 s.
    assert music.start == 100.0 and music.end == pytest.approx(102.88, abs=0.05)
    assert silence.start == pytest.approx(102.4, abs=0.05)
    assert silence.end == pytest.approx(103.84, abs=0.05) and silence.confidence == 1.0
    assert applause.end == pytest.approx(106.0, abs=0.05)
    assert applause.raw_label == "Applause", "an unmapped class never wins"
    assert set(music.to_public()) == {"start", "end", "label", "confidence", "raw_label", "backend"}


def test_silence_needs_no_model(clip: Path) -> None:
    events = ae.detect_audio_events(clip, None)
    assert [e.label for e in events] == ["silence"]
    assert events[0].raw_label is None


def test_min_confidence_drops_weak_windows(clip: Path) -> None:
    events = ae.detect_audio_events(clip, FakeClassifier(), min_confidence=0.85)
    assert [e.label for e in events] == ["music", "silence"]


@pytest.mark.parametrize(("raw", "coarse"), [
    ("Male speech, man speaking", "speech"), ("Speech synthesizer", "speech"),
    ("Belly laugh", "laughter"), ("Clapping", "applause"), ("Smoke detector, smoke alarm", "alarm"),
    ("Civil defense siren", "alarm"), ("Electric guitar", "music"), ("Silence", "silence"),
    ("Vehicle", None),
])
def test_audioset_names_map_to_coarse_labels(raw: str, coarse: str | None) -> None:
    assert ae.coarse_label(raw) == coarse


def test_without_onnxruntime_the_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(DependencyError) as info:
        ae.build_classifier()
    assert info.value.code == "audio_events.missing_dependency"
    assert 'pip install "watch-skill[audio-events]"' in info.value.fix


def test_without_a_configured_model_the_error_says_how(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    with pytest.raises(DependencyError) as info:
        ae.build_classifier()
    assert info.value.code == "audio_events.model_not_configured"
    assert "WATCHSKILL_AUDIO_EVENTS_MODEL_PATH" in info.value.fix


class _Input:
    def __init__(self, shape, kind="tensor(float)") -> None:
        self.name, self.shape, self.type = "waveform", shape, kind


class _Session:
    def __init__(self, inputs, labels: int = 3) -> None:
        self._inputs, self._labels = inputs, labels
        self.fed = None

    def get_inputs(self):
        return self._inputs

    def run(self, outputs, feed):
        self.fed = feed
        frames = np.array([[0.1, 0.7, 0.2], [0.3, 0.5, 0.2]], dtype=np.float32)
        return [frames]


def _labels(tmp_path: Path) -> Path:
    path = tmp_path / "yamnet_class_map.csv"
    path.write_text("index,mid,display_name\n0,/m/09x0r,Speech\n1,/m/04rlf,Music\n"
                    "2,/m/028ght,Applause\n", encoding="utf-8")
    return path


def test_the_onnx_backend_averages_frame_scores(tmp_path: Path) -> None:
    session = _Session([_Input([None])])
    classifier = ae.OnnxYamnetClassifier(tmp_path / "m.onnx", _labels(tmp_path), session=session)
    scores = classifier.classify(np.zeros(15360, dtype=np.float32))
    assert scores[0] == ("Music", pytest.approx(0.6))
    assert session.fed["waveform"].dtype == np.float32


@pytest.mark.parametrize("inputs", [
    [_Input([1, 15360])], [_Input([None], "tensor(int64)")], [_Input([None]), _Input([None])],
])
def test_an_unsupported_model_signature_is_refused(tmp_path: Path, inputs) -> None:
    with pytest.raises(DependencyError) as info:
        ae.OnnxYamnetClassifier(tmp_path / "m.onnx", _labels(tmp_path), session=_Session(inputs))
    assert info.value.code == "audio_events.unsupported_model"


def _speech(*_a, **_k):
    from watch_skill.transcribe.types import Segment, Transcript

    return Transcript(segments=[Segment(0.0, 2.0, "welcome to the keynote")],
                      source="whisper-local (tiny)")


def test_watch_detects_indexes_and_reports_sounds(
    monkeypatch: pytest.MonkeyPatch, clip: Path, tmp_path: Path
) -> None:
    import watch_skill.watch as watch_mod
    from watch_skill.report import render_report

    monkeypatch.setattr(watch_mod, "get_transcript", _speech)
    monkeypatch.setattr(ae, "build_classifier", FakeClassifier)
    result = watch_mod.watch(str(clip), out_dir=tmp_path / "w", audio_events=True)
    assert [e.label for e in result.audio_events] == ["music", "silence", "applause"]
    assert result.to_summary()["audio_events"] == {
        "count": 3, "labels": ["applause", "music", "silence"]}
    assert "## Sounds" in render_report(result)
    assert not [w for w in result.warnings if w["code"].startswith("audio_events")]


def test_a_missing_model_is_a_warning_and_silence_still_counts(
    monkeypatch: pytest.MonkeyPatch, clip: Path, tmp_path: Path
) -> None:
    """D7: the transcript comes back; the agent gets the code and the fix."""
    import watch_skill.watch as watch_mod

    monkeypatch.setattr(watch_mod, "get_transcript", _speech)
    result = watch_mod.watch(str(clip), out_dir=tmp_path / "w", audio_events=True)
    assert result.transcript.segments[0].text == "welcome to the keynote"
    [warning] = [w for w in result.warnings if w["code"].startswith("audio_events")]
    assert warning["code"] == "audio_events.model_not_configured" and warning["fix"]
    assert [e.label for e in result.audio_events] == ["silence"]


def test_sound_events_are_off_unless_asked(
    monkeypatch: pytest.MonkeyPatch, clip: Path, tmp_path: Path
) -> None:
    import watch_skill.watch as watch_mod

    monkeypatch.setattr(watch_mod, "get_transcript", _speech)
    monkeypatch.setattr(ae, "build_classifier", lambda: pytest.fail("not asked for"))
    assert watch_mod.watch(str(clip), out_dir=tmp_path / "w").audio_events == []


def test_the_surfaces_accept_audio_events() -> None:
    import inspect

    from watch_skill.surfaces.api.app import WatchRequest
    from watch_skill.surfaces.cli.main import watch as cli_watch

    assert WatchRequest(source="x", audio_events=True).audio_events is True
    assert "audio_events" in inspect.signature(cli_watch).parameters
    mcp = pytest.importorskip("watch_skill.surfaces.mcp.server")
    tool = mcp.watch_video
    fn = getattr(tool, "fn", tool)
    assert inspect.signature(fn).parameters["audio_events"].default is False
