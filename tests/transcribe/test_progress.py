"""Segment-level progress while local whisper runs.

A long local transcription sat at one "transcribing" message for minutes, so
an agent polling a job could not tell slow from stuck, and a cancel only
landed after the whole file was done.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from watch_skill.errors import TranscriptionError


class _RawSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text = start, end, text
        self.words: list = []


def _install_whisper(monkeypatch: pytest.MonkeyPatch, produced: list[int]) -> None:
    """Five 10 s segments over a 50 s file, yielded lazily like the real one."""

    class FakeModel:
        def __init__(self, size: str, device: str, compute_type: str) -> None:
            pass

        def transcribe(self, path: str, **kwargs: object):
            def segments():
                for i in range(5):
                    produced.append(i)
                    yield _RawSegment(i * 10.0, i * 10.0 + 10.0, f"part {i}")

            return segments(), types.SimpleNamespace(
                language="en", language_probability=0.9, duration=50.0
            )

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setenv("WATCHSKILL_WHISPER_DEVICE", "cpu")


def _audio(tmp_path: Path) -> Path:
    audio = tmp_path / "long talk.mp3"
    audio.write_bytes(b"")
    return audio


def test_progress_rises_monotonically_to_the_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from watch_skill.transcribe.local import transcribe_local

    _install_whisper(monkeypatch, [])
    seen: list[tuple[float, float, int]] = []
    transcript = transcribe_local(
        _audio(tmp_path), model_size="tiny", on_progress=lambda d, t, n: seen.append((d, t, n))
    )
    assert len(transcript.segments) == 5
    done = [d for d, _, _ in seen]
    assert done == sorted(done) and len(seen) >= 2
    assert seen[-1] == (50.0, 50.0, 5), "the final call reports completion"
    assert all(total == 50.0 for _, total, _ in seen)


def test_a_raising_callback_stops_transcription(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Job cancellation raises from the callback; it must not be swallowed."""
    from watch_skill.transcribe.local import transcribe_local

    class Cancelled(Exception):
        pass

    produced: list[int] = []
    _install_whisper(monkeypatch, produced)

    def cancel(done: float, total: float, n: int) -> None:
        if n >= 2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        transcribe_local(_audio(tmp_path), model_size="tiny", on_progress=cancel)
    assert len(produced) < 5, "whisper kept decoding after the cancel"


def test_the_ladder_forwards_progress_to_local_whisper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from watch_skill.transcribe import audio as audio_mod
    from watch_skill.transcribe.ladder import get_transcript

    _install_whisper(monkeypatch, [])
    monkeypatch.setattr(
        audio_mod, "extract_audio",
        lambda video, dest, start_seconds=None, end_seconds=None: Path(dest),
    )
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    seen: list = []
    get_transcript(video, tmp_path, allow_local=True, allow_cloud=False,
                   whisper_model="tiny", on_progress=lambda *a: seen.append(a))
    assert seen and seen[-1][2] == 5


def test_the_job_maps_whisper_progress_to_the_transcribe_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an unknown phase string fell through to PROBE (0.15)."""
    import watch_skill.watch as watch_mod
    from watch_skill.jobs import handlers
    from watch_skill.jobs.types import JobStage

    def fake_watch(source, on_progress=None, **kwargs):
        on_progress("transcribing (local whisper) 25/50s, 3 segments", 0.8)
        raise TranscriptionError("stop", code="transcribe.test_stop", fix="none")

    monkeypatch.setattr(watch_mod, "watch", fake_watch)
    checkpoints: list[tuple] = []
    ctx = types.SimpleNamespace(
        payload={"source": "x.mp4"},
        checkpoint=lambda stage=None, progress=None: checkpoints.append((stage, progress)),
    )
    with pytest.raises(TranscriptionError):
        handlers._watch_job(ctx)
    stage, progress = checkpoints[-1]
    assert stage == JobStage.TRANSCRIBE
    ocr, transcribe = handlers._STAGE_PROGRESS[JobStage.OCR], handlers._STAGE_PROGRESS[JobStage.TRANSCRIBE]
    assert ocr < progress < transcribe
    assert progress == pytest.approx(ocr + (transcribe - ocr) * 0.5)


def test_watch_reports_whisper_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import watch_skill.watch as watch_mod
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind
    from watch_skill.perceive.types import VideoMetadata
    from watch_skill.transcribe.types import Transcript

    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(watch_mod, "acquire", lambda *a, **k: AcquireResult(
        source=str(video), kind=SourceKind.LOCAL_FILE, video_path=video, acquirer="local"))
    monkeypatch.setattr(watch_mod, "probe", lambda p: VideoMetadata(
        duration_seconds=50.0, width=None, height=None, fps=None, codec=None, has_audio=True))

    def fake_transcript(*a, on_progress=None, **k):
        on_progress(25.0, 50.0, 3)
        return Transcript(source="whisper-local (tiny)")

    monkeypatch.setattr(watch_mod, "get_transcript", fake_transcript)
    phases: list[tuple[str, float]] = []
    watch_mod.watch(str(video), out_dir=tmp_path / "w",
                    on_progress=lambda p, f: phases.append((p, f)))
    assert ("transcribing (local whisper) 25/50s, 3 segments", pytest.approx(0.8)) in phases
