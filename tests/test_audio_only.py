"""Audio-only sources: no frames by design, and the agent is told so.

A podcast, a voice memo or an mp3 with cover art used to go through visual
perception anyway: a wav failed in frame extraction, an mp3's cover image was
probed as a 1-frame video, and the CLI never indexed a result without frames,
so the transcript of an audio file could not be asked about later.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from watch_skill.transcribe.types import Segment, Transcript

SUMMARY_KEYS = {
    "source", "media_kind", "duration_seconds", "focused", "transcript",
    "frames", "audio_events", "warnings",
}
TRANSCRIPT_KEYS = {
    "source", "language", "language_probability", "language_source",
    "segment_count", "has_speakers", "has_word_timestamps",
}

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _wav(path: Path, seconds: float = 3.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16_000)))
            for i in range(int(16_000 * seconds))
        )
        handle.writeframes(frames)
    return path


def _mp3_with_cover(path: Path) -> Path:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=f=440:d=3",
         "-f", "lavfi", "-i", "color=c=red:s=64x64",
         "-map", "0:a", "-map", "1:v", "-frames:v", "1",
         "-c:v", "mjpeg", "-disposition:v:0", "attached_pic", str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not build an mp3 with cover art: {result.stderr[-200:]}")
    return path


def _speech(*_a, **_k) -> Transcript:
    return Transcript(
        segments=[Segment(0.0, 2.5, "this is a recorded voice memo about the budget")],
        source="whisper-local (tiny)", language="en", language_probability=0.99,
        language_source="detected",
    )


@pytest.fixture
def no_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    import watch_skill.watch as watch_mod

    def perceive(*_a, **_k):
        raise AssertionError("visual perception ran on an audio-only source")

    monkeypatch.setattr(watch_mod, "perceive", perceive)
    monkeypatch.setattr(watch_mod, "get_transcript", _speech)


def test_cover_art_is_not_a_video_stream(tmp_path: Path) -> None:
    from watch_skill.perceive.media import probe

    meta = probe(_mp3_with_cover(tmp_path / "with cover.mp3"))
    assert meta.width is None and meta.height is None
    assert meta.has_audio


def test_a_wav_skips_perception_and_says_why(tmp_path: Path, no_frames: None) -> None:
    from watch_skill.report import render_report
    from watch_skill.watch import watch

    result = watch(str(_wav(tmp_path / "voice memo.wav")), out_dir=tmp_path / "work")
    assert result.media_kind == "audio"
    assert result.perception is None
    report = render_report(result)
    assert "- **Media:** audio-only (no video stream; frames, scenes and OCR skipped by design)" in report
    assert "## Machine summary" in report

    summary = result.to_summary()
    assert set(summary) == SUMMARY_KEYS
    assert set(summary["transcript"]) == TRANSCRIPT_KEYS
    assert summary["media_kind"] == "audio"
    assert summary["frames"] == {"count": 0, "seconds_per_frame": None, "sparse": False}
    assert summary["audio_events"] == {"count": 0, "labels": []}
    assert summary["transcript"]["segment_count"] == 1
    assert summary["warnings"] == []
    json.dumps(summary)  # JSON-safe


def test_visual_options_on_audio_are_reported_not_silently_ignored(
    tmp_path: Path, no_frames: None,
) -> None:
    from watch_skill.watch import watch

    result = watch(str(_wav(tmp_path / "memo.wav")), out_dir=tmp_path / "w", max_frames=10)
    codes = [w["code"] for w in result.warnings]
    assert codes == ["perceive.audio_only_ignored_options"]
    assert set(result.warnings[0]) == {"code", "message", "fix"}


def test_an_mp3_with_cover_art_is_audio(tmp_path: Path, no_frames: None) -> None:
    from watch_skill.watch import watch

    result = watch(str(_mp3_with_cover(tmp_path / "episode.mp3")), out_dir=tmp_path / "w")
    assert result.media_kind == "audio"


def test_an_empty_transcript_is_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_frames: None,
) -> None:
    import watch_skill.watch as watch_mod
    from watch_skill.report import render_report
    from watch_skill.watch import watch

    monkeypatch.setattr(watch_mod, "get_transcript", lambda *a, **k: Transcript(source="none"))
    result = watch(str(_wav(tmp_path / "memo.wav")), out_dir=tmp_path / "w")
    warning = next(w for w in result.warnings if w["code"] == "transcribe.no_transcript")
    assert 'pip install "watch-skill[whisper]"' in warning["fix"]
    assert "## Warnings" in render_report(result)


def test_cli_json_indexes_an_audio_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_frames: None,
) -> None:
    from typer.testing import CliRunner

    from watch_skill.index import embeddings, list_videos
    from watch_skill.surfaces.cli.main import app

    monkeypatch.setattr(embeddings, "embed_texts", lambda texts, model_name=None: None)
    wav = _wav(tmp_path / "voice memo.wav")
    result = CliRunner().invoke(app, ["watch", str(wav), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload) == {"video_id", "summary", "warnings"}
    assert payload["video_id"] in {row["id"] for row in list_videos()}
    assert payload["summary"]["media_kind"] == "audio"
