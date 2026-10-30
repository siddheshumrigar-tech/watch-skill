"""Diarization: pure speaker-assignment logic + structured degradation."""
from __future__ import annotations

from pathlib import Path

import pytest

from watch_skill.transcribe.diarize import SpeakerTurn, assign_speakers
from watch_skill.transcribe.types import Segment, Transcript, Word


def _transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(0.0, 4.0, "hello and welcome"),
            Segment(4.0, 8.0, "thanks for having me"),
            Segment(20.0, 24.0, "closing remarks"),
        ],
        source="captions",
    )


def test_assign_speakers_dominant_overlap() -> None:
    turns = [
        SpeakerTurn(0.0, 4.5, "SPEAKER_00"),
        SpeakerTurn(4.5, 9.0, "SPEAKER_01"),
    ]
    labeled = assign_speakers(_transcript(), turns)
    assert labeled.segments[0].speaker == "SPEAKER_00"
    assert labeled.segments[1].speaker == "SPEAKER_01"
    assert labeled.segments[2].speaker is None  # no overlapping turn
    assert labeled.source == "captions"
    # original untouched (pure function)
    assert _transcript().segments[0].speaker is None


def test_assign_speakers_picks_larger_overlap() -> None:
    # segment 4..8 overlaps SPEAKER_00 for 1s and SPEAKER_01 for 3s
    turns = [
        SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
        SpeakerTurn(5.0, 30.0, "SPEAKER_01"),
    ]
    labeled = assign_speakers(_transcript(), turns)
    assert labeled.segments[1].speaker == "SPEAKER_01"


def test_formatted_includes_speaker_labels() -> None:
    labeled = assign_speakers(_transcript(), [SpeakerTurn(0.0, 9.0, "SPEAKER_00")])
    text = labeled.formatted()
    assert "[00:00] SPEAKER_00: hello and welcome" in text
    assert "[00:20] closing remarks" in text  # unlabeled stays bare


def test_segment_to_dict_roundtrips_speaker() -> None:
    seg = Segment(1.0, 2.0, "hi", speaker="SPEAKER_03")
    assert seg.to_dict()["speaker"] == "SPEAKER_03"
    assert "speaker" not in Segment(1.0, 2.0, "hi").to_dict()


def test_diarize_audio_uses_pyannote4_token_kwarg(tmp_path, monkeypatch) -> None:
    """Regression (pyannote.audio 3.x → 4.x): Pipeline.from_pretrained renamed
    use_auth_token= to token=. Fake the module and assert the new kwarg."""
    import sys
    import types

    from watch_skill.transcribe import diarize as mod

    seen: dict = {}

    class FakeTurn:
        start, end = 0.0, 2.0

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            return iter([(FakeTurn(), None, "SPEAKER_00")])

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            seen.update(kwargs, model=model)
            return cls()

        def __call__(self, path):
            return FakeAnnotation()

    fake_audio = types.ModuleType("pyannote.audio")
    fake_audio.Pipeline = FakePipeline
    fake_pkg = types.ModuleType("pyannote")
    fake_pkg.audio = fake_audio
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)

    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"fake")
    turns = mod.diarize_audio(audio, hf_token="hf_test")
    assert turns == [mod.SpeakerTurn(0.0, 2.0, "SPEAKER_00")]
    assert seen["token"] == "hf_test"
    assert "use_auth_token" not in seen


def test_diarize_transcript_degrades_without_backend(tmp_path) -> None:
    """No pyannote / no token on this machine -> transcript returned unchanged."""
    from watch_skill.transcribe.diarize import diarize_transcript

    original = _transcript()
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not a real video")
    result = diarize_transcript(original, video, tmp_path)
    assert [s.text for s in result.segments] == [s.text for s in original.segments]
    assert all(s.speaker is None for s in result.segments)


def _worded_transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(
                0.0, 8.0, "hello there yes hi",
                words=[
                    Word(0.0, 1.0, "hello"),
                    Word(1.0, 2.0, "there"),
                    Word(5.0, 6.0, "yes"),
                    Word(6.0, 7.5, "hi"),
                ],
            ),
        ],
        source="whisper-local (tiny)",
    )


def test_assign_speakers_keeps_words() -> None:
    """Regression (A2): diarizing used to rebuild every Segment without its
    words, so --diarize --word-timestamps silently lost word timings."""
    labeled = assign_speakers(_worded_transcript(), [SpeakerTurn(0.0, 9.0, "SPEAKER_00")])
    assert [w.text for w in labeled.segments[0].words] == ["hello", "there", "yes", "hi"]
    assert [(w.start, w.end) for w in labeled.segments[0].words] == [
        (0.0, 1.0), (1.0, 2.0), (5.0, 6.0), (6.0, 7.5),
    ]


def test_assign_speakers_labels_each_word_across_a_turn_change() -> None:
    """A segment spanning a speaker change: the segment keeps its dominant
    speaker and every word carries the speaker actually talking over it."""
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.0, 9.0, "SPEAKER_01")]
    seg = assign_speakers(_worded_transcript(), turns).segments[0]
    assert seg.speaker == "SPEAKER_01"  # 5 s of overlap beats 3 s
    assert [w.speaker for w in seg.words] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01",
    ]
    assert seg.to_dict()["words"][0]["speaker"] == "SPEAKER_00"


def test_word_to_dict_omits_unset_speaker() -> None:
    assert "speaker" not in Word(0.0, 1.0, "hi").to_dict()
    assert Word(0.0, 1.0, "hi", speaker="SPEAKER_02").to_dict()["speaker"] == "SPEAKER_02"


def test_offset_carries_word_speaker() -> None:
    shifted = Transcript(
        segments=[Segment(0.0, 1.0, "hi", words=[Word(0.0, 1.0, "hi", speaker="SPEAKER_00")])],
    ).offset(10.0)
    assert shifted.segments[0].words[0].speaker == "SPEAKER_00"
    assert shifted.segments[0].words[0].start == 10.0


def test_focused_diarization_lands_on_the_source_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (A1): focused mode diarized ``work_dir/audio.mp3`` -- the
    window the local rung extracted -- and laid its window-relative turns on
    source-timeline segments, so every label landed on the wrong segment (or
    none). Diarization must extract the same window itself and shift turns."""
    from watch_skill.transcribe import audio as audio_mod
    from watch_skill.transcribe import diarize as mod

    work_dir = tmp_path / "work dir"
    work_dir.mkdir()
    (work_dir / "audio.mp3").write_bytes(b"stale window audio")  # must not be reused
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    extracted: list[dict] = []

    def fake_extract(video_path, dest, start_seconds=None, end_seconds=None, **kw):
        extracted.append(
            {"dest": Path(dest), "start_seconds": start_seconds, "end_seconds": end_seconds}
        )
        Path(dest).write_bytes(b"window audio")
        return Path(dest)

    diarized: list[Path] = []

    def fake_diarize(audio_path, hf_token=None):
        diarized.append(Path(audio_path))
        return [SpeakerTurn(0.0, 4.0, "SPEAKER_00"), SpeakerTurn(4.0, 8.0, "SPEAKER_01")]

    monkeypatch.setattr(audio_mod, "extract_audio", fake_extract)
    monkeypatch.setattr(mod, "diarize_audio", fake_diarize)

    transcript = Transcript(
        segments=[Segment(100.0, 104.0, "first"), Segment(104.0, 108.0, "second")],
        source="whisper-local (tiny)",
    )
    labeled = mod.diarize_transcript(
        transcript, video, work_dir, start_seconds=100.0, end_seconds=110.0
    )
    assert [s.speaker for s in labeled.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert extracted == [
        {"dest": work_dir / "diarize-audio.mp3", "start_seconds": 100.0, "end_seconds": 110.0}
    ]
    assert diarized == [work_dir / "diarize-audio.mp3"]


def test_watch_forwards_the_window_to_diarization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """watch() must hand its window to diarize_transcript, or the turns can
    never be shifted back onto the source timeline."""
    import watch_skill.watch as watch_mod
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind
    from watch_skill.perceive.types import VideoMetadata
    from watch_skill.transcribe import diarize as diarize_mod

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        watch_mod, "acquire",
        lambda source, **kw: AcquireResult(
            source=source, kind=SourceKind.LOCAL_FILE, video_path=video, acquirer="local"
        ),
    )
    monkeypatch.setattr(
        watch_mod, "probe",
        lambda path: VideoMetadata(
            duration_seconds=200.0, width=None, height=None, fps=None, codec=None,
            has_audio=True,
        ),
    )
    monkeypatch.setattr(watch_mod, "discover_cues", lambda *a, **kw: None)
    monkeypatch.setattr(
        watch_mod, "get_transcript",
        lambda *a, **kw: Transcript(segments=[Segment(100.0, 104.0, "hi")], source="captions"),
    )
    seen: dict = {}

    def fake_diarize_transcript(transcript, video_path, work_dir, **kwargs):
        seen.update(kwargs)
        return transcript

    monkeypatch.setattr(diarize_mod, "diarize_transcript", fake_diarize_transcript)
    watch_mod.watch(
        str(video), start_seconds=100.0, end_seconds=110.0, transcript_only=True,
        diarize=True, out_dir=tmp_path / "work dir",
    )
    assert seen == {"start_seconds": 100.0, "end_seconds": 110.0}
