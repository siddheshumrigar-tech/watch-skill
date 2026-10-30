"""Transcript language: detected, forced, or read from the captions.

faster-whisper always reports the language it detected and how sure it was.
That was thrown away, so an agent could not tell an English transcript from
a Spanish one, and nobody could force the language when detection guessed
wrong on a short or noisy clip.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from watch_skill.errors import TranscriptionError
from watch_skill.transcribe.types import Segment, Transcript, Word


class _RawSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text = start, end, text
        self.words: list = []


def _install_whisper(monkeypatch: pytest.MonkeyPatch, language: str = "de",
                     probability: float = 0.93) -> list[dict]:
    """A fake faster_whisper whose info reports ``language``; records kwargs."""
    calls: list[dict] = []

    class FakeModel:
        def __init__(self, size: str, device: str, compute_type: str) -> None:
            pass

        def transcribe(self, path: str, **kwargs: object):
            calls.append(kwargs)
            info = types.SimpleNamespace(language=language, language_probability=probability)
            return [_RawSegment(0.0, 2.0, "guten Tag")], info

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setenv("WATCHSKILL_WHISPER_DEVICE", "cpu")
    return calls


def test_local_whisper_keeps_the_detected_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (A4): ``model.transcribe`` returned the language as ``_info``
    and it was discarded."""
    from watch_skill.transcribe.local import transcribe_local

    _install_whisper(monkeypatch)
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"")
    transcript = transcribe_local(audio, model_size="tiny")
    assert transcript.language == "de"
    assert transcript.language_probability == pytest.approx(0.93)
    assert transcript.language_source == "detected"


def test_a_forced_language_reaches_whisper_through_the_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from watch_skill.transcribe import audio as audio_mod
    from watch_skill.transcribe.ladder import get_transcript

    calls = _install_whisper(monkeypatch, language="fr", probability=1.0)

    def fake_extract(video_path, dest, start_seconds=None, end_seconds=None):
        Path(dest).write_bytes(b"audio")
        return Path(dest)

    monkeypatch.setattr(audio_mod, "extract_audio", fake_extract)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    transcript = get_transcript(
        video, tmp_path, allow_local=True, allow_cloud=False, whisper_model="tiny",
        language="French",
    )
    assert calls[0]["language"] == "fr"
    assert transcript.language == "fr"
    assert transcript.language_source == "forced"
    assert transcript.language_probability is None


def test_the_language_setting_is_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from watch_skill.config import reset_settings
    from watch_skill.transcribe import audio as audio_mod
    from watch_skill.transcribe.ladder import get_transcript

    monkeypatch.setenv("WATCHSKILL_WHISPER_LANGUAGE", "es")
    reset_settings()
    calls = _install_whisper(monkeypatch)
    monkeypatch.setattr(
        audio_mod, "extract_audio",
        lambda video_path, dest, **kw: (Path(dest).write_bytes(b"a"), Path(dest))[1],
    )
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    get_transcript(video, tmp_path, allow_local=True, allow_cloud=False, whisper_model="tiny")
    assert calls[0]["language"] == "es"


def test_a_bad_language_fails_before_anything_is_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import watch_skill.watch as watch_mod

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("acquire ran before the language was validated")

    monkeypatch.setattr(watch_mod, "acquire", refuse)
    monkeypatch.setattr(watch_mod, "fetch_captions_only", refuse)
    with pytest.raises(TranscriptionError) as excinfo:
        watch_mod.watch("https://example.com/v.mp4", language="klingon", out_dir=tmp_path / "w")
    assert excinfo.value.code == "transcribe.bad_language"
    assert excinfo.value.fix


@pytest.mark.parametrize(
    ("value", "expected"),
    [("en", "en"), ("EN", "en"), ("english", "en"), ("en-US", "en"), ("pt_BR", "pt"),
     ("Spanish", "es"), ("", None), (None, None), ("auto", None)],
)
def test_validate_language_normalizes(value: str | None, expected: str | None) -> None:
    from watch_skill.transcribe.languages import validate_language

    assert validate_language(value) == expected


def test_normalize_language_returns_none_for_unknown() -> None:
    from watch_skill.transcribe.languages import normalize_language

    assert normalize_language("klingon") is None
    assert normalize_language("german") == "de"


def _language_transcript() -> Transcript:
    return Transcript(
        segments=[Segment(0.0, 2.0, "hola"), Segment(5.0, 7.0, "adios")],
        source="whisper-local (tiny)",
        language="es",
        language_probability=0.97,
        language_source="detected",
        warnings=[{"code": "x.y", "message": "m", "fix": "f"}],
    )


def test_offset_and_filter_range_keep_the_language() -> None:
    for derived in (
        _language_transcript().offset(10.0),
        _language_transcript().filter_range(0.0, 3.0),
    ):
        assert derived.language == "es"
        assert derived.language_probability == 0.97
        assert derived.language_source == "detected"
        assert derived.warnings == [{"code": "x.y", "message": "m", "fix": "f"}]


def test_assign_speakers_keeps_the_language() -> None:
    from watch_skill.transcribe.diarize import SpeakerTurn, assign_speakers

    labeled = assign_speakers(_language_transcript(), [SpeakerTurn(0.0, 9.0, "SPEAKER_00")])
    assert (labeled.language, labeled.language_source) == ("es", "detected")


def test_caption_language_comes_from_the_filename(tmp_path: Path) -> None:
    from watch_skill.transcribe.ladder import get_transcript

    vtt = tmp_path / "talk.es-419.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhola a todos\n", encoding="utf-8"
    )
    transcript = get_transcript(None, tmp_path, subtitle_path=vtt)
    assert transcript.source == "captions"
    assert transcript.language == "es"
    assert transcript.language_source == "captions"
    assert transcript.language_probability is None


def test_failed_rungs_leave_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An empty transcript has to say why, in a shape an agent can act on."""
    from watch_skill.transcribe import local
    from watch_skill.transcribe.ladder import get_transcript

    def broken(*args: object, **kwargs: object):
        raise TranscriptionError(
            "faster-whisper is not installed",
            code="transcribe.missing_dependency", fix="pip install it",
        )

    monkeypatch.setattr("watch_skill.transcribe.audio.extract_audio", broken)
    monkeypatch.setattr(local, "transcribe_local", broken)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    transcript = get_transcript(video, tmp_path, allow_local=True, allow_cloud=False)
    assert transcript.source == "none"
    assert transcript.warnings == [
        {"code": "transcribe.missing_dependency",
         "message": "faster-whisper is not installed", "fix": "pip install it"}
    ]


def test_to_public_has_every_key() -> None:
    """D1: agents read one stable shape; null and [] instead of missing keys."""
    transcript = Transcript(
        segments=[
            Segment(0.0, 1.0, "hi", words=[Word(0.0, 1.0, "hi")]),
            Segment(1.0, 2.0, "there", speaker="SPEAKER_00"),
        ],
        source="captions",
    )
    public = transcript.to_public()
    assert set(public) >= {"source", "language", "language_probability",
                           "language_source", "segments"}
    assert public["language"] is None
    first, second = public["segments"]
    assert first == {
        "start": 0.0, "end": 1.0, "text": "hi", "speaker": None,
        "words": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
    }
    assert second["speaker"] == "SPEAKER_00" and second["words"] == []
    # the index format is unchanged: to_dict still omits null keys
    assert "speaker" not in transcript.segments[0].to_dict()
