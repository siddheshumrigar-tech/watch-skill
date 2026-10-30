"""Cloud STT: word timestamps and the reported language. Mocked HTTP only.

Whisper's cloud APIs return word timings only when asked with
``timestamp_granularities[]``, and report the language as an English name
(``"english"``). Neither was asked for or kept, so ``--word-timestamps``
silently did nothing on the cloud rung and the language was lost.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from watch_skill.config import reset_settings
from watch_skill.transcribe import cloud
from watch_skill.transcribe.types import Segment, Word

_RESPONSE = {
    "language": "english",
    "text": "hello there general kenobi",
    "segments": [
        {"start": 0.0, "end": 2.0, "text": " hello there"},
        {"start": 2.0, "end": 4.5, "text": " general kenobi"},
    ],
    "words": [
        {"word": "hello", "start": 0.1, "end": 0.6},
        {"word": "there", "start": 0.7, "end": 1.2},
        {"word": "general", "start": 2.1, "end": 2.9},
        {"word": "kenobi", "start": 3.0, "end": 3.884},
    ],
}


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture()
def posted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    """Cloud STT opted in, ffmpeg and HTTP faked; returns what was posted."""
    monkeypatch.setenv("WATCHSKILL_CLOUD_STT_ENABLED", "1")
    monkeypatch.setenv("WATCHSKILL_GROQ_API_KEY", "gsk-test")
    reset_settings()
    state: dict = {"payload": dict(_RESPONSE), "calls": []}

    def fake_extract(video_path, dest, start_seconds=None, end_seconds=None):
        Path(dest).write_bytes(b"small audio")
        return Path(dest)

    def fake_post(endpoint, **kwargs):
        state["calls"].append({k: v for k, v in kwargs.items() if k != "files"})
        return _Response(state["payload"])

    monkeypatch.setattr(cloud.audio_mod, "extract_audio", fake_extract)
    monkeypatch.setattr(cloud.httpx, "post", fake_post)
    return state


def test_word_timestamps_ask_for_both_granularities(posted: dict, tmp_path: Path) -> None:
    """Regression (A5): the request never named a granularity, so the API
    returned segments only."""
    cloud.transcribe_cloud(tmp_path / "v.mp4", tmp_path, word_timestamps=True)
    data = posted["calls"][0]["data"]
    assert data["timestamp_granularities[]"] == ["word", "segment"]
    assert data["response_format"] == "verbose_json"


def test_without_word_timestamps_the_request_is_unchanged(posted: dict, tmp_path: Path) -> None:
    cloud.transcribe_cloud(tmp_path / "v.mp4", tmp_path)
    data = posted["calls"][0]["data"]
    assert "timestamp_granularities[]" not in data
    assert "language" not in data


def test_returned_words_land_on_their_segments(posted: dict, tmp_path: Path) -> None:
    transcript = cloud.transcribe_cloud(tmp_path / "v.mp4", tmp_path, word_timestamps=True)
    first, second = transcript.segments
    assert [w.text for w in first.words] == ["hello", "there"]
    assert [w.text for w in second.words] == ["general", "kenobi"]
    assert second.words[1].end == 3.88  # rounded to 2 dp like segments
    assert transcript.warnings == []


def test_the_reported_language_is_normalized(posted: dict, tmp_path: Path) -> None:
    transcript = cloud.transcribe_cloud(tmp_path / "v.mp4", tmp_path)
    assert transcript.language == "en"
    assert transcript.language_source == "detected"
    assert transcript.language_probability is None  # verbose_json carries none


def test_a_forced_language_is_sent_and_reported(posted: dict, tmp_path: Path) -> None:
    transcript = cloud.transcribe_cloud(tmp_path / "v.mp4", tmp_path, language="fr")
    assert posted["calls"][0]["data"]["language"] == "fr"
    assert (transcript.language, transcript.language_source) == ("fr", "forced")


def test_missing_words_leave_a_warning(posted: dict, tmp_path: Path) -> None:
    posted["payload"] = {k: v for k, v in _RESPONSE.items() if k != "words"}
    transcript = cloud.transcribe_cloud(tmp_path / "v.mp4", tmp_path, word_timestamps=True)
    assert all(not s.words for s in transcript.segments)
    (warning,) = transcript.warnings
    assert warning["code"] == "transcribe.cloud_no_word_timestamps"
    assert warning["message"] and warning["fix"]


def test_a_chunk_offset_moves_the_words() -> None:
    segments = cloud._segments_from_response(_RESPONSE, offset=30.0)
    assert segments[0].start == 30.0
    assert [(w.start, w.end) for w in segments[0].words] == [(30.1, 30.6), (30.7, 31.2)]
    assert segments[1].words[0].start == 32.1


def test_per_segment_words_are_used_when_present() -> None:
    data = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                          "words": [{"word": " hi", "start": 0.2, "end": 0.8}]}]}
    (segment,) = cloud._segments_from_response(data)
    assert segment.words == [Word(0.2, 0.8, "hi")]


def test_merge_overlapping_keeps_words() -> None:
    chunk_a = [Segment(0, 10, "first", words=[Word(1.0, 2.0, "first")])]
    chunk_b = [Segment(20, 30, "second", words=[Word(21.0, 22.0, "second")])]
    merged = cloud.merge_overlapping([chunk_a, chunk_b])
    assert [w.text for s in merged for w in s.words] == ["first", "second"]


def test_the_ladder_forwards_words_and_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from watch_skill.transcribe.ladder import get_transcript
    from watch_skill.transcribe.types import Transcript

    seen: dict = {}

    def fake_cloud(video_path, work_dir, preferred_backend=None, **kwargs):
        seen.update(kwargs)
        return Transcript(segments=[Segment(0.0, 1.0, "hi")], source="whisper-groq")

    monkeypatch.setattr(cloud, "transcribe_cloud", fake_cloud)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    get_transcript(video, tmp_path, allow_local=False, allow_cloud=True,
                   word_timestamps=True, language="de")
    assert seen == {"word_timestamps": True, "language": "de"}
