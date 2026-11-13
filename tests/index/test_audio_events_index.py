"""Sound events reach the index (v12) and come back through search and moment."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from watch_skill.index import embeddings as emb
from watch_skill.index.db import MIGRATIONS, connect, schema_version
from watch_skill.transcribe.audio_events import AudioEvent
from watch_skill.transcribe.types import Segment, Transcript

EVENTS = [
    AudioEvent(0.0, 3.0, "music", 0.91, "Music", "fake"),
    AudioEvent(12.0, 15.5, "applause", 0.84, "Applause", "fake"),
    AudioEvent(20.0, 22.0, "silence", 1.0, None, "rms"),
]


@pytest.fixture(autouse=True)
def _keyword_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emb, "embed_texts", lambda texts, model_name=None: None)


def _index(tmp_path: Path, events: list[AudioEvent] = EVENTS) -> str:
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind
    from watch_skill.index import index_watch_result
    from watch_skill.perceive.types import VideoMetadata
    from watch_skill.watch import WatchResult

    result = WatchResult(
        acquisition=AcquireResult(source="keynote.mp4", kind=SourceKind.LOCAL_FILE,
                                  video_path=None, acquirer="local"),
        metadata=VideoMetadata(duration_seconds=30.0, width=None, height=None, fps=None,
                               codec=None, has_audio=True),
        perception=None,
        transcript=Transcript(segments=[Segment(10.0, 14.0, "thank you all for coming")],
                              source="whisper-local (tiny)"),
        work_dir=tmp_path,
        audio_events=list(events),
    )
    return index_watch_result(result, describe_scenes=False)


def test_v11_to_v12_migration_preserves_rows(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    for version, migration in enumerate(MIGRATIONS[:11], start=1):
        with conn:
            if callable(migration):
                migration(conn)
            else:
                conn.executescript(migration)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    with conn:
        conn.execute("INSERT INTO videos (id, source, title, duration_seconds, frames_dir) "
                     "VALUES ('aaaaaaaaaaaaaaaa', 'C:/clips/demo.mp4', 'Demo', 12.0, '/f')")
        conn.execute("INSERT INTO segments (video_id, start, end, text, speaker) "
                     "VALUES ('aaaaaaaaaaaaaaaa', 1.0, 3.0, 'kept', 'SPEAKER_00')")
    assert schema_version(conn) == 11
    conn.close()

    conn = connect(path)
    try:
        assert schema_version(conn) == len(MIGRATIONS)
        assert conn.execute("SELECT text, speaker FROM segments").fetchone()[0] == "kept"
        assert conn.execute("SELECT COUNT(*) FROM audio_events").fetchone()[0] == 0
    finally:
        conn.close()


def test_events_round_trip_and_reindex_replaces_them(tmp_path: Path) -> None:
    video_id = _index(tmp_path)
    conn = connect()
    try:
        rows = conn.execute("SELECT start, end, label, confidence, raw_label, backend "
                            "FROM audio_events WHERE video_id = ? ORDER BY start",
                            (video_id,)).fetchall()
    finally:
        conn.close()
    assert [tuple(r) for r in rows][1] == (12.0, 15.5, "applause", 0.84, "Applause", "fake")
    _index(tmp_path, EVENTS[:1])
    conn = connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM audio_events").fetchone()[0] == 1
    finally:
        conn.close()


def test_search_finds_a_sound_by_label(tmp_path: Path) -> None:
    from watch_skill.index.retrieval import hybrid_search

    video_id = _index(tmp_path)
    hits = hybrid_search("applause", video_id=video_id)
    assert hits and hits[0].kind == "audio_event"
    assert hits[0].timestamp == 12.0
    assert "[sound: applause]" in hits[0].text


def test_search_videos_returns_the_sound(tmp_path: Path) -> None:
    from watch_skill.index.retrieval import search_videos

    _index(tmp_path)
    groups = search_videos("applause")
    kinds = [hit["kind"] for group in groups for hit in group["hits"]]
    assert "audio_event" in kinds


def test_get_moment_includes_overlapping_sounds(tmp_path: Path) -> None:
    from watch_skill.index.retrieval import get_moment

    video_id = _index(tmp_path)
    moment = get_moment(video_id, 13.0, window=4.0)
    assert [e["label"] for e in moment.audio_events] == ["applause"]
    assert set(moment.audio_events[0]) == {"start", "end", "label", "confidence", "raw_label"}
    assert get_moment(video_id, 6.0, window=2.0).audio_events == []


def test_agreement_stays_within_one_with_four_kinds() -> None:
    from watch_skill.answer.confidence import evidence_agreement
    from watch_skill.index.retrieval import Hit

    hits = [Hit(video_id="v", kind=k, ref_id=i, timestamp=10.0, text="x", score=0.9)
            for i, k in enumerate(("segment", "ocr", "scene", "audio_event"))]
    assert evidence_agreement(hits) == 1.0
