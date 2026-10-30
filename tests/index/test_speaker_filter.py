"""Speakers reach the index, and ask/search/moment can filter by them (A3).

Diarization labelled segments in memory and the index threw the labels away:
`segments` had no speaker column, so "what did SPEAKER_01 say about X" had
nothing to filter on.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from watch_skill.errors import IndexError_
from watch_skill.index import embeddings as emb
from watch_skill.index.db import MIGRATIONS, connect, schema_version
from watch_skill.transcribe.types import Segment, Transcript

SEGMENTS = [
    Segment(0.0, 4.0, "we should deploy the release today", speaker="SPEAKER_00"),
    Segment(4.0, 8.0, "no, the deploy waits until the tests pass", speaker="SPEAKER_01"),
    Segment(8.0, 12.0, "fine, the release can wait", speaker="SPEAKER_00"),
]


@pytest.fixture(autouse=True)
def _keyword_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """No embedding model: FTS alone is enough to prove the filter."""
    monkeypatch.setattr(emb, "embed_texts", lambda texts, model_name=None: None)


def _result(tmp_path: Path, segments: list[Segment], source: str = "standup.mp4"):
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind
    from watch_skill.perceive.types import VideoMetadata
    from watch_skill.watch import WatchResult

    return WatchResult(
        acquisition=AcquireResult(
            source=source, kind=SourceKind.LOCAL_FILE, video_path=None, acquirer="local"
        ),
        metadata=VideoMetadata(
            duration_seconds=12.0, width=None, height=None, fps=None, codec=None,
            has_audio=True,
        ),
        perception=None,
        transcript=Transcript(
            segments=list(segments), source="whisper-local (tiny)",
            language="es", language_probability=0.97, language_source="detected",
        ),
        work_dir=tmp_path,
    )


def _index(tmp_path: Path, segments: list[Segment] = SEGMENTS, source: str = "standup.mp4") -> str:
    from watch_skill.index import index_watch_result

    return index_watch_result(_result(tmp_path, segments, source), describe_scenes=False)


def test_v10_to_v11_migration_preserves_rows(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    for version, migration in enumerate(MIGRATIONS[:10], start=1):
        with conn:
            if callable(migration):
                migration(conn)
            else:
                conn.executescript(migration)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
    with conn:
        conn.execute(
            "INSERT INTO videos (id, source, title, duration_seconds, frames_dir) "
            "VALUES ('aaaaaaaaaaaaaaaa', 'C:/clips/demo.mp4', 'Demo', 12.0, '/f')"
        )
        conn.execute(
            "INSERT INTO segments (video_id, start, end, text) "
            "VALUES ('aaaaaaaaaaaaaaaa', 1.0, 3.0, 'kept')"
        )
    assert schema_version(conn) == 10
    conn.close()

    conn = connect(path)
    try:
        assert schema_version(conn) == len(MIGRATIONS)
        seg = conn.execute("SELECT text, speaker FROM segments").fetchone()
        assert (seg["text"], seg["speaker"]) == ("kept", None)
        video = conn.execute(
            "SELECT title, transcript_language, transcript_language_probability FROM videos"
        ).fetchone()
        assert tuple(video) == ("Demo", None, None)
        indexes = {r["name"] for r in conn.execute("PRAGMA index_list(segments)")}
        assert "idx_segments_speaker" in indexes
    finally:
        conn.close()


def test_speaker_and_language_round_trip_through_the_index(tmp_path: Path) -> None:
    video_id = _index(tmp_path)
    conn = connect()
    try:
        speakers = [
            r["speaker"] for r in conn.execute(
                "SELECT speaker FROM segments WHERE video_id = ? ORDER BY start", (video_id,)
            )
        ]
        video = conn.execute(
            "SELECT transcript_language, transcript_language_probability FROM videos "
            "WHERE id = ?", (video_id,),
        ).fetchone()
    finally:
        conn.close()
    assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert tuple(video) == ("es", 0.97)


def test_hybrid_search_filters_by_speaker(tmp_path: Path) -> None:
    from watch_skill.index.retrieval import hybrid_search

    video_id = _index(tmp_path)
    everyone = hybrid_search("deploy", video_id=video_id)
    assert {h.speaker for h in everyone} == {"SPEAKER_00", "SPEAKER_01"}

    only = hybrid_search("deploy", video_id=video_id, speaker="speaker_01")
    assert [h.text for h in only] == ["no, the deploy waits until the tests pass"]
    assert only[0].speaker == "SPEAKER_01"


def test_search_videos_filters_by_speaker(tmp_path: Path) -> None:
    from watch_skill.index import search_videos

    _index(tmp_path)
    groups = search_videos("release", speaker="SPEAKER_00")
    texts = [h["text"] for g in groups for h in g["hits"]]
    assert texts and all(h["speaker"] == "SPEAKER_00" for g in groups for h in g["hits"])


def test_get_moment_carries_and_filters_speaker(tmp_path: Path) -> None:
    from watch_skill.index import get_moment

    video_id = _index(tmp_path)
    moment = get_moment(video_id, 6.0, window=12.0)
    assert [s["speaker"] for s in moment.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]

    filtered = get_moment(video_id, 6.0, window=12.0, speaker="SPEAKER_01")
    assert [s["text"] for s in filtered.segments] == ["no, the deploy waits until the tests pass"]


def test_moment_segments_always_have_a_speaker_key(tmp_path: Path) -> None:
    from watch_skill.index import get_moment

    video_id = _index(tmp_path, [Segment(0.0, 4.0, "plain captions")], source="plain.mp4")
    assert get_moment(video_id, 1.0).segments[0]["speaker"] is None


def test_unknown_speaker_names_the_known_ones(tmp_path: Path) -> None:
    from watch_skill.index import ask_video

    video_id = _index(tmp_path)
    with pytest.raises(IndexError_) as info:
        ask_video(video_id, "deploy", speaker="SPEAKER_07")
    assert info.value.code == "index.unknown_speaker"
    assert info.value.details["known_speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert "SPEAKER_00" in info.value.fix


def test_a_video_without_speakers_says_how_to_get_them(tmp_path: Path) -> None:
    from watch_skill.index import get_moment

    video_id = _index(tmp_path, [Segment(0.0, 4.0, "plain captions")], source="plain.mp4")
    with pytest.raises(IndexError_) as info:
        get_moment(video_id, 1.0, speaker="SPEAKER_00")
    assert info.value.code == "index.not_diarized"
    assert "--diarize" in info.value.fix


def test_a_speaker_filter_bypasses_the_answer_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4: a filtered answer must never be served for an unfiltered question."""
    from watch_skill.answer import cache, engine

    video_id = _index(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cache, "lookup", lambda *a, **k: calls.append("lookup"))
    monkeypatch.setattr(cache, "put", lambda *a, **k: calls.append("put"))
    for name in ("dense_resample", "zoom_crops_reocr"):
        monkeypatch.setattr(engine, name, lambda video, hits, deadline: ([], 0, False))

    answer = engine.answer_question(video_id, "deploy", speaker="SPEAKER_01", verify=False)
    assert calls == []
    assert answer.evidence and {e.speaker for e in answer.evidence} == {"SPEAKER_01"}

    engine.answer_question(video_id, "deploy", verify=False)
    assert calls == ["lookup", "put"], "unfiltered asks still use the cache"


def test_mcp_tools_take_a_speaker(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp", reason="mcp extra not installed")
    from fastmcp import Client

    from watch_skill.surfaces.mcp.server import mcp

    _index(tmp_path)

    async def _run():
        async with Client(mcp) as client:
            tools = {t.name: t for t in await client.list_tools()}
            found = await client.call_tool(
                "search_videos", {"query": "deploy", "speaker": "SPEAKER_01"}
            )
            listed = await client.call_tool("list_videos", {})
            return tools, found, listed

    tools, found, listed = asyncio.run(_run())
    for name in ("ask_video", "search_videos", "get_moment"):
        assert "speaker" in tools[name].inputSchema["properties"], name
    assert "SPEAKER_01: no, the deploy waits" in found.content[0].text
    assert "language: es" in listed.content[0].text


def test_rest_and_cli_take_a_speaker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi", reason="api extra not installed")
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from watch_skill.surfaces.api import create_app
    from watch_skill.surfaces.cli.main import app

    video_id = _index(tmp_path)
    client = TestClient(create_app())
    search = client.get("/v1/search", params={"q": "deploy", "speaker": "SPEAKER_01"})
    assert search.status_code == 200, search.text
    assert {h["speaker"] for g in search.json() for h in g["hits"]} == {"SPEAKER_01"}

    moment = client.get(
        f"/v1/videos/{video_id}/moment",
        params={"timestamp": "6", "window": 12, "speaker": "SPEAKER_00"},
    )
    assert moment.status_code == 200, moment.text
    assert {s["speaker"] for s in moment.json()["segments"]} == {"SPEAKER_00"}

    ask = client.post("/v1/ask", json={"video": video_id, "question": "deploy",
                                       "speaker": "SPEAKER_07"})
    assert ask.status_code >= 400
    assert "index.unknown_speaker" in ask.text

    from watch_skill.answer import engine

    seen: dict = {}

    def fake_answer(video, question, **kwargs):
        seen.update(kwargs)
        raise IndexError_("stop", code="index.test_stop", fix="none")

    monkeypatch.setattr(engine, "answer_question", fake_answer)
    import watch_skill.answer as answer_pkg

    monkeypatch.setattr(answer_pkg, "answer_question", fake_answer)
    CliRunner().invoke(app, ["ask", video_id, "deploy", "--speaker", "SPEAKER_01"])
    assert seen["speaker"] == "SPEAKER_01"
    result = CliRunner().invoke(app, ["search", "deploy", "--speaker", "SPEAKER_01"])
    assert "SPEAKER_01: no, the deploy waits" in result.output
