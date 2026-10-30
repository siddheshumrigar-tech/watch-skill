"""`language` reaches the engine the same way from every surface."""
from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

import pytest

from watch_skill.errors import TranscriptionError
from watch_skill.transcribe.types import Segment, Transcript


def _result(tmp_path: Path):
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind
    from watch_skill.perceive.types import VideoMetadata
    from watch_skill.watch import WatchResult

    return WatchResult(
        acquisition=AcquireResult(
            source="clip.mp4", kind=SourceKind.LOCAL_FILE, video_path=None, acquirer="local"
        ),
        metadata=VideoMetadata(
            duration_seconds=12.0, width=None, height=None, fps=None, codec=None,
            has_audio=True,
        ),
        perception=None,
        transcript=Transcript(
            segments=[Segment(0.0, 2.0, "hola")], source="whisper-local (tiny)",
            language="es", language_probability=0.97, language_source="detected",
        ),
        work_dir=tmp_path,
    )


def _record_watch(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub the engine: record its kwargs, then fail so nothing else runs."""
    import watch_skill.watch as watch_mod

    seen: dict = {}

    def fake_watch(source: str, **kwargs: object):
        seen.update(kwargs)
        raise TranscriptionError("stop", code="transcribe.test_stop", fix="none needed")

    monkeypatch.setattr(watch_mod, "watch", fake_watch)
    return seen


def test_report_names_the_language(tmp_path: Path) -> None:
    from watch_skill.report import render_report

    assert "_Source: whisper-local (tiny) · language: es (p=0.97, detected)._" in render_report(
        _result(tmp_path)
    )


def test_cli_watch_language(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from watch_skill.surfaces.cli.main import app

    seen = _record_watch(monkeypatch)
    CliRunner().invoke(app, ["watch", "x.mp4", "--language", "es"])
    assert seen["language"] == "es"


def test_mcp_watch_video_language(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastmcp", reason="mcp extra not installed")
    from fastmcp import Client

    from watch_skill.surfaces.mcp.server import mcp

    seen = _record_watch(monkeypatch)

    async def _run():
        async with Client(mcp) as client:
            return await client.call_tool("watch_video", {"source": "x.mp4", "language": "es"})

    result = asyncio.run(_run())
    assert seen["language"] == "es"
    assert "transcribe.test_stop" in result.content[0].text


def test_mcp_background_watch_carries_language(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastmcp", reason="mcp extra not installed")
    from fastmcp import Client

    from watch_skill import jobs
    from watch_skill.surfaces.mcp.server import mcp

    submitted: list[tuple] = []

    def fake_submit(kind, payload=None, *, idempotency_key=None, **kw):
        submitted.append((kind, payload, idempotency_key))
        return types.SimpleNamespace(job_id="job-1")

    monkeypatch.setattr(jobs, "submit_and_run", fake_submit)

    async def _run(args: dict):
        async with Client(mcp) as client:
            return await client.call_tool("watch_video", args)

    asyncio.run(_run({"source": "x.mp4", "background": True, "language": "es"}))
    asyncio.run(_run({"source": "x.mp4", "background": True}))
    (_, payload_es, key_es), (_, payload_auto, key_auto) = submitted
    assert payload_es["language"] == "es"
    assert payload_auto["language"] is None
    assert key_es != key_auto, "a forced language must not dedupe onto an auto-detect job"
    assert key_auto == "watch:x.mp4:None:None:None", "existing keys must not change"


def test_rest_watch_language(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi", reason="api extra not installed")
    from fastapi.testclient import TestClient

    import watch_skill.index as index_mod
    import watch_skill.watch as watch_mod
    from watch_skill.surfaces.api import create_app

    seen: dict = {}

    def fake_watch(source: str, **kwargs: object):
        seen.update(kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(watch_mod, "watch", fake_watch)
    monkeypatch.setattr(index_mod, "index_watch_result", lambda result: "vid0000000000001")
    response = TestClient(create_app()).post(
        "/v1/watch", json={"source": "clip.mp4", "language": "es"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert seen["language"] == "es"
    assert body["transcript_language"] == "es"
    assert body["transcript_language_probability"] == 0.97


def test_background_job_passes_language(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.jobs import handlers

    seen = _record_watch(monkeypatch)
    ctx = types.SimpleNamespace(
        payload={"source": "x.mp4", "language": "es"},
        checkpoint=lambda *a, **kw: None,
    )
    with pytest.raises(TranscriptionError):
        handlers._watch_job(ctx)
    assert seen["language"] == "es"
    json.dumps(ctx.payload)  # the payload stays JSON-serializable
