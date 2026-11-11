"""Verification checks over the indexed transcript (D5).

`transcript_contains`, `transcript_absent` and `speaker_said` read the index
through a read-only handle named by `CheckContext.index_path`, honour a
start/end window, and say INCONCLUSIVE whenever the evidence is missing.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from watch_skill.index import embeddings as emb
from watch_skill.transcribe.types import Segment, Transcript
from watch_skill.verify import (
    Assurance,
    Check,
    CheckContext,
    CheckStatus,
    draft_contract,
    verify_run,
)
from watch_skill.verify.checks import run_check

SEGMENTS = [
    Segment(0.0, 4.0, "we should deploy the release today", speaker="SPEAKER_00"),
    Segment(4.0, 8.0, "no, the deploy waits until the tests pass", speaker="SPEAKER_01"),
    Segment(8.0, 12.0, "fine, the release can wait", speaker="SPEAKER_00"),
]


@pytest.fixture(autouse=True)
def _keyword_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(emb, "embed_texts", lambda texts, model_name=None: None)


def _index(tmp_path: Path, segments: list[Segment], source: str = "standup.mp4") -> str:
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind
    from watch_skill.index import index_watch_result
    from watch_skill.perceive.types import VideoMetadata
    from watch_skill.watch import WatchResult

    result = WatchResult(
        acquisition=AcquireResult(source=source, kind=SourceKind.LOCAL_FILE,
                                  video_path=None, acquirer="local"),
        metadata=VideoMetadata(duration_seconds=12.0, width=None, height=None, fps=None,
                               codec=None, has_audio=True),
        perception=None,
        transcript=Transcript(segments=list(segments), source="whisper-local (tiny)"),
        work_dir=tmp_path,
    )
    return index_watch_result(result, describe_scenes=False)


def _ctx(tmp_path: Path, index_path: str | None = "default") -> CheckContext:
    from watch_skill.config import get_settings

    path = str(get_settings().index_path) if index_path == "default" else index_path
    return CheckContext(working_dir=str(tmp_path), allowed_roots=[str(tmp_path)],
                        index_path=path)


def _run(tmp_path: Path, kind: str, ctx: CheckContext | None = None, **params):
    return run_check(Check(id="t", type=kind, params=params), ctx or _ctx(tmp_path))


def test_contains_passes_fails_and_reports_timestamps(tmp_path: Path) -> None:
    video_id = _index(tmp_path, SEGMENTS)
    hit = _run(tmp_path, "transcript_contains", video=video_id, phrase="tests pass")
    assert hit.status is CheckStatus.PASS
    assert hit.observed["matches"] == [{"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01"}]
    assert "C:" not in str(hit.observed) and "/" not in str(hit.observed)
    miss = _run(tmp_path, "transcript_contains", video="standup.mp4", phrase="rollback")
    assert miss.status is CheckStatus.FAIL


def test_the_window_bounds_what_counts(tmp_path: Path) -> None:
    video_id = _index(tmp_path, SEGMENTS)
    early = _run(tmp_path, "transcript_contains", video=video_id, phrase="can wait",
                 start="0:00", end="0:04")
    assert early.status is CheckStatus.FAIL
    late = _run(tmp_path, "transcript_contains", video=video_id, phrase="can wait", start=8)
    assert late.status is CheckStatus.PASS
    empty = _run(tmp_path, "transcript_contains", video=video_id, phrase="x",
                 start=100, end=120)
    assert empty.status is CheckStatus.INCONCLUSIVE


def test_absent_passes_and_fails(tmp_path: Path) -> None:
    video_id = _index(tmp_path, SEGMENTS)
    assert _run(tmp_path, "transcript_absent", video=video_id,
                phrase="rollback").status is CheckStatus.PASS
    assert _run(tmp_path, "transcript_absent", video=video_id,
                phrase="DEPLOY").status is CheckStatus.FAIL


def test_speaker_said_checks_who(tmp_path: Path) -> None:
    video_id = _index(tmp_path, SEGMENTS)
    yes = _run(tmp_path, "speaker_said", video=video_id, phrase="release", speaker="speaker_00")
    assert yes.status is CheckStatus.PASS
    no = _run(tmp_path, "speaker_said", video=video_id, phrase="tests pass", speaker="SPEAKER_00")
    assert no.status is CheckStatus.FAIL
    assert no.observed["said_by_others"] == 1
    assert no.observed["speakers_in_window"] == ["SPEAKER_00", "SPEAKER_01"]


def test_speaker_said_without_labels_is_inconclusive(tmp_path: Path) -> None:
    plain = [Segment(s.start, s.end, s.text) for s in SEGMENTS]
    video_id = _index(tmp_path, plain, source="plain.mp4")
    result = _run(tmp_path, "speaker_said", video=video_id, phrase="release", speaker="SPEAKER_00")
    assert result.status is CheckStatus.INCONCLUSIVE
    assert "--diarize" in result.summary


def test_missing_evidence_is_inconclusive(tmp_path: Path) -> None:
    _index(tmp_path, SEGMENTS)
    no_index = _run(tmp_path, "transcript_contains", _ctx(tmp_path, None),
                    video="standup.mp4", phrase="deploy")
    assert no_index.status is CheckStatus.INCONCLUSIVE
    unknown = _run(tmp_path, "transcript_absent", video="other.mp4", phrase="deploy")
    assert unknown.status is CheckStatus.INCONCLUSIVE, "absence in an unindexed video is not proof"


def test_a_pre_v11_index_without_speakers_is_read(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE videos (id TEXT PRIMARY KEY, source TEXT);
        CREATE TABLE segments (id INTEGER PRIMARY KEY, video_id TEXT,
                               start REAL, end REAL, text TEXT);
        INSERT INTO videos VALUES ('v1', 'old.mp4');
        INSERT INTO segments (video_id, start, end, text) VALUES ('v1', 0, 2, 'hello there');
    """)
    conn.commit()
    conn.close()
    ctx = _ctx(tmp_path, str(path))
    assert _run(tmp_path, "transcript_contains", ctx, video="old.mp4",
                phrase="hello").status is CheckStatus.PASS
    assert _run(tmp_path, "speaker_said", ctx, video="old.mp4", phrase="hello",
                speaker="SPEAKER_00").status is CheckStatus.INCONCLUSIVE


def test_the_check_never_writes_to_the_index(tmp_path: Path) -> None:
    from watch_skill.config import get_settings

    _index(tmp_path, SEGMENTS)
    before = get_settings().index_path.read_bytes()
    _run(tmp_path, "transcript_contains", video="standup.mp4", phrase="deploy")
    assert get_settings().index_path.read_bytes() == before


def test_a_backwards_window_is_malformed(tmp_path: Path) -> None:
    video_id = _index(tmp_path, SEGMENTS)
    result = _run(tmp_path, "transcript_contains", video=video_id, phrase="x", start=8, end=4)
    assert result.status is CheckStatus.ERROR
    assert result.error["code"] == "verify.check_malformed"


def test_the_isolated_verifier_reads_the_index(tmp_path: Path) -> None:
    """The child gets no WATCHSKILL_DATA_DIR, so this proves index_path."""
    video_id = _index(tmp_path, SEGMENTS)
    contract = draft_contract(
        "t", [{"id": "said", "type": "speaker_said",
               "params": {"video": video_id, "phrase": "tests pass", "speaker": "SPEAKER_01"}}],
        required_assurance=Assurance.ISOLATED_LOCAL,
    ).freeze()
    bundle, _ = verify_run(contract, working_dir=tmp_path, isolated=True)
    assert [r.status for r in bundle.check_results] == [CheckStatus.PASS], bundle.check_results
    assert bundle.verdict == "pass"
