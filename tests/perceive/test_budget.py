"""Frame budgets: reference-inherited tiers, 2 fps cap, time parsing."""
from __future__ import annotations

import pytest

from watch_skill.perceive.budget import focused_budget, format_time, full_budget, parse_time


def test_full_budget_tiers() -> None:
    assert full_budget(10)[1] == 12  # short floor
    assert full_budget(30)[1] == 30
    assert full_budget(45)[1] == 40
    assert full_budget(120)[1] == 60
    assert full_budget(400)[1] == 80
    assert full_budget(3600)[1] == 100  # cap


def test_full_budget_respects_explicit_cap() -> None:
    fps, target = full_budget(3600, max_frames=40)
    assert target == 40


@pytest.mark.parametrize(("duration", "frames"), [(3600, 100), (7200, 200), (20000, 300)])
def test_long_videos_get_a_budget_that_scales(duration: float, frames: int) -> None:
    """D8: one frame per 36 s past the default cap, up to frame_cap_max.

    A 2-hour talk used to get the same 100 frames as a 1-hour one: one frame
    every 72 seconds, which is not watching it.
    """
    assert full_budget(duration)[1] == frames


def test_an_explicit_cap_still_wins_on_a_long_video() -> None:
    assert full_budget(20000, max_frames=40)[1] == 40


def test_frame_cap_max_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.config import reset_settings

    monkeypatch.setenv("WATCHSKILL_FRAME_CAP_MAX", "150")
    reset_settings()
    assert full_budget(20000)[1] == 150


def _sparse_result(tmp_path, *, seconds_per_frame: float, sparse: bool, focused: bool = False):
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind
    from watch_skill.perceive.types import Frame, PerceptionResult, VideoMetadata
    from watch_skill.transcribe.types import Transcript
    from watch_skill.watch import WatchResult

    meta = VideoMetadata(duration_seconds=7200.0, width=640, height=360, fps=30.0,
                         codec="h264", has_audio=True)
    frames = [Frame(index=i, timestamp_seconds=i * seconds_per_frame, path=tmp_path / f"{i}.jpg",
                    scene_id=0, phash="0", reason="uniform") for i in range(3)]
    perception = PerceptionResult(source="talk.mp4", metadata=meta, frames=frames,
                                  seconds_per_frame=seconds_per_frame, sampling_sparse=sparse)
    return WatchResult(
        acquisition=AcquireResult(source="talk.mp4", kind=SourceKind.LOCAL_FILE,
                                  video_path=None, acquirer="local"),
        metadata=meta, perception=perception, transcript=Transcript(), work_dir=tmp_path,
        start_seconds=0.0 if focused else None,
    )


def test_sparse_sampling_is_measured_and_reported(tmp_path) -> None:
    from watch_skill.report import render_report

    result = _sparse_result(tmp_path, seconds_per_frame=36.0, sparse=True)
    summary = result.to_summary()
    assert summary["frames"] == {"count": 3, "seconds_per_frame": 36.0, "sparse": True}
    assert result.perception.to_dict()["seconds_per_frame"] == 36.0
    assert result.perception.to_dict()["sampling_sparse"] is True
    assert "one frame every 36s" in render_report(result)


def test_dense_sampling_carries_no_sparse_note(tmp_path) -> None:
    from watch_skill.report import render_report

    result = _sparse_result(tmp_path, seconds_per_frame=5.0, sparse=False)
    assert "frame coverage is sparse" not in render_report(result)


def test_watch_adds_the_sparse_sampling_warning(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import watch_skill.watch as watch_mod

    video = tmp_path / "two hour talk.mp4"
    video.write_bytes(b"x")
    staged = _sparse_result(tmp_path, seconds_per_frame=36.0, sparse=True)
    from watch_skill.acquire import AcquireResult
    from watch_skill.acquire.sources import SourceKind

    monkeypatch.setattr(watch_mod, "acquire", lambda *a, **k: AcquireResult(
        source=str(video), kind=SourceKind.LOCAL_FILE, video_path=video, acquirer="local"))
    monkeypatch.setattr(watch_mod, "probe", lambda p: staged.metadata)
    monkeypatch.setattr(watch_mod, "discover_cues", lambda *a, **k: None)
    monkeypatch.setattr(watch_mod, "perceive", lambda *a, **k: staged.perception)
    monkeypatch.setattr(watch_mod, "get_transcript", lambda *a, **k: staged.transcript)

    result = watch_mod.watch(str(video), out_dir=tmp_path / "w")
    warning = next(w for w in result.warnings if w["code"] == "perceive.sparse_sampling")
    assert "WATCHSKILL_FRAME_CAP_MAX" in warning["fix"]


def test_fps_never_exceeds_two() -> None:
    fps, _ = full_budget(1)
    assert fps <= 2.0
    fps, _ = focused_budget(1)
    assert fps <= 2.0


def test_focused_budget_denser_than_full() -> None:
    for duration in (5, 15, 30, 60):
        assert focused_budget(duration)[1] >= full_budget(duration)[1]


def test_focused_tiers() -> None:
    assert focused_budget(5)[1] == 10  # 2 fps cap: 5s * 2 = 10
    assert focused_budget(30)[1] == 60
    assert focused_budget(60)[1] == 80
    assert focused_budget(300)[1] == 100


@pytest.mark.parametrize(
    ("value", "expected"),
    [("90", 90.0), ("1:30", 90.0), ("01:02:03", 3723.0), ("2:15.5", 135.5), (75, 75.0), (None, None), ("", None)],
)
def test_parse_time(value, expected) -> None:
    assert parse_time(value) == expected


def test_parse_time_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_time("abc")


def test_format_time() -> None:
    assert format_time(75) == "01:15"
    assert format_time(3723) == "1:02:03"
    assert format_time(0) == "00:00"
