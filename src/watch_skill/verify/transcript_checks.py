"""Checks over an indexed transcript: what was said, by whom, and when.

They read the Watch Skill index through ``CheckContext.index_path`` and a
``mode=ro`` handle, never through ``index.db.connect()``, which migrates and
writes. The isolated verifier's child process does not inherit
``WATCHSKILL_DATA_DIR``, so the parent names the index file explicitly.

Missing evidence is never a verdict: no index, a video that is not indexed,
an empty window, or (for ``speaker_said``) a window with no speaker labels
all come back INCONCLUSIVE.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_MAX_MATCHES = 20
_FOREVER = 1e18


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _resolve_video(conn: sqlite3.Connection, video: str) -> str | None:
    """The current video id for an id or a source string."""
    cols = _columns(conn, "videos")
    order = []
    if "superseded_at" in cols:
        order.append("superseded_at IS NOT NULL")
    if "last_analyzed_at" in cols:
        order.append("last_analyzed_at DESC")
    clause = f" ORDER BY {', '.join(order)}" if order else ""
    row = conn.execute(
        f"SELECT id FROM videos WHERE id = ? OR source = ?{clause} LIMIT 1",  # noqa: S608
        (video, video),
    ).fetchone()
    return row[0] if row else None


def read_transcript(index_path: str | None, video: str, start: float | None,
                    end: float | None) -> tuple[list[dict[str, Any]], str | None]:
    """Segments of ``video`` overlapping ``[start, end]``, or why there are none.

    Returns ``(segments, None)`` with ``{start, end, text, speaker}`` dicts, or
    ``([], reason)`` when the evidence is missing. Tolerates an index from
    before speaker labels were stored (schema < 11): speakers are then None.
    """
    from watch_skill.verify.checks import read_only_sqlite_uri  # noqa: PLC0415

    if not index_path or not Path(index_path).is_file():
        return [], "no Watch Skill index is available to this check"
    try:
        conn = sqlite3.connect(read_only_sqlite_uri(Path(index_path)), uri=True, timeout=10.0)
    except sqlite3.Error as exc:
        return [], f"the index could not be opened: {exc}"
    try:
        video_id = _resolve_video(conn, video)
        if video_id is None:
            return [], f"video {video!r} is not in the index"
        speaker = "speaker" if "speaker" in _columns(conn, "segments") else "NULL"
        rows = conn.execute(
            f"SELECT start, end, text, {speaker} FROM segments "  # noqa: S608
            "WHERE video_id = ? AND end > ? AND start < ? ORDER BY start",
            (video_id, start if start is not None else -_FOREVER,
             end if end is not None else _FOREVER),
        ).fetchall()
    except sqlite3.Error as exc:
        return [], f"the index could not be read: {exc}"
    finally:
        conn.close()
    if not rows:
        return [], "no transcript segments fall in the requested window"
    return [{"start": r[0], "end": r[1], "text": r[2], "speaker": r[3]} for r in rows], None


def _window(params: dict[str, Any]) -> tuple[float | None, float | None]:
    from watch_skill.perceive.budget import parse_time  # noqa: PLC0415

    start, end = parse_time(params.get("start")), parse_time(params.get("end"))
    if start is not None and end is not None and end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    return start, end


def _matching(segments: list[dict[str, Any]], phrase: str,
              threshold: float | None) -> list[dict[str, Any]]:
    from watch_skill.verify.checks import _text_occurrences  # noqa: PLC0415

    return [s for s in segments if _text_occurrences(phrase, [s["text"]], threshold)]


def _observed(hits: list[dict[str, Any]], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Timestamps and speakers only: no text dump, no paths."""
    return {
        "matches": [{"start": round(h["start"], 2), "end": round(h["end"], 2),
                     "speaker": h["speaker"]} for h in hits[:_MAX_MATCHES]],
        "match_count": len(hits),
        "segments_in_window": len(segments),
    }


def _gather(check: Any, ctx: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]],
                                          str | None]:
    """(phrase, segments, hits, inconclusive_reason) for a transcript check."""
    from watch_skill.verify.checks import _similarity_threshold  # noqa: PLC0415

    params = check.params
    video, phrase = str(params["video"]), str(params["phrase"])
    threshold = _similarity_threshold(params.get("min_similarity"))
    start, end = _window(params)
    segments, reason = read_transcript(ctx.index_path, video, start, end)
    if reason is not None:
        return phrase, [], [], reason
    return phrase, segments, _matching(segments, phrase, threshold), None


def transcript_contains(check: Any, ctx: Any) -> tuple[Any, Any, Any, str]:
    """PASS when the phrase was said in the window, FAIL when it was not."""
    from watch_skill.verify.checks import CheckStatus  # noqa: PLC0415

    phrase, segments, hits, reason = _gather(check, ctx)
    expected = f"said: {phrase}"
    if reason is not None:
        return CheckStatus.INCONCLUSIVE, expected, None, reason
    return (
        CheckStatus.PASS if hits else CheckStatus.FAIL, expected, _observed(hits, segments),
        f"{len(hits)} of {len(segments)} segment(s) contain {phrase!r}",
    )


def transcript_absent(check: Any, ctx: Any) -> tuple[Any, Any, Any, str]:
    """PASS when the phrase was not said in the window, FAIL when it was."""
    from watch_skill.verify.checks import CheckStatus  # noqa: PLC0415

    phrase, segments, hits, reason = _gather(check, ctx)
    expected = f"not said: {phrase}"
    if reason is not None:
        return CheckStatus.INCONCLUSIVE, expected, None, reason
    return (
        CheckStatus.FAIL if hits else CheckStatus.PASS, expected, _observed(hits, segments),
        f"{len(hits)} of {len(segments)} segment(s) contain {phrase!r}",
    )


def speaker_said(check: Any, ctx: Any) -> tuple[Any, Any, Any, str]:
    """PASS when ``speaker`` said the phrase in the window.

    INCONCLUSIVE when no segment in the window has a speaker label: without
    diarization nobody can be shown not to have said it.
    """
    from watch_skill.verify.checks import CheckStatus  # noqa: PLC0415

    speaker = str(check.params["speaker"])
    phrase, segments, hits, reason = _gather(check, ctx)
    expected = f"{speaker} said: {phrase}"
    if reason is None and not any(s["speaker"] for s in segments):
        reason = ("no segment in the window has a speaker label; re-watch with "
                  "--diarize to make speaker checks decidable")
    if reason is not None:
        return CheckStatus.INCONCLUSIVE, expected, None, reason
    theirs = [h for h in hits if str(h["speaker"] or "").casefold() == speaker.casefold()]
    observed = _observed(theirs, segments)
    observed["speakers_in_window"] = sorted({s["speaker"] for s in segments if s["speaker"]})
    observed["said_by_others"] = len(hits) - len(theirs)
    return (
        CheckStatus.PASS if theirs else CheckStatus.FAIL, expected, observed,
        f"{speaker} said {phrase!r} in {len(theirs)} segment(s); "
        f"{len(hits) - len(theirs)} other match(es) by other speakers",
    )


RUNNERS = {
    "transcript_contains": transcript_contains,
    "transcript_absent": transcript_absent,
    "speaker_said": speaker_said,
}
