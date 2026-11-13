"""Read path: hybrid FTS5 + vector retrieval. Analyze once, ask forever."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from watch_skill.errors import IndexError_
from watch_skill.index import embeddings as emb
from watch_skill.index.db import connect
from watch_skill.index.store import get_video

_FTS_CANDIDATES = 24
_VECTOR_CANDIDATES = 24


@dataclass
class Hit:
    """One retrieved piece of evidence."""

    video_id: str
    kind: str  # segment | scene | ocr | audio_event
    ref_id: int
    timestamp: float | None
    text: str
    score: float
    # How many near-identical OCR observations this hit stands for, and the
    # span they covered. A caption that sits on screen across a dozen frames
    # is ONE thing the video showed, not a dozen independent witnesses — the
    # cluster is collapsed to a representative, and these say what it covered
    # so the evidence is summarizable without being re-derived.
    duplicate_count: int = 1
    duplicate_first: float | None = None
    duplicate_last: float | None = None
    # Diarized speaker of a transcript-segment hit; None for OCR/scene hits
    # and for videos that were never diarized.
    speaker: str | None = None
    # Reciprocal-rank fusion: what ORDERS hits. `score` above stays the
    # calibrated 0.45/0.55 weighted relevance the confidence model reads; RRF
    # values (~0.03) on that scale would break every confidence decision.
    rrf_score: float = 0.0
    fts_rank: int | None = None  # 1-based position in the keyword list
    vector_rank: int | None = None  # 1-based position in the vector list


@dataclass
class MomentContext:
    """Frames + text surrounding one moment of one video."""

    video_id: str
    timestamp: float
    window: float
    frames: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    ocr: list[dict[str, Any]] = field(default_factory=list)
    # Non-speech sounds overlapping the window: {start, end, label, confidence, raw_label}.
    audio_events: list[dict[str, Any]] = field(default_factory=list)


def _fts_query(text: str) -> str:
    """Sanitize free text into an OR-of-terms FTS5 MATCH over the normalized column.

    Terms are folded with the same normalization used at index time, so
    Arabic hamza/diacritic variants (and case) match reliably. A term whose
    normalization contains spaces (CJK runs are character-segmented) becomes
    an FTS5 phrase query, so the characters must appear adjacent — substring
    search over unspaced CJK text, including two-character queries.
    """
    import unicodedata

    from watch_skill.index.textnorm import normalize_for_search

    def keep(ch: str) -> bool:
        # combining marks (Devanagari matras, etc.) are part of words —
        # str.isalnum() alone would strip them and break the match
        return ch.isalnum() or ch == " " or unicodedata.category(ch).startswith("M")

    terms: list[str] = []
    for token in text.split():
        normalized = normalize_for_search(token)
        cleaned = "".join(ch for ch in normalized if keep(ch))
        cleaned = " ".join(cleaned.split())
        if cleaned:
            terms.append(cleaned)
    if not terms:
        return '""'
    return " OR ".join(f'text_norm:"{t}"' for t in terms)


def _speaker_clause(speaker: str | None, video_id: str | None) -> tuple[str, list[Any]]:
    """SQL that keeps only transcript segments spoken by ``speaker``.

    Pushed into both candidate queries, so the filter shapes the candidate
    pool instead of trimming a top-k that other speakers already filled.
    """
    if not speaker:
        return "", []
    sql = (
        " AND kind = 'segment' AND ref_id IN "
        "(SELECT id FROM segments WHERE speaker = ? COLLATE NOCASE"
    )
    params: list[Any] = [speaker]
    if video_id:
        sql += " AND video_id = ?"
        params.append(video_id)
    return sql + ")", params


def _fts_hits(
    conn: sqlite3.Connection, query: str, video_id: str | None, speaker: str | None = None
) -> list[Hit]:
    sql = (
        "SELECT video_id, kind, ref_id, timestamp, text, bm25(fts) AS rank "
        "FROM fts WHERE fts MATCH ?"
    )
    params: list[Any] = [_fts_query(query)]
    if video_id:
        sql += " AND video_id = ?"
        params.append(video_id)
    clause, extra = _speaker_clause(speaker, video_id)
    sql += clause
    params += extra
    sql += " ORDER BY rank LIMIT ?"
    params.append(_FTS_CANDIDATES)
    hits = []
    for row in conn.execute(sql, params).fetchall():
        # bm25 rank is lower-is-better and unbounded; squash to (0, 1]
        score = 1.0 / (1.0 + max(0.0, float(row["rank"])))
        hits.append(
            Hit(row["video_id"], row["kind"], row["ref_id"], row["timestamp"], row["text"], score)
        )
    return hits


def _vector_hits(
    conn: sqlite3.Connection, query: str, video_id: str | None, speaker: str | None = None
) -> list[Hit]:
    from watch_skill.index.db import get_meta

    # the query must embed with the same model that wrote the stored vectors
    model_name = get_meta(conn, "embedding_model") or emb.MODEL_NAME
    query_vecs = emb.embed_texts([query], model_name=model_name)
    if not query_vecs:
        return []
    query_vec = query_vecs[0]
    sql = "SELECT video_id, kind, ref_id, timestamp, text, vector, dim FROM embeddings WHERE 1=1"
    params: list[Any] = []
    if video_id:
        sql += " AND video_id = ?"
        params.append(video_id)
    clause, extra = _speaker_clause(speaker, video_id)
    sql += clause
    params += extra
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []
    scores = _batch_cosine(query_vec, rows)
    scored = [
        Hit(row["video_id"], row["kind"], row["ref_id"], row["timestamp"], row["text"], score)
        for row, score in zip(rows, scores, strict=False)
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:_VECTOR_CANDIDATES]


def _batch_cosine(query_vec: list[float], rows: list) -> list[float]:
    """Cosine of the query against every stored vector.

    numpy path: one matrix product (measured 45x faster than the pure-Python
    loop at 10k vectors — 122 ms vs 5.5 s on the dev machine). numpy ships
    with the index extra; the loop stays as a fallback for exotic installs.
    """
    try:
        import numpy as np  # noqa: PLC0415

        # Width is per row, not per index: vectors written before the float16
        # switch sit beside ones written after it, and an index that was only
        # ever appended to holds both. Reading a float16 blob as float32
        # returns numbers rather than an error, so guessing one width for the
        # batch would score silently wrong instead of failing.
        widths = {len(row["vector"]) // (row["dim"] or 1) for row in rows}
        if len(widths) == 1:
            # One buffer, one reinterpret. Falling into the per-row loop below
            # for a uniformly float16 index cost 650 ms per 100k scan against
            # 218 ms — the storage win is not worth paying for on every query.
            only = widths.pop()
            matrix = np.frombuffer(
                b"".join(row["vector"] for row in rows),
                dtype="<f2" if only == 2 else "<f4",
            ).reshape(len(rows), -1).astype(np.float32, copy=False)
        else:
            matrix = np.empty((len(rows), rows[0]["dim"]), dtype=np.float32)
            for i, row in enumerate(rows):
                dtype = "<f2" if len(row["vector"]) == row["dim"] * 2 else "<f4"
                matrix[i] = np.frombuffer(row["vector"], dtype=dtype)

        query = np.asarray(query_vec, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        norms[norms == 0] = 1.0
        return (matrix @ query / norms).tolist()
    except (ImportError, ValueError):  # ValueError: mixed-dim rows (corrupt index)
        return [
            emb.cosine_similarity(query_vec, emb.unpack_vector(row["vector"], row["dim"]))
            for row in rows
        ]


def _ocr_confidences(conn: sqlite3.Connection, ref_ids: list[int]) -> dict[int, float]:
    """OCR confidence for the candidate blocks, in one query.

    Used only to break ties between cluster members. A missing row (an older
    index, a block inserted without one) simply scores 0.0 and loses the
    tie-break to a block that has a real number.
    """
    if not ref_ids:
        return {}
    marks = ",".join("?" * len(ref_ids))
    rows = conn.execute(
        f"SELECT id, confidence FROM ocr_blocks WHERE id IN ({marks})", ref_ids
    ).fetchall()
    return {int(r["id"]): float(r["confidence"] or 0.0) for r in rows}


def _near_identical(a: str, b: str, threshold: float) -> bool:
    """Same on-screen text, allowing for recognition noise.

    Compared after the project's own search normalization, so Arabic folding,
    CJK/Thai segmentation and digit folding apply here exactly as they do to
    the query — a dedup that used raw bytes would silently not work on any
    script that normalization exists for.
    """
    if a == b:
        return True
    if not a or not b:
        return False
    # Length gate first: it is cheap, and it stops a short string from
    # accidentally scoring high against a long one that merely contains it.
    shorter, longer = sorted((len(a), len(b)))
    if longer and shorter / longer < threshold:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _pick_representative(cluster: list[Hit], confidences: dict[int, float]) -> Hit:
    """The one hit that speaks for a cluster.

    Retrieval score first (it is what the ranking is denominated in), then OCR
    confidence, then the longer text — a truncated read of a caption is worse
    evidence than a complete one — and finally the earliest timestamp purely
    so the choice is deterministic rather than dependent on row order.
    """
    return max(
        cluster,
        key=lambda h: (
            round(h.score, 6),
            confidences.get(h.ref_id, 0.0),
            len(h.text.strip()),
            -(h.timestamp if h.timestamp is not None else 0.0),
        ),
    )


def collapse_ocr_duplicates(
    hits: list[Hit],
    conn: sqlite3.Connection | None = None,
    *,
    window: float | None = None,
    similarity: float | None = None,
) -> list[Hit]:
    """Collapse runs of near-identical OCR from one persistent on-screen text.

    The defect this fixes, measured: a static caption OCR'd on eleven adjacent
    frames produced eleven hits with *identical* scores, which took ranks 2-12
    and pushed the transcript line that actually answered the question down to
    rank 15 — where no top-8 could reach it. Eleven readings of one caption is
    one observation, and it should occupy one slot.

    Deliberately narrow:

    - OCR only. Transcript segments are a different modality and two adjacent
      segments are two different statements, so they are never clustered and
      never absorbed into an OCR cluster.
    - Text alone is not enough. The same caption recurring later in the video
      is a genuinely separate occurrence and stays separate; clustering
      chains through *time*, so a run only continues while consecutive
      readings stay inside ``window`` of each other.
    - Nothing is dropped from the ranking's denominator: a representative
      keeps its own score and then competes normally against transcript hits
      for the final slots.
    """
    from watch_skill.config import get_settings

    settings = get_settings()
    if not settings.retrieval_ocr_dedup_enabled:
        return hits
    window = settings.retrieval_ocr_dedup_window_seconds if window is None else window
    similarity = (
        settings.retrieval_ocr_dedup_similarity if similarity is None else similarity
    )

    ocr = [h for h in hits if h.kind == "ocr" and h.timestamp is not None]
    if len(ocr) < 2:
        return hits

    from watch_skill.index.textnorm import normalize_for_search

    confidences: dict[int, float] = {}
    if conn is not None:
        confidences = _ocr_confidences(conn, [h.ref_id for h in ocr])

    # Chain by time within a video: sorting by timestamp is what makes a
    # persistent caption one run and a later reappearance a new one.
    clusters: list[list[Hit]] = []
    by_video: dict[str, list[Hit]] = {}
    for hit in ocr:
        by_video.setdefault(hit.video_id, []).append(hit)
    for group in by_video.values():
        group.sort(key=lambda h: h.timestamp or 0.0)
        norms = {id(h): normalize_for_search(h.text) for h in group}
        for hit in group:
            for cluster in reversed(clusters):
                if cluster[0].video_id != hit.video_id:
                    continue
                last = cluster[-1]
                gap = abs((hit.timestamp or 0.0) - (last.timestamp or 0.0))
                if gap <= window and _near_identical(
                    norms[id(hit)], norms[id(last)], similarity
                ):
                    cluster.append(hit)
                    break
            else:
                clusters.append([hit])

    winners: dict[int, Hit] = {}
    absorbed: set[int] = set()
    for cluster in clusters:
        if len(cluster) == 1:
            continue
        rep = _pick_representative(cluster, confidences)
        stamps = [h.timestamp for h in cluster if h.timestamp is not None]
        rep.duplicate_count = len(cluster)
        rep.duplicate_first = min(stamps) if stamps else None
        rep.duplicate_last = max(stamps) if stamps else None
        winners[id(rep)] = rep
        for member in cluster:
            if member is not rep:
                absorbed.add(id(member))

    # Original order is preserved for everything that survives, so the caller's
    # own ranking still decides; this only removes redundant competitors.
    return [h for h in hits if id(h) not in absorbed]


def _attach_speakers(conn: sqlite3.Connection, hits: list[Hit]) -> None:
    """Fill ``Hit.speaker`` for segment hits, in one query."""
    ids = [h.ref_id for h in hits if h.kind == "segment"]
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, speaker FROM segments WHERE id IN ({marks})", ids
    ).fetchall()
    speakers = {int(r["id"]): r["speaker"] for r in rows}
    for hit in hits:
        if hit.kind == "segment":
            hit.speaker = speakers.get(int(hit.ref_id))


def require_speaker(
    conn: sqlite3.Connection, video_id: str | None, speaker: str
) -> str:
    """Validate a speaker filter; returns the label as stored.

    A filter that silently matched nothing would read as "this speaker never
    said that", which is a claim. So a video that was never diarized, or a
    label it does not contain, is an error that says what to do instead.
    """
    sql = "SELECT DISTINCT speaker FROM segments WHERE speaker IS NOT NULL"
    params: list[Any] = []
    if video_id:
        sql += " AND video_id = ?"
        params.append(video_id)
    known = sorted(r["speaker"] for r in conn.execute(sql, params).fetchall())
    scope = f"video {video_id}" if video_id else "the index"
    if not known:
        raise IndexError_(
            f"{scope} has no speaker labels, so it cannot be filtered by speaker",
            code="index.not_diarized",
            fix="re-watch with --diarize (needs the diarize extra + "
            "WATCHSKILL_HUGGINGFACE_TOKEN), or drop the speaker filter",
            details={"video_id": video_id, "speaker": speaker},
        )
    match = next((k for k in known if k.lower() == speaker.strip().lower()), None)
    if match is None:
        raise IndexError_(
            f"speaker {speaker!r} does not appear in {scope}",
            code="index.unknown_speaker",
            fix=f"use one of: {', '.join(known)} (labels are case-insensitive)",
            details={"video_id": video_id, "speaker": speaker, "known_speakers": known},
        )
    return match


def checked_speaker(video_id: str | None, speaker: str | None) -> str | None:
    """None when no filter was asked for; otherwise the validated label."""
    if speaker is None or not speaker.strip():
        return None
    conn = connect()
    try:
        return require_speaker(conn, video_id, speaker)
    finally:
        conn.close()


def hybrid_search(
    query: str, video_id: str | None = None, k: int = 8, speaker: str | None = None
) -> list[Hit]:
    """Fuse keyword (FTS5 bm25) and vector (cosine) hits by reciprocal rank.

    Ordering is ``rrf = sum(1 / (retrieval_rrf_k + rank))`` over the two
    lists, ties broken by the weighted relevance and then by identity so the
    order is deterministic. ``Hit.score`` keeps the 0.45/0.55 weighted
    relevance, which is what answer confidence is calibrated on.

    ``speaker`` keeps only transcript segments spoken by that diarized label
    (case-insensitive); it is not validated here — callers that take it from
    an agent go through :func:`require_speaker` first.
    """
    from watch_skill.config import get_settings

    rrf_k = get_settings().retrieval_rrf_k
    conn = connect()
    try:
        merged: dict[tuple[str, str, int], Hit] = {}
        for source, weight, hits in (
            ("fts", 0.45, _fts_hits(conn, query, video_id, speaker)),
            ("vector", 0.55, _vector_hits(conn, query, video_id, speaker)),
        ):
            for rank, hit in enumerate(hits, start=1):
                key = (hit.video_id, hit.kind, hit.ref_id)
                weighted = hit.score * weight
                target = merged.get(key)
                if target is None:
                    hit.score, hit.rrf_score = weighted, 0.0
                    merged[key] = target = hit
                else:
                    target.score += weighted
                target.rrf_score += 1.0 / (rrf_k + rank)
                setattr(target, f"{source}_rank", rank)
        ranked = sorted(
            merged.values(),
            key=lambda h: (-h.rrf_score, -h.score, h.kind, int(h.ref_id), h.video_id),
        )
        # Collapse BEFORE the cut, over the whole candidate pool. Doing it
        # after would dedup a top-k that redundancy had already filled, which
        # is the bug wearing a smaller hat.
        ranked = collapse_ocr_duplicates(ranked, conn)[:k]
        _attach_speakers(conn, ranked)
        return ranked
    finally:
        conn.close()


def frames_near(
    conn: sqlite3.Connection, video_id: str, timestamp: float, limit: int = 2
) -> list[dict[str, Any]]:
    """The frames closest in time to ``timestamp`` for one video."""
    rows = conn.execute(
        """SELECT id, scene_id, timestamp, frame_path, description
           FROM scenes WHERE video_id = ?
           ORDER BY ABS(timestamp - ?) LIMIT ?""",
        (video_id, timestamp, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def ask_video(
    video_id_or_source: str, question: str, k: int = 8, max_frames: int = 6,
    allow_stale: bool = False, speaker: str | None = None,
) -> dict[str, Any]:
    """Retrieval-based answer context: top hits + the frames around them.

    Returns text evidence with timestamps plus a handful of frame paths —
    NOT a re-run of the full analysis. Refuses to answer from artifacts whose
    source has demonstrably changed; ``allow_stale`` opts into reading the
    historical revision anyway. ``speaker`` keeps only what that diarized
    speaker said.
    """
    from watch_skill.index.store import require_current

    video = get_video(video_id_or_source)
    if video is None:
        raise IndexError_(
            f"video not indexed: {video_id_or_source}",
            code="index.video_not_found",
            fix="run watch_video/`watch-skill watch --index` on it first, or list_videos()",
        )
    freshness = require_current(video_id_or_source, allow_stale=allow_stale)
    speaker = checked_speaker(video["id"], speaker)
    hits = hybrid_search(question, video_id=video["id"], k=k, speaker=speaker)
    conn = connect()
    try:
        frame_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.timestamp is None or len(frame_rows) >= max_frames:
                continue
            for frame in frames_near(conn, video["id"], hit.timestamp, limit=1):
                if frame["frame_path"] not in seen and Path(frame["frame_path"]).is_file():
                    seen.add(frame["frame_path"])
                    frame_rows.append(frame)
    finally:
        conn.close()
    return {
        "video": video,
        "question": question,
        "hits": [vars(h) for h in hits],
        "frames": frame_rows,
        "freshness": freshness,
        "speaker": speaker,
    }


def search_videos(query: str, k: int = 12, speaker: str | None = None) -> list[dict[str, Any]]:
    """Cross-video hybrid search; hits grouped with their video rows.

    ``speaker`` keeps only transcript segments with that diarized label, in
    any video that has it.
    """
    speaker = checked_speaker(None, speaker)
    hits = hybrid_search(query, video_id=None, k=k, speaker=speaker)
    videos: dict[str, dict[str, Any]] = {}
    for hit in hits:
        videos.setdefault(hit.video_id, {"video": get_video(hit.video_id), "hits": []})
        videos[hit.video_id]["hits"].append(vars(hit))
    return list(videos.values())


def get_moment(
    video_id_or_source: str, timestamp: float, window: float = 10.0, max_frames: int = 6,
    allow_stale: bool = False, speaker: str | None = None,
) -> MomentContext:
    """Dense context around one moment: nearby frames, transcript, OCR.

    Every segment carries a ``speaker`` key (``None`` when not diarized);
    ``speaker`` keeps only that speaker's segments. Frames and OCR are not
    filtered — nobody "says" a frame.
    """
    from watch_skill.index.store import require_current

    video = get_video(video_id_or_source)
    if video is None:
        raise IndexError_(
            f"video not indexed: {video_id_or_source}",
            code="index.video_not_found",
            fix="run watch_video on it first",
        )
    require_current(video_id_or_source, allow_stale=allow_stale)
    speaker = checked_speaker(video["id"], speaker)
    lo, hi = timestamp - window / 2, timestamp + window / 2
    conn = connect()
    try:
        frames = conn.execute(
            """SELECT id, scene_id, timestamp, frame_path, description FROM scenes
               WHERE video_id = ? AND timestamp BETWEEN ? AND ?
               ORDER BY timestamp""",
            (video["id"], lo, hi),
        ).fetchall()
        if not frames:
            frames = frames_near(conn, video["id"], timestamp, limit=max_frames)
            frames = sorted(frames, key=lambda f: f["timestamp"])
        else:
            frames = [dict(r) for r in frames][:max_frames]
        seg_sql = (
            "SELECT start, end, text, speaker, words_json FROM segments "
            "WHERE video_id = ? AND end >= ? AND start <= ?"
        )
        seg_params: list[Any] = [video["id"], lo, hi]
        if speaker:
            seg_sql += " AND speaker = ? COLLATE NOCASE"
            seg_params.append(speaker)
        segments = conn.execute(seg_sql + " ORDER BY start", seg_params).fetchall()
        ocr = conn.execute(
            """SELECT timestamp, text, confidence FROM ocr_blocks
               WHERE video_id = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp""",
            (video["id"], lo, hi),
        ).fetchall()
        sounds = conn.execute(
            """SELECT start, end, label, confidence, raw_label FROM audio_events
               WHERE video_id = ? AND end >= ? AND start <= ? ORDER BY start""",
            (video["id"], lo, hi),
        ).fetchall()
    finally:
        conn.close()
    # Words are stored as JSON on the segment; decode them here so callers
    # get structure instead of a string, and name the word actually being
    # spoken at the asked-for instant — the point of storing them at all.
    import json as _json  # noqa: PLC0415

    decoded: list[dict] = []
    for row in segments:
        seg = dict(row)
        raw = seg.pop("words_json", None)
        words = _json.loads(raw) if raw else []
        if words:
            seg["words"] = words
            spoken = next(
                (w for w in words if w["start"] <= timestamp <= w["end"]), None
            )
            if spoken is not None:
                seg["word_at_timestamp"] = spoken
        decoded.append(seg)

    return MomentContext(
        video_id=video["id"], timestamp=timestamp, window=window,
        frames=[dict(f) for f in frames],
        segments=decoded,
        ocr=[dict(o) for o in ocr],
        audio_events=[dict(s) for s in sounds],
    )
