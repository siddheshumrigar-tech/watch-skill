"""Retrieval regression set, and the case reciprocal-rank fusion exists for.

The embedder is a deterministic hashed bag of words, so every number here is
reproducible offline and no model is loaded.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from watch_skill.index import embeddings as emb

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_cases.json"
_DIM = 64


def _bag_of_words(text: str) -> list[float]:
    from watch_skill.index.textnorm import normalize_for_search

    vec = [0.0] * _DIM
    for token in normalize_for_search(text).split():
        token = "".join(ch for ch in token if ch.isalnum())
        if token:
            vec[int(hashlib.sha1(token.encode()).hexdigest(), 16) % _DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _seed(videos: list[dict]) -> None:
    """Insert videos, segments and OCR, then index their text the real way."""
    from watch_skill.index.db import connect
    from watch_skill.index.store import _index_texts

    conn = connect()
    try:
        with conn:
            for video in videos:
                conn.execute(
                    "INSERT INTO videos (id, source, title, duration_seconds, frames_dir) "
                    "VALUES (?, ?, ?, 60.0, 'd')",
                    (video["id"], f"{video['id']}.mp4", video["title"]),
                )
                items = []
                for start, end, text in video["segments"]:
                    cur = conn.execute(
                        "INSERT INTO segments (video_id, start, end, text) VALUES (?, ?, ?, ?)",
                        (video["id"], start, end, text),
                    )
                    items.append(("segment", cur.lastrowid, start, text))
                for at, text in video.get("ocr", []):
                    cur = conn.execute(
                        "INSERT INTO ocr_blocks (video_id, timestamp, text, confidence) "
                        "VALUES (?, ?, ?, 0.9)",
                        (video["id"], at, text),
                    )
                    items.append(("ocr", cur.lastrowid, at, text))
                _index_texts(conn, video["id"], items)
    finally:
        conn.close()


def _first_match(hits, case: dict) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if (hit.video_id == case["video"] and hit.kind == case["kind"]
                and case["contains"].lower() in hit.text.lower()):
            return rank
    return None


def test_the_regression_set_keeps_recall_and_mrr(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.index.retrieval import hybrid_search

    monkeypatch.setattr(
        emb, "embed_texts", lambda texts, model_name=None: [_bag_of_words(t) for t in texts]
    )
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    _seed(data["videos"])

    ranks = []
    for case in data["cases"]:
        rank = _first_match(hybrid_search(case["query"], k=8), case)
        assert rank is not None and rank <= 3, f"{case['query']!r} -> rank {rank}"
        ranks.append(rank)
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    assert mrr >= data["baseline_mrr"], f"MRR {mrr:.3f} fell below {data['baseline_mrr']}"


def test_hits_report_both_ranks_and_keep_the_calibrated_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from watch_skill.index.retrieval import hybrid_search

    monkeypatch.setattr(
        emb, "embed_texts", lambda texts, model_name=None: [_bag_of_words(t) for t in texts]
    )
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    _seed(data["videos"])

    top = hybrid_search("oat milk fridge", k=3)[0]
    assert (top.fts_rank, top.vector_rank) == (1, 1)
    assert top.rrf_score == pytest.approx(2 / 61)
    # The calibrated relevance the confidence model reads, not the fused rank.
    assert 0.45 < top.score <= 1.0


# The case the summed score gets wrong. Every bm25 score squashes to the same
# 1.0, so the old sum ranked keyword hits by cosine alone and let a single
# high-cosine vector outlier, which no keyword matched, beat the segment both
# retrievers placed second.
_QUERY = "invoice"
_VECTORS = {
    _QUERY: [1.0, 0.0, 0.0],
    # keyword rank 1 (the term three times), vector rank 6
    "invoice invoice invoice": [-0.5, 0.0, 0.866],
    # keyword rank 2, vector rank 2: the answer both lists agree on
    "the client said the invoice total was wrong last week": [0.10, 0.995, 0.0],
    # vector rank 1, no keyword match: the outlier
    "billing statement figures": [0.99, 0.141, 0.0],
    "filler one": [0.05, 0.0, 0.999],
    "filler two": [0.04, 0.0, 0.999],
    "filler three": [0.03, 0.0, 0.9995],
}


def test_agreement_beats_a_single_list_outlier(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.index.retrieval import hybrid_search

    monkeypatch.setattr(
        emb, "embed_texts", lambda texts, model_name=None: [_VECTORS[t] for t in texts]
    )
    _seed([{
        "id": "rv00000000000009", "title": "outlier",
        "segments": [[float(i), float(i + 1), text]
                     for i, text in enumerate(t for t in _VECTORS if t != _QUERY)],
    }])
    hits = hybrid_search(_QUERY, k=5)
    assert hits[0].text == "the client said the invoice total was wrong last week", [
        (h.text, round(h.score, 3)) for h in hits
    ]
    assert "billing statement figures" in [h.text for h in hits], "the outlier still ranks"


def test_rrf_k_is_a_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.config import get_settings, reset_settings

    assert get_settings().retrieval_rrf_k == 60
    monkeypatch.setenv("WATCHSKILL_RETRIEVAL_RRF_K", "10")
    reset_settings()
    assert get_settings().retrieval_rrf_k == 10
