"""The embedding model loads on first use, once, and is released when idle.

A long-lived server used to build the model at startup and hold it for the
life of the process, a few hundred MB that a box running only list/watch
never needed.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from watch_skill.index import embeddings


class _Vec(list):
    def tolist(self) -> list[float]:
        return list(self)


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A fastembed whose TextEmbedding records each construction."""
    built: list[str] = []

    class TextEmbedding:
        def __init__(self, model_name: str) -> None:
            time.sleep(0.05)  # widen the race a missing lock would lose
            built.append(model_name)

        def embed(self, texts):
            return [_Vec([1.0, 0.0]) for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=TextEmbedding))
    monkeypatch.setattr(embeddings, "_unavailable", False)
    embeddings.release_models()
    yield built
    embeddings.release_models()


def test_concurrent_first_calls_build_one_model(fake_fastembed: list[str]) -> None:
    barrier = threading.Barrier(8)

    def call() -> None:
        barrier.wait()
        embeddings.embed_texts(["x"])

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake_fastembed == [embeddings.MODEL_NAME]


def test_an_idle_model_is_released_and_reloads_on_next_use(fake_fastembed: list[str]) -> None:
    embeddings.embed_texts(["x"])
    used = embeddings.last_used()
    assert embeddings.release_if_idle(now=used + 299) == 0, "released before the idle window"
    assert embeddings.release_if_idle(now=used + 301) == 1
    assert embeddings.loaded_models() == []
    embeddings.embed_texts(["x"])
    assert fake_fastembed == [embeddings.MODEL_NAME] * 2


def test_use_postpones_the_release(fake_fastembed: list[str]) -> None:
    embeddings.embed_texts(["x"])
    first = embeddings.last_used()
    time.sleep(0.05)  # Windows' monotonic clock ticks every ~16 ms
    embeddings.embed_texts(["y"])
    second = embeddings.last_used()
    assert second > first
    # Idle long enough counted from the first use, not from the second.
    assert embeddings.release_if_idle(now=first + 300 + (second - first) / 2) == 0
    assert embeddings.release_if_idle(now=second + 300) == 1


def test_zero_never_releases(fake_fastembed: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from watch_skill.config import reset_settings

    monkeypatch.setenv("WATCHSKILL_EMBEDDING_IDLE_SECONDS", "0")
    reset_settings()
    embeddings.embed_texts(["x"])
    assert embeddings.release_if_idle(now=embeddings.last_used() + 10**9) == 0
    assert embeddings.loaded_models() == [embeddings.MODEL_NAME]


def test_the_default_idle_window_is_five_minutes() -> None:
    from watch_skill.config import get_settings

    assert get_settings().embedding_idle_seconds == 300


def test_the_first_load_starts_one_daemon_reaper(fake_fastembed: list[str]) -> None:
    embeddings.embed_texts(["x"])
    embeddings.embed_texts(["y"])
    reapers = [t for t in threading.enumerate() if t.name == "ws-embedding-reaper"]
    assert len(reapers) == 1 and reapers[0].daemon


def test_import_native_stack_builds_no_model(fake_fastembed: list[str]) -> None:
    embeddings.import_native_stack()
    assert fake_fastembed == []
    assert embeddings.loaded_models() == []
