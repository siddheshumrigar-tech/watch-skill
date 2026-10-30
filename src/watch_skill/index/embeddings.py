"""Local text embeddings via fastembed (ONNX MiniLM-class; no torch).

The default model is multilingual: same 384 dims and interface as the
English-only all-MiniLM-L6-v2 it replaced, but it actually retrieves
Arabic/Russian/Hindi/Chinese — including cross-lingual (English question
over an Arabic transcript). Benchmark in docs/DECISIONS.md.

Degrades loudly: when fastembed is not installed, the index still works with
FTS5 keyword search only — hybrid retrieval just loses its vector half.
"""
from __future__ import annotations

import struct
import sys
import threading
import time

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_models: dict[str, object] = {}
_unavailable = False
# One lock for build and release: concurrent first calls must construct one
# model, and a release must never interleave with a build.
_lock = threading.Lock()
_last_used: float | None = None  # time.monotonic() of the last embed/load
_reaper: threading.Thread | None = None


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()


def _get_model(name: str):
    model = _models.get(name)
    if model is not None or _unavailable:
        if model is not None:
            _touch()
        return model
    with _lock:
        model = _models.get(name)
        if model is None and not _unavailable:
            _build_model(name)
            model = _models.get(name)
            if model is not None:
                _start_reaper()
    if model is not None:
        _touch()
    return model


def _build_model(name: str) -> None:
    """Construct one model into ``_models``. Caller holds ``_lock``."""
    global _unavailable
    try:
        import warnings  # noqa: PLC0415

        from fastembed import TextEmbedding  # noqa: PLC0415

        with warnings.catch_warnings():
            # informational pooling-change notice; the multilingual bench in
            # docs/DECISIONS.md was measured on the current (mean) pooling
            warnings.filterwarnings("ignore", message=".*mean pooling.*")
            _models[name] = TextEmbedding(model_name=name)
    except ImportError:
        _unavailable = True
        print(
            "[watch-skill] fastembed not installed — keyword-only search "
            '(install with `pip install "watch-skill[index]"`, or `uv sync --extra '
            'index` from a checkout)',
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        # Installed but unloadable: out of memory, a truncated model cache, a
        # runtime the CPU cannot use. Semantic search is a ranking
        # improvement, not a precondition for retrieval, so degrade to
        # keyword-only rather than taking every query down with it. Announced
        # once, on stderr, because silently worse results are their own bug.
        _unavailable = True
        print(
            f"[watch-skill] embedding model unavailable ({type(exc).__name__}: "
            f"{exc}) — falling back to keyword-only search",
            file=sys.stderr,
        )


def _idle_seconds() -> float:
    from watch_skill.config import get_settings  # noqa: PLC0415

    return float(get_settings().embedding_idle_seconds)


def _start_reaper() -> None:
    """Start the idle reaper once, on the first load. Caller holds ``_lock``."""
    global _reaper
    if _reaper is not None and _reaper.is_alive():
        return
    _reaper = threading.Thread(target=_reap_forever, name="ws-embedding-reaper", daemon=True)
    _reaper.start()


def _reap_forever() -> None:
    while True:
        idle = _idle_seconds()
        time.sleep(min(30.0, idle / 2) if idle > 0 else 30.0)
        try:
            release_if_idle()
        except Exception:  # noqa: BLE001 - the reaper must never die loudly
            pass


def release_if_idle(now: float | None = None) -> int:
    """Release loaded models unused for ``embedding_idle_seconds``.

    Returns how many were released. ``0`` disables the release. The next
    embed reloads the model, paying the load once more rather than holding a
    few hundred MB for a server that may not search again for hours.
    """
    idle = _idle_seconds()
    if idle <= 0:
        return 0
    now = time.monotonic() if now is None else now
    with _lock:
        if not _models or _last_used is None or now - _last_used < idle:
            return 0
        count = len(_models)
        _models.clear()
    print(f"[watch-skill] embedding model released after {idle:.0f}s idle", file=sys.stderr)
    return count


def last_used() -> float | None:
    """``time.monotonic()`` of the last embed or load, or None."""
    return _last_used


def loaded_models() -> list[str]:
    """Names of the embedding models currently in memory."""
    return sorted(_models)


def import_native_stack() -> bool:
    """Import fastembed, numpy and onnxruntime on the calling thread.

    Builds no model. The import is what deadlocks on a worker thread; the
    model itself loads on first use, wherever that happens. Returns whether
    the stack is importable.
    """
    try:
        import numpy  # noqa: F401, PLC0415
        import onnxruntime  # noqa: F401, PLC0415
        from fastembed import TextEmbedding  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def warm_native_imports() -> None:
    """Import the native-extension stack on the calling thread, before serving.

    Both servers in this project answer requests on worker threads: FastMCP
    runs sync tools in a pool, and the Bridge owns a bounded pool of its own.
    The embedding stack (fastembed -> numpy -> onnxruntime) is imported lazily
    at its call sites, so the first import lands inside a worker — where
    loading the numpy C extension deadlocks and never returns.

    The symptom is asymmetric and misleading in both. Measured on the Bridge
    against 1.4.3: the first ``watch.library.search`` in a fresh Core did not
    answer in fifteen minutes; a second search issued afterwards, in the same
    process, answered in 51 seconds, and a third in 1.2. Everything that does
    not embed stayed instant throughout, so the engine looked healthy and the
    one tool an agent reaches for to find a source hung forever.

    The MCP server has had this call since the deadlock was first found there.
    The Bridge did not, which is the whole of the defect: two servers with the
    same shape and one of them warmed. It lives here now so a third cannot be
    written without it.

    Only the imports are warmed. The model is built on first use (and the
    stack is per process, not per model, so an index pinned to an older
    model needs nothing extra here), and released again after
    ``embedding_idle_seconds`` without use.

    Best-effort by design: a box where fastembed is missing or unloadable
    degrades to keyword-only search, and serving must start either way.
    """
    try:
        import_native_stack()
    except Exception as exc:  # noqa: BLE001 - never block serving on a warmup
        print(f"[watch-skill] embedding warmup skipped ({exc})", file=sys.stderr)


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]] | None:
    """Embed a batch of texts; ``None`` when embeddings are unavailable.

    ``model_name`` overrides the default — the index read/write paths pass
    the model recorded in the index meta so stored vectors and query vectors
    always come from the same model.
    """
    model = _get_model(model_name or MODEL_NAME)
    if model is None or not texts:
        return None if model is None else []
    return [vec.tolist() for vec in model.embed(texts)]


def pack_vector(vector: list[float]) -> bytes:
    """float32 little-endian blob for SQLite storage.

    float16 was tried and measured, and rejected. It does halve the index —
    100k vectors go from 197 MB to 80 MB — and costs nothing in ranking
    (largest cosine error on this model's output: 2.3e-5, top-20 identical).
    But every read has to widen it back, and that dominates the scan: 115 ms
    becomes ~310 ms per 100k, the same whether the conversion is done in one
    astype, in cache-sized blocks, or by letting numpy handle a float16
    matmul. Trading 200 ms on every query for 118 MB of disk is the wrong way
    round for a search path.

    :func:`unpack_vector` still reads either width, so an index written
    during that experiment keeps working.
    """
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Decode a stored vector, float16 or the float32 an older index wrote.

    The width is inferred from the blob rather than recorded, so an index
    built before the switch keeps working without a migration and without a
    rewrite of every row.
    """
    if len(blob) == dim * 2:
        return list(struct.unpack(f"<{dim}e", blob))
    return list(struct.unpack(f"<{dim}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity without numpy (vectors are short; hot path is SQL)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def release_models() -> int:
    """Drop cached embedding models and return how many were released.

    Same reasoning as the OCR engine cache: shared deliberately at runtime,
    but inherited accidentally between tests.
    """
    count = len(_models)
    _models.clear()
    return count
