"""Pre-cache the live speech model, deliberately, before a session needs it.

A live session never downloads a model: capture is already running, the
network may be off by policy, and a surprise fetch mid-session is both a
stall and an egress nobody agreed to. That rule left an agent with an error
and no command to run. ``watch-skill models pull asr`` is that command.
"""
from __future__ import annotations

import time
from typing import Any

from watch_skill.errors import WatchSkillError


def live_asr_model_size() -> str:
    """The model a live session will load: configured, else what fits.

    Live transcription competes with OCR and embeddings for memory, so this
    is deliberately more conservative than the batch path: being a little
    less accurate beats being killed by the allocator mid-session.
    """
    from watch_skill.config import get_settings  # noqa: PLC0415

    configured = get_settings().whisper_model
    if configured and configured != "auto":
        return configured
    try:
        import psutil  # noqa: PLC0415

        available_gb = psutil.virtual_memory().available / 1024**3
    except Exception:  # noqa: BLE001
        return "tiny"
    if available_gb >= 8:
        return "small"
    return "base" if available_gb >= 4 else "tiny"


def _faster_whisper() -> Any:
    try:
        import faster_whisper  # noqa: PLC0415
    except ImportError as exc:
        raise WatchSkillError(
            "faster-whisper is not installed",
            code="models.missing_dependency",
            fix='pip install "watch-skill[whisper]" (from a checkout: '
            "`uv sync --extra whisper`)",
        ) from exc
    return faster_whisper


def is_asr_model_cached(size: str) -> bool:
    """Whether ``size`` loads from the local cache with no network."""
    try:
        faster_whisper = _faster_whisper()
        faster_whisper.download_model(size, local_files_only=True)
    except Exception:  # noqa: BLE001 - "not cached" covers every way it can fail
        return False
    return True


def pull_asr_model(size: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Download the live speech model once, then prove it loads offline.

    Returns ``{"status", "model", "path", "seconds"}``; ``status`` is
    ``already_cached``, ``would_download`` (dry run) or ``downloaded``.
    Refuses in offline mode rather than reaching out.
    """
    from watch_skill.config import get_settings  # noqa: PLC0415

    model = size or live_asr_model_size()
    faster_whisper = _faster_whisper()
    started = time.monotonic()
    if is_asr_model_cached(model):
        path = faster_whisper.download_model(model, local_files_only=True)
        return {"status": "already_cached", "model": model, "path": str(path),
                "seconds": round(time.monotonic() - started, 2)}
    if dry_run:
        return {"status": "would_download", "model": model, "path": None, "seconds": 0.0}
    if get_settings().offline:
        raise WatchSkillError(
            f"offline mode forbids downloading the speech model {model!r}",
            code="models.offline_refused",
            fix="run this once with WATCHSKILL_OFFLINE unset, or copy the "
            "model into the Hugging Face cache by hand",
            details={"model": model},
        )
    try:
        path = faster_whisper.download_model(model)
        faster_whisper.WhisperModel(model, device="cpu", compute_type="int8",
                                    local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise WatchSkillError(
            f"could not fetch the speech model {model!r}: {exc}",
            code="models.download_failed",
            fix="check the network and free disk space, then run "
            f"`watch-skill models pull asr --size {model}` again",
            details={"model": model, "error": str(exc)[:200]},
        ) from exc
    return {"status": "downloaded", "model": model, "path": str(path),
            "seconds": round(time.monotonic() - started, 2)}
