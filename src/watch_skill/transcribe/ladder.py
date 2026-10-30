"""The transcription ladder: captions -> local whisper -> cloud (opt-in).

Each rung's failure is reported to stderr, recorded as a warning
(``{"code", "message", "fix"}``) on the Transcript finally returned, and the
next rung tried; the Transcript records which rung produced it (``source``).
Returns an empty Transcript (source="none") rather than raising when every
rung fails — frames-only analysis is still useful, and the warnings name the
gap.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from watch_skill.config import get_settings
from watch_skill.errors import TranscriptionError
from watch_skill.transcribe import audio as audio_mod
from watch_skill.transcribe import cloud, local, vtt
from watch_skill.transcribe.languages import normalize_language, validate_language
from watch_skill.transcribe.types import Transcript


def _warning(code: str, message: str, fix: str) -> dict:
    return {"code": code, "message": message, "fix": fix}


def _from_error(exc: TranscriptionError) -> dict:
    return _warning(exc.code, exc.message, exc.fix or "")


def caption_language(subtitle_path: Path) -> str | None:
    """The language yt-dlp put in a subtitle filename (``talk.es-419.vtt`` -> ``es``)."""
    tag = Path(Path(subtitle_path).stem).suffix.lstrip(".")
    return normalize_language(tag) if tag else None


def _captions(subtitle_path: Path, warnings: list[dict]) -> Transcript | None:
    try:
        transcript = vtt.parse_vtt(Path(subtitle_path))
    except OSError as exc:
        print(f"[watch-skill] caption parse failed: {exc}", file=sys.stderr)
        warnings.append(_warning(
            "transcribe.captions_unreadable", f"caption parse failed: {exc}",
            "re-download the source, or pass --no-cache so captions are fetched again; "
            "local whisper transcribes the audio meanwhile",
        ))
        return None
    if not transcript:
        return None
    language = caption_language(Path(subtitle_path))
    return replace(transcript, language=language,
                   language_source="captions" if language else None)


def get_transcript(
    video_path: Path | None,
    work_dir: Path,
    subtitle_path: Path | None = None,
    has_audio: bool = True,
    allow_local: bool | None = None,
    allow_cloud: bool | None = None,
    whisper_model: str | None = None,
    preferred_cloud_backend: str | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    word_timestamps: bool = False,
    language: str | None = None,
    on_progress: Callable[[float, float, int], None] | None = None,
) -> Transcript:
    """Run the ladder. Any rung may be force-disabled; defaults come from settings.

    ``on_progress(done_seconds, total_seconds, segments)`` reaches the
    local-whisper rung only (window-relative seconds); captions are instant
    and cloud STT reports nothing until it answers.

    ``language`` forces the spoken language (a code or English name; None
    falls back to ``WATCHSKILL_WHISPER_LANGUAGE``, then auto-detection). It
    reaches the local-whisper and cloud rungs; captions report the language
    in their filename. ``word_timestamps`` reaches the local-whisper and
    cloud rungs; captions carry no word alignment. A cloud backend that
    returns no words leaves a ``transcribe.cloud_no_word_timestamps``
    warning on the transcript.
    """
    settings = get_settings()
    use_local = settings.local_whisper_enabled if allow_local is None else allow_local
    use_cloud = settings.cloud_stt_enabled if allow_cloud is None else allow_cloud
    model = whisper_model or settings.whisper_model
    lang = validate_language(language if language is not None else settings.whisper_language)
    warnings: list[dict] = []

    def done(transcript: Transcript) -> Transcript:
        return replace(transcript, warnings=[*warnings, *transcript.warnings])

    if subtitle_path is not None and Path(subtitle_path).is_file():
        transcript = _captions(Path(subtitle_path), warnings)
        if transcript is not None:
            return done(transcript)

    if video_path is None or not has_audio:
        if video_path is not None:
            print("[watch-skill] no audio stream — skipping transcription", file=sys.stderr)
            warnings.append(_warning(
                "transcribe.no_audio", "the source has no audio stream",
                "a silent video is fine — OCR and vision still index it; if audio "
                "should exist, re-download the source",
            ))
        return done(Transcript(source="none"))

    if use_local:
        try:
            # focused mode: whisper only ever sees the requested window
            audio_path = audio_mod.extract_audio(
                video_path, work_dir / "audio.mp3",
                start_seconds=start_seconds, end_seconds=end_seconds,
            )
            transcript = local.transcribe_local(
                audio_path, model_size=model, language=lang,
                word_timestamps=word_timestamps, on_progress=on_progress,
            )
            return done(transcript.offset(start_seconds or 0.0))
        except TranscriptionError as exc:
            print(f"[watch-skill] local whisper unavailable ({exc.code})", file=sys.stderr)
            warnings.append(_from_error(exc))

    if use_cloud:
        try:
            return done(cloud.transcribe_cloud(
                video_path, work_dir, preferred_backend=preferred_cloud_backend,
                word_timestamps=word_timestamps, language=lang,
            ))
        except TranscriptionError as exc:
            print(f"[watch-skill] cloud STT failed ({exc.code})", file=sys.stderr)
            warnings.append(_from_error(exc))

    if not use_local and not use_cloud:
        warnings.append(_warning(
            "transcribe.no_rung_enabled",
            "no captions, and local whisper and cloud STT are both disabled",
            "set WATCHSKILL_LOCAL_WHISPER_ENABLED=1 (pip install \"watch-skill[whisper]\") "
            "or opt in to cloud STT with WATCHSKILL_CLOUD_STT_ENABLED=1",
        ))
    return done(Transcript(source="none"))
