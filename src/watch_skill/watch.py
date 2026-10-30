"""The core watch pipeline: acquire -> perceive -> transcribe.

This is the engine's front door for one-shot analysis. Indexing (Milestone 2)
builds on the same WatchResult.
"""
from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watch_skill.acquire import AcquireResult, acquire, fetch_captions_only
from watch_skill.acquire.sources import classify_source, is_url_kind
from watch_skill.errors import PerceptionError
from watch_skill.perceive import PerceptionResult, perceive, probe
from watch_skill.perceive.cues import discover_cues
from watch_skill.perceive.types import VideoMetadata
from watch_skill.transcribe import Transcript, get_transcript
from watch_skill.transcribe.languages import validate_language


@dataclass
class WatchResult:
    """Everything one watch pass produced."""

    acquisition: AcquireResult
    metadata: VideoMetadata
    perception: PerceptionResult | None
    transcript: Transcript
    work_dir: Path
    start_seconds: float | None = None
    end_seconds: float | None = None
    # video | audio (no video stream: frames skipped by design) | none (no
    # media was fetched, e.g. a captions-only pass)
    media_kind: str = "video"
    # One {"code", "message", "fix"} per degradation: the transcript ladder's
    # failed rungs first, then the pipeline's own.
    warnings: list[dict] = field(default_factory=list)
    # Non-speech sound events, when audio-event detection ran.
    audio_events: list = field(default_factory=list)

    @property
    def focused(self) -> bool:
        return self.start_seconds is not None or self.end_seconds is not None

    def to_summary(self) -> dict[str, Any]:
        """Stable machine-readable summary: every key always present."""
        transcript = self.transcript
        perception = self.perception
        frames = perception.frames if perception is not None else []
        seconds_per_frame = getattr(perception, "seconds_per_frame", None)
        if seconds_per_frame is None and frames:
            lo = self.start_seconds or 0.0
            hi = (self.end_seconds if self.end_seconds is not None
                  else self.metadata.duration_seconds)
            seconds_per_frame = round((hi - lo) / len(frames), 2) if hi > lo else None
        labels = sorted({
            e.get("label") if isinstance(e, dict) else getattr(e, "label", None)
            for e in self.audio_events
        } - {None})
        return {
            "source": self.acquisition.source,
            "media_kind": self.media_kind,
            "duration_seconds": self.metadata.duration_seconds,
            "focused": self.focused,
            "transcript": {
                "source": transcript.source,
                "language": transcript.language,
                "language_probability": transcript.language_probability,
                "language_source": transcript.language_source,
                "segment_count": len(transcript.segments),
                "has_speakers": any(s.speaker for s in transcript.segments),
                "has_word_timestamps": any(s.words for s in transcript.segments),
            },
            "frames": {
                "count": len(frames),
                "seconds_per_frame": seconds_per_frame,
                "sparse": bool(getattr(perception, "sampling_sparse", False)),
            },
            "audio_events": {"count": len(self.audio_events), "labels": labels},
            "warnings": [dict(w) for w in self.warnings],
        }


def _warning(code: str, message: str, fix: str) -> dict:
    return {"code": code, "message": message, "fix": fix}


def _ignored_visual_options(
    max_frames: int | None,
    cue_timestamps: list[float] | None,
    run_ocr: bool | None,
    frame_width: int | None,
) -> list[str]:
    """Visual options the caller passed that an audio-only source cannot use."""
    passed = {
        "max_frames": max_frames is not None,
        "cue_timestamps": cue_timestamps is not None,
        "run_ocr": run_ocr is True,
        "frame_width": frame_width is not None,
    }
    return [name for name, given in passed.items() if given]


def _validate_budget(max_frames: int | None) -> None:
    """A frame budget below 1 is not a budget.

    Zero and negatives used to fall through to the sampler, which clamped
    them to a single frame and reported a normal-looking selection — so a
    typo produced a confident report built from one frame.
    """
    if max_frames is not None and max_frames < 1:
        raise PerceptionError(
            f"max_frames must be at least 1 (got {max_frames})",
            code="perceive.bad_budget",
            fix="drop max_frames to use the automatic budget, or pass a "
            "positive count; --detail transcript skips frames entirely",
        )


def _validate_window(
    start: float | None, end: float | None, duration: float
) -> None:
    if start is not None and start < 0:
        raise PerceptionError(
            "start must be non-negative", code="perceive.bad_window",
            fix="pass start as SS, MM:SS, or HH:MM:SS from the video's own timeline",
        )
    if start is not None and end is not None and end <= start:
        raise PerceptionError(
            "end must be greater than start", code="perceive.bad_window",
            fix="swap the values or drop end to sample from start to the video's end",
        )
    if duration > 0 and start is not None and start >= duration:
        raise PerceptionError(
            f"start {start:.1f}s is past the end of the video ({duration:.1f}s)",
            code="perceive.bad_window",
            fix=f"use a start below {duration:.0f}s, or drop the window to watch the whole video",
        )


def watch(
    source: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    max_frames: int | None = None,
    frame_width: int | None = None,
    cue_timestamps: list[float] | None = None,
    transcript_only: bool = False,
    run_ocr: bool | None = None,
    allow_local_whisper: bool | None = None,
    allow_cloud_stt: bool | None = None,
    whisper_model: str | None = None,
    word_timestamps: bool = False,
    diarize: bool | None = None,
    duration_cap: float | None = None,
    out_dir: Path | None = None,
    use_cache: bool = True,
    on_progress: Callable[[str, float], None] | None = None,
    language: str | None = None,
) -> WatchResult:
    """Analyze any video source; returns frames + transcript + metadata.

    ``transcript_only`` skips media download entirely when captions exist
    (reference-proven fast path). ``cue_timestamps`` pins frames at given
    absolute times. ``duration_cap`` bounds live-stream capture.
    ``language`` forces the spoken language (ISO 639-1 code or English
    name); omitted, whisper detects it and the transcript reports it.
    """
    # Cheap argument checks first: a bad budget or language should not cost
    # a download.
    _validate_budget(max_frames)
    language = validate_language(language)
    if language is None:
        from watch_skill.config import get_settings

        validate_language(get_settings().whisper_language)  # a bad setting fails here too

    work_dir = Path(out_dir).resolve() if out_dir else Path(tempfile.mkdtemp(prefix="watch-skill-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    def progress(phase: str, fraction: float) -> None:
        if on_progress is not None:
            on_progress(phase, fraction)

    progress("acquiring source", 0.05)
    kind = classify_source(source)
    acq: AcquireResult
    if transcript_only and not cue_timestamps and is_url_kind(kind):
        acq = fetch_captions_only(source)
        if acq.subtitle_path is None:
            # no captions -> whisper needs audio after all
            acq = acquire(source, audio_only=True, duration_cap=duration_cap, use_cache=use_cache)
    else:
        acq = acquire(source, duration_cap=duration_cap, use_cache=use_cache)

    if acq.video_path is not None:
        metadata = probe(acq.video_path)
    else:
        metadata = VideoMetadata(
            duration_seconds=float(acq.info.get("duration") or 0),
            width=None, height=None, fps=None, codec=None, has_audio=False,
        )
    _validate_window(start_seconds, end_seconds, metadata.duration_seconds)

    # An audio file (wav, mp3, a podcast; cover art does not count as video)
    # has nothing to sample. Running perception anyway failed in frame
    # extraction, so it is skipped by design and the caller is told.
    warnings: list[dict] = []
    media_kind = "none" if acq.video_path is None else "video"
    if acq.video_path is not None and metadata.width is None and metadata.has_audio:
        media_kind = "audio"
        ignored = _ignored_visual_options(max_frames, cue_timestamps, run_ocr, frame_width)
        if ignored:
            warnings.append(_warning(
                "perceive.audio_only_ignored_options",
                f"the source has no video stream, so {', '.join(ignored)} had no effect",
                "drop the visual options for audio sources; the transcript is the "
                "whole result",
            ))

    # A recording made by `watch-skill capture --script` carries the moments
    # its interactions happened at, and those are the frames it exists to
    # show -- also the frames near-duplicate selection is likeliest to drop,
    # a page where three numbers change hashing within the threshold of the
    # page before it. Reading them here rather than in a surface is what gives
    # every ingestion path the same behaviour: the CLI, MCP, REST, the job
    # queue and the batch runner all arrive through this function. An explicit
    # `cue_timestamps` is the caller naming the moments that matter, and wins.
    if cue_timestamps is None and acq.video_path is not None and media_kind == "video":
        cue_timestamps = discover_cues(acq.video_path, metadata.duration_seconds)

    progress("extracting frames (scenes, dedup, OCR)", 0.35)
    perception: PerceptionResult | None = None
    if not transcript_only and acq.video_path is not None and media_kind == "video":
        perception = perceive(
            acq.video_path,
            work_dir / "frames",
            source_label=source,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            max_frames=max_frames,
            frame_width=frame_width,
            cue_timestamps=cue_timestamps,
            run_ocr=run_ocr,
            ocr_lang=acq.info.get("language"),  # auto: Arabic video -> Arabic OCR
            metadata=metadata,
        )

    progress("transcribing (captions -> local whisper)", 0.7)

    def whisper_progress(done: float, total: float, segments: int) -> None:
        # A long local transcription is most of a watch; reporting it per
        # segment is what lets a poller tell slow from stuck.
        share = min(1.0, done / total) if total > 0 else 0.0
        progress(
            f"transcribing (local whisper) {done:.0f}/{total:.0f}s, {segments} segments",
            0.7 + 0.2 * share,
        )

    transcript = get_transcript(
        acq.video_path,
        work_dir,
        subtitle_path=acq.subtitle_path,
        has_audio=metadata.has_audio,
        allow_local=allow_local_whisper,
        allow_cloud=allow_cloud_stt,
        whisper_model=whisper_model,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        word_timestamps=word_timestamps,
        language=language,
        on_progress=whisper_progress,
    )
    if start_seconds is not None or end_seconds is not None:
        transcript = transcript.filter_range(start_seconds, end_seconds)

    from watch_skill.config import get_settings

    do_diarize = get_settings().diarization_enabled if diarize is None else diarize
    if do_diarize and transcript and acq.video_path is not None:
        from watch_skill.transcribe.diarize import diarize_transcript

        transcript = diarize_transcript(
            transcript, acq.video_path, work_dir,
            start_seconds=start_seconds, end_seconds=end_seconds,
        )

    # A silent video already carries transcribe.no_audio from the ladder; an
    # empty transcript of something that could have speech is worth naming.
    if not transcript and (metadata.has_audio or acq.video_path is None):
        warnings.append(_warning(
            "transcribe.no_transcript",
            "no transcript was produced (no captions, and no speech-to-text rung "
            "succeeded); the other warnings name each rung that failed",
            'install local whisper: pip install "watch-skill[whisper]" (from a '
            "checkout: uv sync --extra whisper), or opt in to cloud STT",
        ))

    return WatchResult(
        acquisition=acq,
        metadata=metadata,
        perception=perception,
        transcript=transcript,
        work_dir=work_dir,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        media_kind=media_kind,
        warnings=[*transcript.warnings, *warnings],
    )
