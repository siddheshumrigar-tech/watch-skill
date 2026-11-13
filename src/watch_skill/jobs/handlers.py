"""The durable job kinds Watch Skill ships.

Each handler returns a *reference* — a video_id, a session_id — rather than a
live object, because that is what still means something after the process
that produced it has exited.
"""
from __future__ import annotations

from watch_skill.jobs.registry import JobContext, register
from watch_skill.jobs.types import JobStage

# Rough share of a watch that each stage accounts for. Progress from real
# stage boundaries rather than a hard-coded percentage per phase, so a video
# that spends ten minutes in transcription does not sit at "70%" throughout.
_STAGE_PROGRESS: dict[JobStage, float] = {
    JobStage.ACQUIRE: 0.10,
    JobStage.PROBE: 0.15,
    JobStage.FRAMES: 0.40,
    JobStage.OCR: 0.55,
    JobStage.TRANSCRIBE: 0.80,
    JobStage.EMBED: 0.92,
    JobStage.FINALIZE: 0.98,
}


def _stage_for(phase: str, fraction: float) -> tuple[JobStage, float]:
    """Map a watch() progress phase to a job stage and overall progress.

    Matched by prefix: local whisper reports
    ``transcribing (local whisper) 120/600s, 14 segments`` many times, and an
    exact-string table sent every one of those to PROBE. Transcription
    progress (watch() reports 0.7 -> 0.9) is interpolated between the OCR and
    TRANSCRIBE stage values, so a long transcription visibly moves.
    """
    if phase.startswith("acquiring"):
        return JobStage.ACQUIRE, _STAGE_PROGRESS[JobStage.ACQUIRE]
    if phase.startswith("extracting frames"):
        return JobStage.FRAMES, _STAGE_PROGRESS[JobStage.FRAMES]
    if phase.startswith("transcribing"):
        lo, hi = _STAGE_PROGRESS[JobStage.OCR], _STAGE_PROGRESS[JobStage.TRANSCRIBE]
        share = max(0.0, min(1.0, (fraction - 0.7) / 0.2))
        return JobStage.TRANSCRIBE, lo + (hi - lo) * share
    return JobStage.PROBE, _STAGE_PROGRESS.get(JobStage.PROBE, fraction)


def _watch_job(ctx: JobContext) -> tuple[str | None, str | None]:
    """Watch and index a source, checkpointing at every stage boundary."""
    from watch_skill.index.store import index_watch_result
    from watch_skill.perceive.budget import parse_time
    from watch_skill.watch import watch

    payload = ctx.payload
    ctx.checkpoint(JobStage.ACQUIRE, _STAGE_PROGRESS[JobStage.ACQUIRE])

    def on_progress(phase: str, fraction: float) -> None:
        # The pipeline's own progress callback is mapped onto stages so a
        # cancel lands between phases, and inside a long local-whisper run,
        # rather than only at the end.
        stage, progress = _stage_for(phase, fraction)
        ctx.checkpoint(stage, progress)

    result = watch(
        payload["source"],
        start_seconds=parse_time(payload.get("start")),
        end_seconds=parse_time(payload.get("end")),
        max_frames=payload.get("budget"),
        language=payload.get("language"),
        audio_events=payload.get("audio_events") or None,
        on_progress=on_progress,
    )
    ctx.checkpoint(JobStage.EMBED, _STAGE_PROGRESS[JobStage.EMBED])
    video_id = index_watch_result(result)
    ctx.checkpoint(JobStage.FINALIZE, _STAGE_PROGRESS[JobStage.FINALIZE])
    return video_id, "video_id"


def _finalize_live_job(ctx: JobContext) -> tuple[str | None, str | None]:
    """Turn a stopped live session into permanent searchable memory."""
    from watch_skill.live.finalize import finalize_session

    ctx.checkpoint(JobStage.FINALIZE, 0.2)
    return finalize_session(ctx.payload["session_id"], ctx=ctx), "video_id"


register("watch", _watch_job)
register("finalize_live", _finalize_live_job)
