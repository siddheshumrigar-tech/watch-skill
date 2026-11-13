"""Render a WatchResult as the agent-facing markdown report.

The format is deliberately compatible with the reference project's proven
contract: header facts, a frame list with `t=MM:SS` + selection reason, and a
fenced timestamped transcript. Agents already know how to consume this.
"""
from __future__ import annotations

import json

from watch_skill.perceive.budget import format_time
from watch_skill.transcribe.types import Transcript
from watch_skill.watch import WatchResult

AUDIO_ONLY_NOTE = "audio-only (no video stream; frames, scenes and OCR skipped by design)"


def _header_lines(result: WatchResult) -> list[str]:
    meta = result.metadata
    info = result.acquisition.info
    lines = ["# watch-skill: video report", ""]
    lines.append(f"- **Source:** {result.acquisition.source}")
    if info.get("title"):
        lines.append(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        lines.append(f"- **Uploader:** {info['uploader']}")
    lines.append(
        f"- **Duration:** {format_time(meta.duration_seconds)} ({meta.duration_seconds:.1f}s)"
    )
    if result.focused:
        lo = result.start_seconds or 0.0
        hi = result.end_seconds if result.end_seconds is not None else meta.duration_seconds
        lines.append(f"- **Focus range:** {format_time(lo)} -> {format_time(hi)} ({hi - lo:.1f}s)")
    if result.media_kind == "audio":
        lines.append(f"- **Media:** {AUDIO_ONLY_NOTE}")
    if meta.width and meta.height:
        lines.append(f"- **Resolution:** {meta.width}x{meta.height} ({meta.codec or 'unknown codec'})")
    lines.append(f"- **Acquired via:** {result.acquisition.acquirer}"
                 + (" (cache)" if result.acquisition.from_cache else ""))
    return lines


def _frames_lines(result: WatchResult) -> list[str]:
    lines = ["", "## Frames", ""]
    perception = result.perception
    if result.media_kind == "audio":
        lines.append(f"_None: the source is {AUDIO_ONLY_NOTE}._")
        return lines
    if perception is None or not perception.frames:
        lines.append("_No frames extracted._")
        return lines
    lines.append(
        f"- **Selection:** {len(perception.frames)} kept from {perception.candidate_count} "
        f"candidates ({perception.engine} engine, {perception.scene_count} scenes, "
        f"{perception.deduped_count} near-duplicates dropped)"
    )
    ocr_frames = sum(1 for f in perception.frames if f.ocr_blocks)
    if ocr_frames:
        lines.append(f"- **OCR:** on-screen text found in {ocr_frames} frames (inline below)")
    lines.append("")
    lines.append(
        "**Read each frame path below with the Read tool to view the image.** "
        "Frames are chronological; `t=MM:SS` is the absolute source timestamp."
    )
    lines.append("")
    for frame in perception.frames:
        line = (
            f"- `{frame.path}` (t={format_time(frame.timestamp_seconds)}, "
            f"scene={frame.scene_id}, reason={frame.reason})"
        )
        if frame.ocr_text:
            snippet = frame.ocr_text.replace("\n", " / ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            line += f"\n  - OCR: {snippet}"
        lines.append(line)
    return lines


def _language_note(transcript: Transcript) -> str:
    """`` · language: es (p=0.97, detected)``, or nothing when unknown."""
    if not transcript.language:
        return ""
    how = [transcript.language_source] if transcript.language_source else []
    if transcript.language_probability is not None:
        how.insert(0, f"p={transcript.language_probability:.2f}")
    detail = f" ({', '.join(how)})" if how else ""
    return f" · language: {transcript.language}{detail}"


def _transcript_lines(result: WatchResult) -> list[str]:
    lines = ["", "## Transcript", ""]
    transcript = result.transcript
    if transcript:
        lines.append(f"_Source: {transcript.source}{_language_note(transcript)}._")
        lines.append("")
        lines.append("```")
        lines.append(transcript.formatted())
        lines.append("```")
    else:
        lines.append(
            "_No transcript available — captions missing and no whisper rung succeeded. "
            "Proceed with frames only; run `watch-skill doctor` if local whisper should work._"
        )
    return lines


def render_report(result: WatchResult) -> str:
    """Full markdown report for a WatchResult."""
    lines = _header_lines(result)
    lines += _frames_lines(result)

    perception = result.perception
    if not result.focused and perception is not None and perception.sampling_sparse:
        mins = int(result.metadata.duration_seconds // 60)
        lines += [
            "",
            f"> **Warning:** frame coverage is sparse: one frame every "
            f"{perception.seconds_per_frame:.0f}s across this {mins}-minute video. "
            "Re-run focused (`--start/--end`) on the section that matters.",
        ]

    lines += _transcript_lines(result)
    lines += _sound_lines(result)
    lines += _warning_lines(result)
    lines += ["", "---", f"_Work dir: `{result.work_dir}` — frames live here._"]
    lines += [
        "", "## Machine summary", "",
        "```json", json.dumps(result.to_summary(), ensure_ascii=False, indent=2), "```",
    ]
    return "\n".join(lines)


def _sound_lines(result: WatchResult) -> list[str]:
    """Non-speech sound events, one per line, when detection ran."""
    if not result.audio_events:
        return []
    lines = ["", "## Sounds", ""]
    for event in result.audio_events:
        lines.append(
            f"- [{format_time(event.start)}–{format_time(event.end)}] {event.label} "
            f"({event.confidence:.2f})"
        )
    return lines


def _warning_lines(result: WatchResult) -> list[str]:
    """One line per degradation, with the fix that removes it."""
    if not result.warnings:
        return []
    lines = ["", "## Warnings", ""]
    for warning in result.warnings:
        line = f"- `{warning.get('code')}`: {warning.get('message')}"
        if warning.get("fix"):
            line += f" (fix: {warning['fix']})"
        lines.append(line)
    return lines
