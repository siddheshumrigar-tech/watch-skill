"""Speaker diarization: who spoke when, merged onto the transcript.

Optional and flag-gated (``WATCHSKILL_DIARIZATION_ENABLED`` or
``--diarize``). The heavy backend (pyannote.audio, torch) is an optional
extra — ``pip install watch-skill[diarize]`` — and needs a Hugging Face token
for the gated pyannote models. Everything downstream works on the pure
:class:`SpeakerTurn` contract, so tests and alternative backends never need
torch installed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from watch_skill.config import get_settings
from watch_skill.errors import DependencyError, TranscriptionError
from watch_skill.transcribe.types import Segment, Transcript, Word

_PIPELINE_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass
class SpeakerTurn:
    """One contiguous span attributed to one speaker."""

    start: float
    end: float
    speaker: str  # e.g. "SPEAKER_00"


def diarize_audio(audio_path: Path, hf_token: str | None = None) -> list[SpeakerTurn]:
    """Run pyannote speaker diarization over an audio file.

    Requires the ``diarize`` extra and a Hugging Face token accepted for the
    pyannote model. Raises a structured error otherwise so agents can act.
    """
    token = hf_token or _configured_token()
    try:
        from pyannote.audio import Pipeline  # noqa: PLC0415
    except ImportError as exc:
        raise TranscriptionError(
            "pyannote.audio is not installed",
            code="transcribe.diarization_unavailable",
            fix="pip install 'watch-skill[diarize]' (needs torch; ~2 GB)",
        ) from exc
    if not token:
        raise TranscriptionError(
            "diarization needs a Hugging Face token",
            code="transcribe.diarization_no_token",
            fix="set WATCHSKILL_HUGGINGFACE_TOKEN (accept the pyannote model terms on hf.co first)",
        )
    try:
        # pyannote.audio 4.x renamed use_auth_token= to token=
        pipeline = Pipeline.from_pretrained(_PIPELINE_MODEL, token=token)
        annotation = pipeline(str(audio_path))
    except Exception as exc:
        raise TranscriptionError(
            f"diarization failed: {exc}",
            code="transcribe.diarization_failed",
            fix="check the HF token has accepted the pyannote model terms",
            details={"model": _PIPELINE_MODEL, "audio": str(audio_path)},
        ) from exc
    return [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def _configured_token() -> str | None:
    token = get_settings().huggingface_token
    return token.get_secret_value().strip() if token else None


def _overlap(span: Segment | Word, turn: SpeakerTurn) -> float:
    return max(0.0, min(span.end, turn.end) - max(span.start, turn.start))


def _dominant_speaker(span: Segment | Word, turns: list[SpeakerTurn]) -> str | None:
    """The speaker whose turns overlap ``span`` the longest, or None."""
    best: tuple[float, str | None] = (0.0, None)
    for turn in turns:
        overlap = _overlap(span, turn)
        if overlap > best[0]:
            best = (overlap, turn.speaker)
    return best[1]


def _shift_turns(turns: list[SpeakerTurn], seconds: float) -> list[SpeakerTurn]:
    """Move window-relative turns onto the source timeline."""
    if not seconds:
        return turns
    return [SpeakerTurn(t.start + seconds, t.end + seconds, t.speaker) for t in turns]


def assign_speakers(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    """Label each segment, and each of its words, with the dominant-overlap speaker.

    Pure function over the SpeakerTurn contract — usable with any diarization
    backend. A segment that spans a turn change keeps its dominant speaker,
    while its words carry the speaker actually talking over each one. Spans
    with no overlapping turn keep ``speaker=None``.
    """
    labeled = [
        replace(
            seg,
            speaker=_dominant_speaker(seg, turns),
            words=[replace(w, speaker=_dominant_speaker(w, turns)) for w in seg.words],
        )
        for seg in transcript.segments
    ]
    return replace(transcript, segments=labeled)


def diarize_transcript(
    transcript: Transcript,
    video_path: Path,
    work_dir: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> Transcript:
    """Extract audio, diarize, and merge speakers onto ``transcript``.

    In focused mode (``start_seconds``/``end_seconds``) only that window is
    diarized and its turns are shifted back onto the source timeline the
    transcript uses. Diarization always extracts its own
    ``diarize-audio.mp3``: ``audio.mp3`` holds the window when local whisper
    wrote it and the whole file when cloud STT did.

    Degrades loudly (stderr) but never fatally: on any diarization failure the
    original transcript is returned unchanged — speech content beats labels.
    """
    if not transcript:
        return transcript
    try:
        from watch_skill.transcribe import audio as audio_mod  # noqa: PLC0415

        audio_path = audio_mod.extract_audio(
            video_path, work_dir / "diarize-audio.mp3",
            start_seconds=start_seconds, end_seconds=end_seconds,
        )
        turns = diarize_audio(audio_path)
    except (TranscriptionError, DependencyError) as exc:
        print(f"[watch-skill] diarization skipped ({exc.code}): {exc.message}", file=sys.stderr)
        return transcript
    return assign_speakers(transcript, _shift_turns(turns, start_seconds or 0.0))
