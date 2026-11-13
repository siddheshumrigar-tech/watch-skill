"""Non-speech sound events: music, applause, laughter, alarms, silence (A10).

A transcript says what was said and nothing about what else was heard. A
fire alarm, a burst of applause or a minute of dead air is evidence an agent
asks about ("when did the audience laugh?"), and none of it reaches Whisper.

The classifier is pluggable (:class:`AudioEventClassifier`). The shipped
backend runs Google's YAMNet (521 AudioSet classes, about 4 MB, CPU) through
onnxruntime from a model file the operator points at: no weights are
downloaded, because their provenance and hash cannot be pinned offline here
(docs/audio-events.md says how to export one). Silence needs no model at all;
it is measured from RMS.
"""
from __future__ import annotations

import csv
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from watch_skill.errors import DependencyError, TranscriptionError

SAMPLE_RATE = 16_000
SILENCE_DBFS = -50.0
COARSE_LABELS = ("speech", "music", "applause", "laughter", "alarm", "silence")

# AudioSet display names fold onto the coarse labels by keyword, checked in
# this order: "Speech synthesizer" is speech, not music; "Smoke detector,
# smoke alarm" is an alarm. A class matching nothing here is not reported.
_COARSE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("silence", ("silence",)),
    ("speech", ("speech", "speaking", "conversation", "narration", "whisper", "babbling")),
    ("laughter", ("laugh", "giggle", "chuckle", "snicker", "chortle")),
    ("applause", ("applause", "clapping", "cheering")),
    ("alarm", ("alarm", "siren", "buzzer", "beep", "ringtone", "telephone bell", "smoke detector")),
    ("music", ("music", "singing", "song", "guitar", "piano", "drum", "violin", "orchestra",
               "choir", "strum", "bass")),
)

_EXTRA_FIX = ('pip install "watch-skill[audio-events]" '
              "(from a checkout: uv sync --extra audio-events)")


def coarse_label(raw_label: str) -> str | None:
    """The coarse label for an AudioSet display name, or None when unmapped."""
    lowered = raw_label.lower()
    for coarse, keywords in _COARSE_KEYWORDS:
        if any(word in lowered for word in keywords):
            return coarse
    return None


@dataclass
class AudioEvent:
    """One stretch of time with one coarse sound label."""

    start: float
    end: float
    label: str
    confidence: float
    raw_label: str | None = None
    backend: str = ""

    def to_public(self) -> dict[str, Any]:
        """Every key always present, for agent-facing JSON."""
        return {"start": round(self.start, 2), "end": round(self.end, 2), "label": self.label,
                "confidence": round(self.confidence, 3), "raw_label": self.raw_label,
                "backend": self.backend}


class AudioEventClassifier(Protocol):
    """Scores one window of 16 kHz mono float32 audio."""

    name: str

    def classify(self, samples: Any) -> list[tuple[str, float]]:
        """(raw_label, score) pairs, any order."""
        ...


class OnnxYamnetClassifier:
    """YAMNet exported to ONNX with a single rank-1 float32 waveform input.

    That is what tf2onnx produces from the TF-Hub model. Any other signature
    is refused up front rather than fed audio it would misread.
    """

    name = "yamnet-onnx"

    def __init__(self, model_path: Path, labels_path: Path, session: Any = None) -> None:
        self.labels = _read_labels(labels_path)
        if session is None:
            import onnxruntime  # noqa: PLC0415

            session = onnxruntime.InferenceSession(str(model_path),
                                                   providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        shape = list(getattr(inputs[0], "shape", []) or []) if len(inputs) == 1 else []
        if len(inputs) != 1 or len(shape) != 1 or "float" not in str(inputs[0].type):
            raise DependencyError(
                "the audio-events model does not take a single rank-1 float32 waveform",
                code="audio_events.unsupported_model",
                fix="export TF-Hub YAMNet with tf2onnx so its only input is the "
                "waveform; see docs/audio-events.md",
                details={"inputs": [(i.name, str(i.type), list(i.shape or [])) for i in inputs]},
            )
        self._session, self._input = session, inputs[0].name

    def classify(self, samples: Any) -> list[tuple[str, float]]:
        import numpy as np  # noqa: PLC0415

        scores = self._session.run(None, {self._input: np.asarray(samples, dtype=np.float32)})[0]
        mean = np.asarray(scores, dtype=np.float32).reshape(-1, len(self.labels)).mean(axis=0)
        top = np.argsort(mean)[::-1][:5]
        return [(self.labels[i], float(mean[i])) for i in top]


def _read_labels(path: Path) -> list[str]:
    """Display names from YAMNet's ``yamnet_class_map.csv`` (index,mid,display_name)."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row.get("display_name") or "" for row in rows]
    if not labels:
        raise DependencyError(
            f"no labels in {Path(path).name}",
            code="audio_events.model_not_configured",
            fix="point WATCHSKILL_AUDIO_EVENTS_LABELS_PATH at yamnet_class_map.csv; "
            "see docs/audio-events.md",
        )
    return labels


def build_classifier() -> AudioEventClassifier:
    """The configured classifier, or a structured reason there is none."""
    from watch_skill.config import get_settings  # noqa: PLC0415

    try:
        import numpy  # noqa: F401, PLC0415
        import onnxruntime  # noqa: F401, PLC0415
    except ImportError as exc:
        raise DependencyError(
            "sound-event detection needs onnxruntime and numpy",
            code="audio_events.missing_dependency", fix=_EXTRA_FIX,
        ) from exc
    settings = get_settings()
    model, labels = settings.audio_events_model_path, settings.audio_events_labels_path
    if not model or not labels or not Path(model).is_file() or not Path(labels).is_file():
        raise DependencyError(
            "no sound-event model is configured",
            code="audio_events.model_not_configured",
            fix="set WATCHSKILL_AUDIO_EVENTS_MODEL_PATH and "
            "WATCHSKILL_AUDIO_EVENTS_LABELS_PATH; see docs/audio-events.md",
            details={"model_path_set": bool(model), "labels_path_set": bool(labels)},
        )
    return OnnxYamnetClassifier(Path(model), Path(labels))


def _decode(audio_path: Path) -> Any:
    """16 kHz mono float32 samples of ``audio_path``, via ffmpeg."""
    import numpy as np  # noqa: PLC0415

    from watch_skill.health.binaries import require_binary  # noqa: PLC0415

    result = subprocess.run(
        [str(require_binary("ffmpeg")), "-hide_banner", "-loglevel", "error",
         "-i", str(Path(audio_path).resolve()), "-f", "s16le", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise TranscriptionError(
            "could not decode audio for sound-event detection",
            code="audio_events.decode_failed",
            fix="the audio track may be corrupt; re-acquire the source, and run "
            "`watch-skill doctor` to check ffmpeg",
            details={"stderr": result.stderr.decode("utf-8", "replace")[-500:]},
        )
    pcm = result.stdout[: len(result.stdout) - len(result.stdout) % 2]
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def extract_event_audio(media_path: Path, out_path: Path, start_seconds: float | None = None,
                        end_seconds: float | None = None) -> Path:
    """The (optionally windowed) audio of ``media_path`` as 16 kHz mono wav."""
    from watch_skill.health.binaries import require_binary  # noqa: PLC0415

    window: list[str] = []
    if start_seconds:
        window += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        window += ["-to", f"{end_seconds:.3f}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(require_binary("ffmpeg")), "-hide_banner", "-loglevel", "error", "-y", *window,
         "-i", str(Path(media_path).resolve()), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-c:a", "pcm_s16le", str(out_path.resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0 or not out_path.is_file():
        raise TranscriptionError(
            "could not extract audio for sound-event detection",
            code="audio_events.decode_failed",
            fix="the audio track may be corrupt; re-acquire the source, and run "
            "`watch-skill doctor` to check ffmpeg",
            details={"stderr": result.stderr[-500:]},
        )
    return out_path


def sound_events_for_watch(media_path: Path, work_dir: Path, start_seconds: float | None,
                           end_seconds: float | None) -> tuple[list[AudioEvent], list[dict]]:
    """Events for one watch window, plus warnings; never raises for a missing piece.

    A missing extra or model is a warning with its fix (D7), and silence is
    still reported because it needs no model.
    """
    warnings: list[dict] = []
    try:
        classifier: AudioEventClassifier | None = build_classifier()
    except DependencyError as exc:
        warnings.append({"code": exc.code, "message": exc.message, "fix": exc.fix})
        classifier = None
        try:
            import numpy  # noqa: F401, PLC0415
        except ImportError:
            return [], warnings
    try:
        wav = extract_event_audio(media_path, work_dir / "events-audio.wav",
                                  start_seconds, end_seconds)
        return detect_audio_events(wav, classifier, start_offset=start_seconds or 0.0), warnings
    except (TranscriptionError, DependencyError) as exc:
        warnings.append({"code": exc.code, "message": exc.message, "fix": exc.fix})
        return [], warnings


def _label_window(window: Any, classifier: AudioEventClassifier | None,
                  min_confidence: float) -> tuple[str, float, str | None, str] | None:
    """(label, confidence, raw_label, backend) for one window, or None."""
    import numpy as np  # noqa: PLC0415

    rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
    if 20.0 * math.log10(rms + 1e-12) < SILENCE_DBFS:
        return "silence", 1.0, None, "rms"
    if classifier is None:
        return None
    best: dict[str, tuple[float, str]] = {}
    for raw, score in classifier.classify(window):
        coarse = coarse_label(raw)
        if coarse and coarse != "silence" and score > best.get(coarse, (-1.0, ""))[0]:
            best[coarse] = (float(score), raw)
    if not best:
        return None
    label, (score, raw) = max(best.items(), key=lambda item: (item[1][0], item[0]))
    return (label, score, raw, classifier.name) if score >= min_confidence else None


def _merge(windows: Sequence[tuple[float, float, tuple[str, float, str | None, str]]]
           ) -> list[AudioEvent]:
    """Join overlapping or touching windows that share a label."""
    events: list[AudioEvent] = []
    for start, end, (label, confidence, raw, backend) in windows:
        last = events[-1] if events else None
        if last and last.label == label and start <= last.end + 1e-6:
            last.end = max(last.end, end)
            if confidence > last.confidence:
                last.confidence, last.raw_label = confidence, raw
            continue
        events.append(AudioEvent(start, end, label, confidence, raw, backend))
    return events


def detect_audio_events(
    audio_path: Path,
    classifier: AudioEventClassifier | None = None,
    *,
    window_seconds: float = 0.96,
    hop_seconds: float = 0.48,
    min_confidence: float | None = None,
    start_offset: float = 0.0,
) -> list[AudioEvent]:
    """Timestamped sound events in ``audio_path``, in time order.

    Without a classifier only silence is reported. ``start_offset`` puts a
    focused window's events on the source timeline.
    """
    from watch_skill.config import get_settings  # noqa: PLC0415

    threshold = get_settings().audio_events_min_confidence if min_confidence is None \
        else min_confidence
    samples = _decode(Path(audio_path))
    win, hop = int(window_seconds * SAMPLE_RATE), max(1, int(hop_seconds * SAMPLE_RATE))
    starts = list(range(0, len(samples) - win + 1, hop)) if len(samples) >= win else [0]
    if starts[-1] + win < len(samples):
        # The hop rarely divides the length; without this the last fraction
        # of a second (often where the applause is) is never listened to.
        starts.append(len(samples) - win)
    labelled = []
    for i in starts:
        window = samples[i:i + win]
        found = _label_window(window, classifier, threshold)
        if found is not None:
            begin = start_offset + i / SAMPLE_RATE
            labelled.append((round(begin, 2), round(begin + len(window) / SAMPLE_RATE, 2), found))
    return _merge(labelled)
