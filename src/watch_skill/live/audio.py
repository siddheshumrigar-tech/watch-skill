"""Live audio: capture, segment, transcribe — without losing speech.

Speech is continuous and unrepeatable. A dropped frame costs one sample of a
scene that is still there; a dropped half-second of audio costs a word that
will never be said again. So audio and vision share nothing but a clock:
separate ffmpeg output, separate queues, separate threads. Vision falling
behind cannot cost a syllable.

The format is fixed at the boundary — mono 16 kHz signed 16-bit PCM — because
that is what every local ASR this project can reach actually wants, and
resampling once at capture is cheaper and more predictable than making each
backend guess.

Utterances are assembled with **overlap**. Cutting audio into adjacent blocks
and transcribing each independently reliably loses the word sitting on the
boundary; carrying a little of the previous block into the next one is what
stops "checkout total" becoming "…total".
"""
from __future__ import annotations

import subprocess
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from watch_skill.errors import WatchSkillError

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes; signed 16-bit little-endian
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH

CHUNK_SECONDS = 1.0
"""How much audio is read from ffmpeg at a time. Small enough that a cancel
lands promptly, large enough that the read loop is not the bottleneck."""

UTTERANCE_SECONDS = 4.0
OVERLAP_SECONDS = 0.5


class AudioCaptureError(WatchSkillError):
    """Audio could not be captured from a source."""

    default_code = "live.audio_failed"


@dataclass
class AudioChunk:
    """One block of normalized PCM, with the clocks that let it be cited."""

    session_id: str
    seq: int
    pcm: bytes
    media_ts: float
    wall_ts: float
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    provenance: str = "capture"
    gap_before_seconds: float = 0.0
    """Silence we know we *missed*, as opposed to silence that was recorded.
    A transcript with an unmarked hole in it invites the reader to assume
    nobody spoke."""

    @property
    def sample_count(self) -> int:
        return len(self.pcm) // (SAMPLE_WIDTH * self.channels)

    @property
    def duration(self) -> float:
        return self.sample_count / float(self.sample_rate)

    @property
    def end_media_ts(self) -> float:
        return self.media_ts + self.duration

    def to_public(self) -> dict[str, Any]:
        """Metadata only — never the samples. Audio bytes in a tool result
        are a privacy problem and a context problem at the same time."""
        return {
            "seq": self.seq,
            "media_ts": round(self.media_ts, 3),
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.sample_count,
            "gap_before_seconds": round(self.gap_before_seconds, 3),
            "provenance": self.provenance,
        }


def source_has_audio(path: Path) -> bool:
    """Whether a media file carries an audio stream at all.

    Asked before starting an audio pipeline, so a silent video reports "no
    audio track" instead of an ASR backend that mysteriously never speaks.
    """
    from watch_skill.health.binaries import find_binary

    ffprobe = find_binary("ffprobe")
    if ffprobe is None:
        return False
    try:
        result = subprocess.run(
            [str(ffprobe), "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "audio" in (result.stdout or "")


class FfmpegAudioSource:
    """Normalized PCM read from an ffmpeg process, as it is produced.

    A raw stdout pipe rather than files: audio has no frame boundaries to
    respect, so a byte stream is the natural shape and avoids the write-then-
    poll latency the video path needs.
    """

    def __init__(self, args: list[str], *, label: str = "audio",
                 realtime: bool = True) -> None:
        self._args = args
        self.label = label
        self.realtime = realtime
        self._process: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._stderr_tail: list[str] = []

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        self._process = subprocess.Popen(
            self._args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )

        def drain() -> None:
            assert self._process is not None and self._process.stderr is not None
            for raw in self._process.stderr:
                self._stderr_tail.append(raw.decode("utf-8", "replace").rstrip())
                del self._stderr_tail[:-40]

        threading.Thread(target=drain, name=f"ws-{self.label}-stderr",
                         daemon=True).start()

    def chunks(self, session_id: str) -> Iterator[AudioChunk]:
        if self._process is None:
            self.start()
        assert self._process is not None and self._process.stdout is not None

        want = int(BYTES_PER_SECOND * CHUNK_SECONDS)
        seq, consumed = 0, 0
        while not self._stop.is_set():
            block = self._process.stdout.read(want)
            if not block:
                return
            seq += 1
            media_ts = consumed / float(BYTES_PER_SECOND)
            consumed += len(block)
            yield AudioChunk(
                session_id=session_id, seq=seq, pcm=block,
                media_ts=media_ts, wall_ts=time.time(),
            )

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr_tail)


def _ffmpeg() -> str:
    from watch_skill.health.binaries import require_binary

    return str(require_binary("ffmpeg"))


def file_replay_audio(target: str) -> FfmpegAudioSource:
    """Audio from a local file, paced at real time to match the video."""
    return FfmpegAudioSource(
        [_ffmpeg(), "-hide_banner", "-loglevel", "warning", "-re", "-i", target,
         "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        label="file-replay-audio",
    )


def stream_audio(target: str) -> FfmpegAudioSource:
    """Audio from a network stream — already real time, so no `-re`."""
    return FfmpegAudioSource(
        [_ffmpeg(), "-hide_banner", "-loglevel", "warning",
         "-rtsp_transport", "tcp", "-i", target,
         "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        label="stream-audio",
    )


# --- utterance assembly -------------------------------------------------------


@dataclass
class Utterance:
    """A span of audio worth handing to ASR, with its overlap accounted for."""

    session_id: str
    index: int
    pcm: bytes
    media_ts: float
    overlap_seconds: float = 0.0
    """Trailing audio carried into the NEXT utterance (0 when flushed)."""
    leading_overlap_seconds: float = 0.0
    """Audio at the START of this one that the previous utterance also had.
    0 for the first utterance and for the one after a flush. Whatever is
    recognised inside it was usually recognised once already."""

    @property
    def duration(self) -> float:
        return len(self.pcm) / float(BYTES_PER_SECOND)

    @property
    def end_media_ts(self) -> float:
        return self.media_ts + self.duration

    def write_wav(self, path: Path) -> Path:
        """Materialise as a WAV file, which is what local ASR backends take."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(CHANNELS)
            handle.setsampwidth(SAMPLE_WIDTH)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(self.pcm)
        return path


@dataclass
class UtteranceAssembler:
    """Accumulates chunks into overlapping spans for transcription."""

    session_id: str
    utterance_seconds: float = UTTERANCE_SECONDS
    overlap_seconds: float = OVERLAP_SECONDS
    _buffer: bytearray = field(default_factory=bytearray)
    _buffer_start_ts: float | None = None
    _index: int = 0
    _leading_overlap: float = 0.0
    def add(self, chunk: AudioChunk) -> Utterance | None:
        """Feed a chunk; get an utterance back once enough has accumulated."""
        if self._buffer_start_ts is None:
            self._buffer_start_ts = chunk.media_ts
        self._buffer.extend(chunk.pcm)
        if len(self._buffer) < int(BYTES_PER_SECOND * self.utterance_seconds):
            return None
        return self._emit()

    def flush(self, min_seconds: float = 0.4) -> Utterance | None:
        """Emit whatever is left when the stream ends.

        Without this the final utterance — often the one that matters, since
        it is where a session was stopped — would be silently discarded.
        """
        if len(self._buffer) < int(BYTES_PER_SECOND * min_seconds):
            return None
        return self._emit(keep_overlap=False)

    def _emit(self, keep_overlap: bool = True) -> Utterance:
        self._index += 1
        pcm = bytes(self._buffer)
        start = self._buffer_start_ts or 0.0
        overlap_bytes = int(BYTES_PER_SECOND * self.overlap_seconds)
        leading = self._leading_overlap
        if keep_overlap and overlap_bytes and len(pcm) > overlap_bytes:
            tail = pcm[-overlap_bytes:]
            self._buffer = bytearray(tail)
            self._buffer_start_ts = start + (len(pcm) - overlap_bytes) / BYTES_PER_SECOND
            self._leading_overlap = overlap_bytes / BYTES_PER_SECOND
        else:
            self._buffer = bytearray()
            self._buffer_start_ts = None
            self._leading_overlap = 0.0
        return Utterance(
            session_id=self.session_id, index=self._index, pcm=pcm,
            media_ts=start,
            overlap_seconds=self.overlap_seconds if keep_overlap else 0.0,
            leading_overlap_seconds=leading,
        )


class SpeechGate(Protocol):
    """Decides whether an utterance is worth waking the ASR model for."""

    name: str
    fallback_reason: str | None

    def is_speech(self, pcm: bytes) -> bool: ...


class EnergyGate:
    """The mean-amplitude gate. Free, but deaf to a quiet speaker."""

    name = "energy"

    def __init__(self, fallback_reason: str | None = None) -> None:
        self.fallback_reason = fallback_reason

    def is_speech(self, pcm: bytes) -> bool:
        return not is_probably_silent(pcm)


class SileroGate:
    """Silero VAD from faster-whisper, which ships its ONNX file in the wheel.

    It hears speech by its shape rather than its loudness, so a quiet voice on
    a laptop microphone gets through. If it cannot run, the first failure
    switches this gate to energy for good and keeps the reason, because a
    live session must degrade rather than stop hearing.
    """

    def __init__(self, vad: Any) -> None:
        self._vad = vad
        self._energy: EnergyGate | None = None
        self.fallback_reason: str | None = None

    @property
    def name(self) -> str:
        return "energy" if self._energy is not None else "silero"

    def is_speech(self, pcm: bytes) -> bool:
        if self._energy is not None:
            return self._energy.is_speech(pcm)
        try:
            import numpy as np  # noqa: PLC0415

            usable = pcm[: len(pcm) - (len(pcm) % SAMPLE_WIDTH)]
            audio = np.frombuffer(usable, dtype="<i2").astype(np.float32) / 32768.0
            if audio.size == 0:
                return False
            options = self._vad.VadOptions(min_silence_duration_ms=100, speech_pad_ms=30)
            spans = self._vad.get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001 - any failure degrades to energy
            self.fallback_reason = f"silero VAD failed, using the energy gate: {str(exc)[:160]}"
            self._energy = EnergyGate(self.fallback_reason)
            return self._energy.is_speech(pcm)
        return bool(spans)


def build_speech_gate(prefer: str | None = None) -> SpeechGate:
    """The gate the ``live_speech_gate`` setting asks for, or the best available.

    Never raises: without faster-whisper the answer is the energy gate plus a
    reason, which live status reports.
    """
    from watch_skill.config import get_settings  # noqa: PLC0415

    choice = prefer or get_settings().live_speech_gate
    if choice == "energy":
        return EnergyGate()
    import importlib  # noqa: PLC0415

    try:
        # import_module, not `from faster_whisper import vad`: the from-form
        # hands back None for a module blocked in sys.modules instead of
        # raising, and a None backend would only fail later, mid-session.
        vad = importlib.import_module("faster_whisper.vad")
    except Exception as exc:  # noqa: BLE001 - ImportError, or a broken native stack
        return EnergyGate(
            fallback_reason=f"silero VAD unavailable ({str(exc)[:120]}); "
            'install it with pip install "watch-skill[whisper]"'
        )
    return SileroGate(vad)


def is_probably_silent(pcm: bytes, threshold: int = 350) -> bool:
    """Cheap energy gate so silence never reaches a transcription model.

    Deliberately arithmetic rather than a VAD model: this runs on every
    utterance, and the expensive thing it protects against is exactly the
    model we would otherwise have to load to make the decision.
    """
    import array

    if not pcm:
        return True
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return True
    total = sum(abs(int(value)) for value in samples)
    return (total / len(samples)) < threshold
