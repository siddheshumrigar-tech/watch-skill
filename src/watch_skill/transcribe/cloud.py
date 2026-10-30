"""Cloud STT (Groq whisper-large-v3, then OpenAI) — OPT-IN ONLY.

Hard privacy rule, enforced here and tested: this module refuses to send
anything unless ``cloud_stt_enabled`` is true, and the only thing it ever
sends is the extracted mono-16 kHz audio file — never the video.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

from watch_skill.config import get_settings
from watch_skill.errors import TranscriptionError
from watch_skill.transcribe import audio as audio_mod
from watch_skill.transcribe.languages import normalize_language
from watch_skill.transcribe.types import Segment, Transcript, Word

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"
OPENAI_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"

MAX_ATTEMPTS = 4
RETRY_BASE_DELAY = 2.0


def _require_cloud_enabled(backend: str | None = None) -> None:
    """Two gates, both of which must open.

    The opt-in setting is the operator's standing consent; the policy is the
    rule for *this run*. Offline mode overrides a standing yes, which is the
    point of having a policy at all.
    """
    from watch_skill.policy import Channel, guard_egress  # noqa: PLC0415

    if not get_settings().cloud_stt_enabled:
        raise TranscriptionError(
            "cloud STT is disabled (privacy default)",
            code="transcribe.cloud_disabled",
            fix="set WATCHSKILL_CLOUD_STT_ENABLED=1 to opt in — only extracted "
            "audio is sent, never the video file",
        )
    guard_egress(Channel.AUDIO, provider=backend)


def pick_backend(preferred: str | None = None) -> tuple[str, str]:
    """Return (backend, api_key); Groq preferred (cheaper, faster)."""
    settings = get_settings()
    options = [
        ("groq", settings.groq_api_key),
        ("openai", settings.openai_api_key),
    ]
    if preferred:
        options = [opt for opt in options if opt[0] == preferred]
    for backend, key in options:
        if key is not None and key.get_secret_value().strip():
            return backend, key.get_secret_value().strip()
    raise TranscriptionError(
        "no cloud STT API key configured",
        code="transcribe.no_api_key",
        fix="set WATCHSKILL_GROQ_API_KEY (preferred) or WATCHSKILL_OPENAI_API_KEY",
    )


def _post_once(
    endpoint: str,
    api_key: str,
    model: str,
    audio_path: Path,
    word_timestamps: bool = False,
    language: str | None = None,
) -> dict:
    data: dict = {"model": model, "response_format": "verbose_json", "temperature": "0"}
    if word_timestamps:
        # without a granularity the API returns segments only
        data["timestamp_granularities[]"] = ["word", "segment"]
    if language:
        data["language"] = language
    with audio_path.open("rb") as fh:
        response = httpx.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                # default python UAs trip Groq's Cloudflare WAF (reference-learned)
                "User-Agent": "watch-skill/0.6 (+https://github.com/oxbshw/watch-skill)",
            },
            data=data,
            files={"file": (audio_path.name, fh, "audio/mpeg")},
            timeout=300.0,
        )
    response.raise_for_status()
    return response.json()


def _post_with_retry(
    endpoint: str,
    api_key: str,
    model: str,
    audio_path: Path,
    word_timestamps: bool = False,
    language: str | None = None,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return _post_once(
                endpoint, api_key, model, audio_path,
                word_timestamps=word_timestamps, language=language,
            )
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if 400 <= code < 500 and code != 429:
                raise TranscriptionError(
                    f"cloud STT rejected the request (HTTP {code})",
                    code="transcribe.cloud_rejected",
                    fix="401/403: check the STT API key; 413: the chunk is too "
                    "large (lower chunk length); details.body has the reason",
                    details={"status": code, "body": exc.response.text[:400]},
                ) from exc
            retry_after = exc.response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else RETRY_BASE_DELAY * 2**attempt
            last_error = exc
        except httpx.HTTPError as exc:
            delay = RETRY_BASE_DELAY * (attempt + 1)
            last_error = exc
        if attempt < MAX_ATTEMPTS - 1:
            print(f"[watch-skill] cloud STT retry in {delay:.1f}s…", file=sys.stderr)
            time.sleep(delay)
    raise TranscriptionError(
        f"cloud STT failed after {MAX_ATTEMPTS} attempts: {last_error}",
        code="transcribe.cloud_failed",
        fix="retry later or drop --cloud-stt — local whisper handles it offline",
    )


def _words_from(raw_words: list[dict] | None, offset: float) -> list[Word]:
    """verbose_json words (``{"word", "start", "end"}``) on the source timeline."""
    words: list[Word] = []
    for raw in raw_words or []:
        text = (raw.get("word") or "").strip()
        if text:
            words.append(Word(
                start=round(float(raw.get("start") or 0.0) + offset, 2),
                end=round(float(raw.get("end") or 0.0) + offset, 2),
                text=text,
            ))
    return words


def _attach_words(segments: list[Segment], words: list[Word]) -> None:
    """Give each word to the segment its start falls in (else the nearest before it)."""
    for word in words:
        home = segments[0]
        for seg in segments:
            if seg.start <= word.start:
                home = seg
            if seg.start <= word.start < seg.end:
                break
        home.words.append(word)


def _segments_from_response(data: dict, offset: float = 0.0) -> list[Segment]:
    out: list[Segment] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if text:
            out.append(
                Segment(
                    start=round(float(seg.get("start") or 0.0) + offset, 2),
                    end=round(float(seg.get("end") or 0.0) + offset, 2),
                    text=text,
                    words=_words_from(seg.get("words"), offset),
                )
            )
    if not out and (data.get("text") or "").strip():
        out.append(Segment(start=offset, end=offset, text=data["text"].strip()))
    top_level = _words_from(data.get("words"), offset)
    if out and top_level and not any(s.words for s in out):
        _attach_words(out, top_level)
    return out


def _language_from_response(data: dict) -> str | None:
    """verbose_json reports the language as an English name (``"english"``)."""
    reported = data.get("language")
    return normalize_language(reported) if isinstance(reported, str) else None


def merge_overlapping(chunks: list[list[Segment]]) -> list[Segment]:
    """Concatenate chunk transcripts, dropping duplicated overlap segments.

    Chunks overlap by ~2 s; a segment from chunk n+1 that starts before the
    previous chunk's last kept segment ends AND repeats its text is the
    overlap echo — skip it.
    """
    merged: list[Segment] = []
    for segments in chunks:
        for seg in segments:
            if merged and seg.start < merged[-1].end - 0.2:
                if seg.text == merged[-1].text or merged[-1].text.endswith(seg.text):
                    continue
            merged.append(seg)
    return merged


def transcribe_cloud(
    video_path: Path,
    work_dir: Path,
    preferred_backend: str | None = None,
    word_timestamps: bool = False,
    language: str | None = None,
) -> Transcript:
    """Extract audio, upload (chunked with overlap if large), merge segments.

    ``word_timestamps`` asks for word and segment granularities and attaches
    the returned words to their segments; when none come back the transcript
    carries a ``transcribe.cloud_no_word_timestamps`` warning. ``language``
    (a Whisper code) is sent as-is; otherwise the API's reported language is
    kept with ``language_source="detected"`` (verbose_json has no probability).
    """
    _require_cloud_enabled(preferred_backend)
    backend, api_key = pick_backend(preferred_backend)
    _require_cloud_enabled(backend)  # the allowlist applies to the CHOSEN backend
    endpoint, model = (
        (GROQ_ENDPOINT, GROQ_MODEL) if backend == "groq" else (OPENAI_ENDPOINT, OPENAI_MODEL)
    )

    audio_path = audio_mod.extract_audio(video_path, work_dir / "audio.mp3")
    size = audio_path.stat().st_size
    options = {"word_timestamps": word_timestamps, "language": language}
    reported: str | None = None
    if size <= audio_mod.MAX_UPLOAD_BYTES:
        data = _post_with_retry(endpoint, api_key, model, audio_path, **options)
        segments = _segments_from_response(data)
        reported = _language_from_response(data)
    else:
        duration = audio_mod.audio_duration(audio_path)
        plan = audio_mod.plan_chunks(duration, size)
        print(
            f"[watch-skill] audio {size / 1024**2:.0f} MB — {len(plan)} overlapping chunks…",
            file=sys.stderr,
        )
        pieces: list[list[Segment]] = []
        failures = 0
        for i, (chunk_path, offset) in enumerate(
            audio_mod.split_audio(audio_path, work_dir / "chunks", plan)
        ):
            try:
                data = _post_with_retry(endpoint, api_key, model, chunk_path, **options)
                pieces.append(_segments_from_response(data, offset=offset))
                reported = reported or _language_from_response(data)
            except TranscriptionError as exc:
                failures += 1
                print(f"[watch-skill] chunk {i + 1}/{len(plan)} failed: {exc.code}", file=sys.stderr)
        if failures == len(plan):
            raise TranscriptionError(
                "cloud STT failed on every chunk",
                code="transcribe.cloud_failed",
                fix="retry later or drop --cloud-stt — local whisper handles it offline",
            )
        segments = merge_overlapping(pieces)

    if not segments:
        raise TranscriptionError(
            "cloud STT returned no segments",
            code="transcribe.empty",
            fix="the audio may be silence or music only — that is a valid "
            "result; OCR and vision still index the video",
        )
    return _finish(
        Transcript(segments=segments, source=f"whisper-{backend}"),
        backend, word_timestamps, language, reported,
    )


def _finish(
    transcript: Transcript,
    backend: str,
    word_timestamps: bool,
    language: str | None,
    reported: str | None,
) -> Transcript:
    """Record the language and warn when requested word timings never came back."""
    if language:
        transcript.language, transcript.language_source = language, "forced"
    elif reported:
        transcript.language, transcript.language_source = reported, "detected"
    if word_timestamps and not any(s.words for s in transcript.segments):
        transcript.warnings.append({
            "code": "transcribe.cloud_no_word_timestamps",
            "message": f"word timestamps were requested but whisper-{backend} returned none",
            "fix": "segment timestamps are still exact to the segment; for word-level "
            "citations use local whisper (WATCHSKILL_LOCAL_WHISPER_ENABLED=1)",
        })
    return transcript
