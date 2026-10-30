"""Transcription data types shared across the ladder."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class Word:
    """One word with its own span.

    A segment can run for ten seconds, so "when was X said" answered from
    segment bounds is only ever accurate to the segment. These carry the
    real moment.
    """

    start: float
    end: float
    text: str
    # Set by diarization: the speaker talking over this word, which can differ
    # from the segment's dominant speaker when a segment spans a turn change.
    speaker: str | None = None

    def to_dict(self) -> dict:
        out = {"start": self.start, "end": self.end, "text": self.text}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out

    def to_public(self) -> dict:
        """Agent-facing shape: every key present, ``null`` when unknown."""
        return {"start": self.start, "end": self.end, "text": self.text,
                "speaker": self.speaker}


@dataclass
class Segment:
    """One timestamped span of speech (``speaker`` set when diarization ran)."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    # Populated only when word timestamps were requested and the backend
    # produced them; captions and cloud STT leave it empty.
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {"start": self.start, "end": self.end, "text": self.text}
        if self.speaker is not None:
            out["speaker"] = self.speaker
        if self.words:
            out["words"] = [w.to_dict() for w in self.words]
        return out

    def to_public(self) -> dict:
        """Agent-facing shape: every key present (``null`` / ``[]`` when empty).

        :meth:`to_dict` stays compact because the index stores it.
        """
        return {"start": self.start, "end": self.end, "text": self.text,
                "speaker": self.speaker, "words": [w.to_public() for w in self.words]}

    def word_at(self, seconds: float) -> Word | None:
        """The word being spoken at ``seconds``, when words are available."""
        for word in self.words:
            if word.start <= seconds <= word.end:
                return word
        return None

    def find_word(self, needle: str) -> Word | None:
        """First word matching ``needle`` (case- and punctuation-insensitive)."""
        target = "".join(ch for ch in needle.lower() if ch.isalnum())
        if not target:
            return None
        for word in self.words:
            if "".join(ch for ch in word.text.lower() if ch.isalnum()) == target:
                return word
        return None


@dataclass
class Transcript:
    """Full transcript plus where it came from.

    ``source`` is one of: ``captions``, ``whisper-local (<model>)``,
    ``whisper-groq``, ``whisper-openai``, or ``none``.

    ``language`` is an ISO 639-1 code (``yue``/``haw`` for the two Whisper
    languages without one). ``language_source`` says where it came from:
    ``detected`` (the model's guess, with ``language_probability``),
    ``forced`` (the caller named it) or ``captions`` (the subtitle file's
    name). ``warnings`` holds one ``{"code", "message", "fix"}`` per ladder
    rung that failed before this transcript was produced.
    """

    segments: list[Segment] = field(default_factory=list)
    source: str = "none"
    language: str | None = None
    language_probability: float | None = None
    language_source: str | None = None
    warnings: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def to_public(self) -> dict:
        """Agent-facing shape: every key present (``null`` / ``[]`` when empty)."""
        return {
            "source": self.source,
            "language": self.language,
            "language_probability": self.language_probability,
            "language_source": self.language_source,
            "segments": [s.to_public() for s in self.segments],
            "warnings": [dict(w) for w in self.warnings],
        }

    def offset(self, seconds: float) -> Transcript:
        """Shift all timestamps (window-extracted audio back to source time)."""
        if not seconds:
            return self
        return replace(
            self,
            segments=[
                Segment(
                    s.start + seconds,
                    s.end + seconds,
                    s.text,
                    s.speaker,
                    # shift the words too, or they point at the window's
                    # timeline while the segment points at the source's
                    [
                        Word(w.start + seconds, w.end + seconds, w.text, w.speaker)
                        for w in s.words
                    ],
                )
                for s in self.segments
            ],
        )

    def filter_range(self, start: float | None, end: float | None) -> Transcript:
        """Segments overlapping [start, end] (same semantics as the reference)."""
        if start is None and end is None:
            return self
        lo = start if start is not None else float("-inf")
        hi = end if end is not None else float("inf")
        kept = [s for s in self.segments if s.end >= lo and s.start <= hi]
        return replace(self, segments=kept)

    def formatted(self) -> str:
        """`[MM:SS] text` lines (speaker-prefixed when diarized) — the agent-facing rendering."""
        lines = []
        for seg in self.segments:
            start = int(seg.start)
            stamp = f"[{start // 60:02d}:{start % 60:02d}]"
            who = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"{stamp} {who}{seg.text}")
        return "\n".join(lines)
