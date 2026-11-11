"""Remove the words two overlapping audio spans both heard.

Live utterances carry the last 0.5 s of the previous one so a word sitting on
the boundary survives. The cost is that the words inside that half second are
recognised twice, once at the end of one piece and again at the start of the
next. This finds that repeat and trims it. ``cloud.merge_overlapping`` does
the same job for batch chunks at segment granularity.
"""
from __future__ import annotations

import unicodedata


def _fold(token: str) -> str:
    """Case- and punctuation-folded form of one word."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", token).casefold()
        if not unicodedata.category(ch).startswith(("P", "S"))
    )


def strip_overlap_echo(previous: str, current: str, max_words: int = 8) -> tuple[str, int]:
    """Drop the longest run of words that ends ``previous`` and starts ``current``.

    Words are compared case- and punctuation-folded, so ``"Total."`` echoes
    ``"total"``. At most ``max_words`` are considered: the overlap is half a
    second, and a longer match is far more likely to be a real repetition by
    the speaker than an echo. Returns the trimmed text (original spelling and
    punctuation of what is kept) and how many words were removed.
    """
    prev_words = [w for w in (_fold(t) for t in previous.split()) if w]
    cur_tokens = current.split()
    cur_folded = [_fold(t) for t in cur_tokens]
    limit = min(max_words, len(prev_words), len(cur_tokens))
    for n in range(limit, 0, -1):
        head = cur_folded[:n]
        if "" in head:
            continue
        if prev_words[-n:] == head:
            return " ".join(cur_tokens[n:]), n
    return current, 0
