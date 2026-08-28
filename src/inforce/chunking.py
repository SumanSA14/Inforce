"""M3 — deliberately naive chunking.

Fixed-size character windows with overlap, snapped to word boundaries. No
section awareness, no table handling, no sentence splitting. This is the
control group: its weaknesses are the point, and improving it would destroy
the measurement it exists to produce.

One thing it is NOT allowed to be is *unfairly* bad. A baseline that is broken
rather than simple would inflate N and make the headline result dishonest.
So: correct word-boundary handling, correct overlap, no silent truncation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CHUNK_CHARS = 1000
OVERLAP_CHARS = 150

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    char_start: int
    char_end: int


def normalise(text: str) -> str:
    """PDF extraction leaves ragged whitespace and hard line wraps."""
    return _WS.sub(" ", text).strip()


def chunk_text(
    text: str,
    *,
    size: int = CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size or the window never advances")

    cleaned = normalise(text)
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    n = len(cleaned)

    while start < n:
        end = min(start + size, n)

        # Snap the end back to a word boundary, unless that would cut the
        # chunk in half or we are at the end of the document.
        if end < n:
            space = cleaned.rfind(" ", start, end)
            if space > start + size // 2:
                end = space

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(Chunk(ordinal, piece, start, end))
            ordinal += 1

        if end >= n:
            break

        nxt = end - overlap
        # Guarantee forward progress even in pathological inputs.
        start = nxt if nxt > start else end

    return chunks
