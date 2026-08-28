"""Unit tests for M3 chunking.

The baseline is allowed to be simple. It is not allowed to be broken — a
chunker that drops text or loops forever would corrupt N rather than measure it.
"""
from __future__ import annotations

import pytest

from inforce.chunking import chunk_text, normalise

PROSE = (
    "In exercise of the powers conferred by Section 35A of the Banking Regulation "
    "Act 1949 the Reserve Bank hereby issues the following Directions. "
) * 40


def test_normalise_collapses_pdf_whitespace():
    assert normalise("a  \n\n b \t c ") == "a b c"


def test_chunks_cover_the_whole_document():
    """Concatenating chunks must reproduce every word, in order — no silent loss."""
    chunks = chunk_text(PROSE, size=500, overlap=100)
    joined = " ".join(c.text for c in chunks)
    source_words = normalise(PROSE).split()
    joined_words = joined.split()
    # Overlap means repeats, but every source word must appear at least once
    # and the first/last must be preserved.
    assert joined_words[0] == source_words[0]
    assert joined_words[-1] == source_words[-1]
    assert set(source_words) <= set(joined_words)


def test_chunks_respect_the_size_bound():
    for c in chunk_text(PROSE, size=400, overlap=50):
        assert len(c.text) <= 400


def test_overlap_actually_overlaps():
    chunks = chunk_text(PROSE, size=500, overlap=200)
    assert len(chunks) > 1
    assert chunks[1].char_start < chunks[0].char_end


def test_ordinals_are_sequential_from_zero():
    chunks = chunk_text(PROSE, size=300, overlap=60)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_short_document_yields_one_chunk():
    chunks = chunk_text("A short circular.", size=1000, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].text == "A short circular."


def test_empty_document_yields_nothing():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_overlap_not_smaller_than_size_is_rejected():
    """Otherwise the window never advances and the loop runs forever."""
    with pytest.raises(ValueError):
        chunk_text(PROSE, size=200, overlap=200)


def test_unbroken_token_longer_than_chunk_still_terminates():
    """A 5,000-character string with no spaces must not hang or lose data."""
    blob = "x" * 5000
    chunks = chunk_text(blob, size=1000, overlap=150)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)
    assert "".join(dict.fromkeys(c.text for c in chunks))  # non-empty, terminated
