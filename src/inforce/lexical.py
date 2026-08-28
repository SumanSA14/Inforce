"""M9 — sparse (BM25) retrieval via SQLite FTS5.

Dense retrieval alone tops out at 53.8% recall on the golden set: a third of
questions never surface the correct document, which caps every downstream
filter. Exact terms are where embeddings are weakest — circular references
(`DOR.CRE.REC.402/07-01-001/2025-26`), rupee thresholds (`₹50,000`), acronyms
(`CRAR`, `MRR`, `ANBC`) — and those are exactly what regulatory questions turn
on.

FTS5 ships with Python's bundled SQLite, so this works despite huggingface.co
being unreachable (which blocks the cross-encoder reranker the plan assumed).
"""
from __future__ import annotations

import re
import sqlite3

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    content='chunk',
    content_rowid='chunk_id',
    tokenize='unicode61 remove_diacritics 2'
);
"""

# FTS5 MATCH has its own query syntax, so raw user text is not safe to pass in:
# a stray quote, hyphen or parenthesis is a syntax error, not a search for those
# characters. Reduce the question to bare alphanumeric terms and OR them.
_TERM = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset(
    "the a an of to for in on is are was were be been must shall may can could "
    "what which how when who whom why does do did done and or not with that this "
    "these those from by at as it its their there has have had if then than".split()
)


def build(conn: sqlite3.Connection, *, log=print) -> int:
    """(Re)build the FTS index from the chunk table. Idempotent.

    This is an *external-content* table (`content='chunk'`), so it stores only
    the inverted index and reads the text from `chunk`. That means it must not
    be populated or cleared with ordinary INSERT/DELETE — doing so desynchronises
    the index from its content table and SQLite reports "database disk image is
    malformed". The `rebuild` command is the supported way, and it handles both
    first build and refresh.
    """
    conn.executescript(FTS_SCHEMA)
    conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
    log(f"FTS index: {n:,} chunks")
    return n


def is_built(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
    ).fetchone()
    if not row:
        return False
    return conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] > 0


def to_match_query(question: str, *, min_len: int = 2) -> str:
    terms = [t for t in _TERM.findall(question or "")
             if len(t) >= min_len and t.lower() not in _STOP]
    # Quote each term so FTS5 treats it as a literal, never as an operator.
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def search(conn: sqlite3.Connection, question: str, *, limit: int = 100) -> list[int]:
    """Chunk ids ranked by BM25, best first. Empty list if the query reduces to
    nothing (a question made entirely of stopwords)."""
    match = to_match_query(question)
    if not match:
        return []
    if not is_built(conn):
        return []
    rows = conn.execute(
        """SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?
           ORDER BY bm25(chunk_fts) LIMIT ?""",
        (match, limit),
    ).fetchall()
    return [r[0] for r in rows]


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, k: int = 60, limit: int | None = None
) -> list[tuple[int, float]]:
    """Standard RRF: score = sum over rankers of 1 / (k + rank).

    Rank-based rather than score-based, so BM25 scores (unbounded, negative in
    SQLite) and cosine similarities (0..1) never need to be put on a common
    scale — which is the usual source of silent weighting bugs in hybrid search.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    out = sorted(scores.items(), key=lambda kv: -kv[1])
    return out[:limit] if limit else out
