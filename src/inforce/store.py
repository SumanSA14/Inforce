"""SQLite storage for M1.

SQLite rather than Postgres deliberately: M1's output is a plain relational
table with no vectors, so requiring Docker here would add setup friction to
the milestone that must not be skipped. The schema is written in portable SQL
and the scraper — not the database — is the source of truth, so re-running
against Postgres at M3 is a re-ingest, not a migration.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone

from . import config
from .annex import AnnexRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS annex_entry (
    natural_key         TEXT PRIMARY KEY,
    batch               TEXT NOT NULL,
    source_table_index  INTEGER NOT NULL,
    serial_no           INTEGER,
    circular_number     TEXT,
    subject             TEXT NOT NULL,
    department          TEXT,
    circular_date_raw   TEXT NOT NULL,
    circular_date       TEXT,
    rbi_doc_id          INTEGER,
    source_url          TEXT,
    deferred_repeal     INTEGER NOT NULL DEFAULT 0,
    dup_index           INTEGER NOT NULL DEFAULT 0,
    withdrawn_on        TEXT,
    status              TEXT NOT NULL DEFAULT 'withdrawn',
    first_seen_at       TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_annex_doc_id      ON annex_entry(rbi_doc_id);
CREATE INDEX IF NOT EXISTS idx_annex_batch       ON annex_entry(batch);
CREATE INDEX IF NOT EXISTS idx_annex_circ_number ON annex_entry(circular_number);
CREATE INDEX IF NOT EXISTS idx_annex_date        ON annex_entry(circular_date);

CREATE TABLE IF NOT EXISTS document (
    doc_key       TEXT PRIMARY KEY,
    source_kind   TEXT NOT NULL,   -- notification | master_direction
    status        TEXT NOT NULL,   -- withdrawn | in_force
    rbi_doc_id    INTEGER,
    url           TEXT NOT NULL,
    title         TEXT,
    rbi_ref       TEXT,
    dept_ref      TEXT,
    doc_date      TEXT,
    category      TEXT,
    updated_as_on TEXT,
    pdf_url       TEXT,
    pdf_size_kb   INTEGER,
    body_text     TEXT,
    body_len      INTEGER,
    body_source   TEXT,             -- html | pdf
    content_sha1  TEXT,
    cache_path    TEXT,
    fetch_status  TEXT NOT NULL,   -- ok | error | thin
    fetch_error   TEXT,
    fetched_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doc_status   ON document(status);
CREATE INDEX IF NOT EXISTS idx_doc_kind     ON document(source_kind);
CREATE INDEX IF NOT EXISTS idx_doc_rbi_id   ON document(rbi_doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_deptref  ON document(dept_ref);
CREATE INDEX IF NOT EXISTS idx_doc_fetch    ON document(fetch_status);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_key     TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    embedding   BLOB,
    embed_model TEXT,
    UNIQUE(doc_key, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunk_doc     ON chunk(doc_key);
CREATE INDEX IF NOT EXISTS idx_chunk_pending ON chunk(embedding) WHERE embedding IS NULL;

-- M4: the golden question set. Hand-written, reviewed, and the gate on N.
CREATE TABLE IF NOT EXISTS question (
    qid              TEXT PRIMARY KEY,
    text             TEXT NOT NULL,
    topic            TEXT,
    category         TEXT,            -- entity category, e.g. 'Commercial Banks'
    answerable       INTEGER NOT NULL DEFAULT 1,
    expected_doc_key TEXT,            -- the in-force document that should answer it
    expected_quote   TEXT,            -- supporting span, for manual verification
    trap_doc_keys    TEXT,            -- JSON array: withdrawn docs likely to be retrieved
    difficulty       TEXT,            -- easy | medium | hard
    notes            TEXT,
    reviewed         INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_question_answerable ON question(answerable);
CREATE INDEX IF NOT EXISTS idx_question_expected   ON question(expected_doc_key);

-- Candidate confusable pairs: an in-force document and a withdrawn one that
-- cover the same ground. These are where retrieval will go wrong, so they are
-- the highest-value seeds for questions.
CREATE TABLE IF NOT EXISTS confusable_pair (
    in_force_doc_key  TEXT NOT NULL,
    withdrawn_doc_key TEXT NOT NULL,
    max_similarity    REAL NOT NULL,
    mean_similarity   REAL NOT NULL,
    matches           INTEGER NOT NULL,
    computed_at       TEXT NOT NULL,
    PRIMARY KEY (in_force_doc_key, withdrawn_doc_key)
);

CREATE INDEX IF NOT EXISTS idx_pair_sim ON confusable_pair(max_similarity DESC);

CREATE TABLE IF NOT EXISTS scrape_run (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    source_url    TEXT NOT NULL,
    from_cache    INTEGER NOT NULL,
    content_sha1  TEXT NOT NULL,
    rows_parsed   INTEGER NOT NULL,
    rows_inserted INTEGER NOT NULL,
    rows_updated  INTEGER NOT NULL,
    warnings      TEXT
);
"""

UPSERT = """
INSERT INTO annex_entry (
    natural_key, batch, source_table_index, serial_no, circular_number,
    subject, department, circular_date_raw, circular_date, rbi_doc_id,
    source_url, deferred_repeal, dup_index, withdrawn_on, status,
    first_seen_at, last_seen_at
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'withdrawn',?,?)
ON CONFLICT(natural_key) DO UPDATE SET
    serial_no       = excluded.serial_no,
    circular_number = excluded.circular_number,
    rbi_doc_id      = excluded.rbi_doc_id,
    source_url      = excluded.source_url,
    deferred_repeal = excluded.deferred_repeal,
    dup_index       = excluded.dup_index,
    withdrawn_on    = excluded.withdrawn_on,
    circular_date   = excluded.circular_date,
    last_seen_at    = excluded.last_seen_at
"""


class DuplicateKeyError(RuntimeError):
    """Two parsed rows produced the same natural key. Storing them would
    silently drop one, so fail loudly instead."""


def connect(db_path=None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # A long crawl holds the write lock in bursts; let a concurrent reader
    # (e.g. `status` in another shell) wait rather than fail immediately.
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_rows(conn: sqlite3.Connection, rows: Iterable[AnnexRow]) -> tuple[int, int]:
    """Idempotent upsert. Returns (inserted, updated)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = list(rows)
    keys = [r.natural_key for r in rows]

    if len(keys) != len(set(keys)):
        dupes = [k for k, n in Counter(keys).items() if n > 1][:5]
        raise DuplicateKeyError(
            f"{len(keys) - len(set(keys))} colliding natural keys "
            f"(e.g. {dupes}) — storing would drop rows"
        )

    existing: set[str] = set()
    cur = conn.cursor()
    for i in range(0, len(keys), 500):
        chunk = keys[i : i + 500]
        placeholders = ",".join("?" * len(chunk))
        cur.execute(
            f"SELECT natural_key FROM annex_entry WHERE natural_key IN ({placeholders})",
            chunk,
        )
        existing.update(r[0] for r in cur.fetchall())

    payload = [
        (
            r.natural_key,
            r.batch,
            r.source_table_index,
            r.serial_no,
            r.circular_number,
            r.subject,
            r.department,
            r.circular_date_raw,
            r.circular_date.isoformat() if r.circular_date else None,
            r.rbi_doc_id,
            r.source_url,
            1 if r.deferred_repeal else 0,
            r.dup_index,
            r.withdrawn_on,
            now,
            now,
        )
        for r in rows
    ]
    cur.executemany(UPSERT, payload)
    conn.commit()

    updated = len(existing)
    inserted = len(rows) - updated
    return inserted, updated


def record_run(
    conn: sqlite3.Connection,
    *,
    source_url: str,
    from_cache: bool,
    content_sha1: str,
    rows_parsed: int,
    rows_inserted: int,
    rows_updated: int,
    warnings: str | None,
) -> None:
    conn.execute(
        """INSERT INTO scrape_run (started_at, source_url, from_cache, content_sha1,
                                   rows_parsed, rows_inserted, rows_updated, warnings)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_url,
            1 if from_cache else 0,
            content_sha1,
            rows_parsed,
            rows_inserted,
            rows_updated,
            warnings,
        ),
    )
    conn.commit()


DOC_UPSERT = """
INSERT INTO document (
    doc_key, source_kind, status, rbi_doc_id, url, title, rbi_ref, dept_ref,
    doc_date, category, updated_as_on, pdf_url, pdf_size_kb, body_text,
    body_len, body_source, content_sha1, cache_path, fetch_status, fetch_error,
    fetched_at
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(doc_key) DO UPDATE SET
    title         = excluded.title,
    rbi_ref       = excluded.rbi_ref,
    dept_ref      = excluded.dept_ref,
    doc_date      = excluded.doc_date,
    category      = excluded.category,
    updated_as_on = excluded.updated_as_on,
    pdf_url       = excluded.pdf_url,
    pdf_size_kb   = excluded.pdf_size_kb,
    body_text     = excluded.body_text,
    body_len      = excluded.body_len,
    body_source   = excluded.body_source,
    content_sha1  = excluded.content_sha1,
    cache_path    = excluded.cache_path,
    fetch_status  = excluded.fetch_status,
    fetch_error   = excluded.fetch_error,
    fetched_at    = excluded.fetched_at
"""


def upsert_document(conn: sqlite3.Connection, **f) -> None:
    conn.execute(
        DOC_UPSERT,
        (
            f["doc_key"], f["source_kind"], f["status"], f.get("rbi_doc_id"),
            f["url"], f.get("title"), f.get("rbi_ref"), f.get("dept_ref"),
            f.get("doc_date"), f.get("category"), f.get("updated_as_on"),
            f.get("pdf_url"), f.get("pdf_size_kb"), f.get("body_text"),
            f.get("body_len"), f.get("body_source"), f.get("content_sha1"),
            f.get("cache_path"), f["fetch_status"], f.get("fetch_error"),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def pending_notification_ids(conn: sqlite3.Connection, limit: int | None = None) -> list[int]:
    """Annex rows with a stable doc ID that have not yet been fetched cleanly.

    Ordered newest-first: recent circulars are the ones most likely to be
    confused with current Master Directions, so they matter most for N.
    """
    sql = """
        SELECT DISTINCT a.rbi_doc_id
        FROM annex_entry a
        LEFT JOIN document d
               ON d.rbi_doc_id = a.rbi_doc_id AND d.source_kind = 'notification'
        WHERE a.rbi_doc_id IS NOT NULL
          AND (d.doc_key IS NULL OR d.fetch_status <> 'ok')
        ORDER BY a.circular_date DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(sql).fetchall()]


def document_summary(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT source_kind, status, fetch_status,
                  COUNT(*)          AS n,
                  SUM(body_len)     AS total_chars,
                  AVG(body_len)     AS avg_chars
           FROM document
           GROUP BY source_kind, status, fetch_status
           ORDER BY source_kind, fetch_status"""
    ).fetchall()


def summary(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT batch,
                  COUNT(*)                                        AS rows,
                  SUM(rbi_doc_id IS NOT NULL)                     AS linked,
                  SUM(circular_number IS NOT NULL)                AS with_circ_no,
                  SUM(circular_date IS NULL)                      AS unparsed_dates,
                  SUM(deferred_repeal)                            AS deferred,
                  MIN(circular_date)                              AS earliest,
                  MAX(circular_date)                              AS latest
           FROM annex_entry
           GROUP BY batch
           ORDER BY batch"""
    ).fetchall()
