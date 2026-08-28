"""M3 — build the naive vector index.

Storage is SQLite plus a NumPy matrix rather than Postgres + pgvector, because
Docker is not installed on this machine and M3's baseline deliberately does no
date filtering — which is precisely where pgvector earns its place. The plan's
argument for pgvector holds and bites at M6/M7; see README for the trigger to
switch. The scraper and chunk table remain the source of truth, so moving is a
re-index, not a migration.

Brute force is honest here: ~50k chunks x 768 dims is a 150 MB float32 matrix
and a query is one matmul. ANN indexing would be premature.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import chunking, config, embedding

INDEX_DIR = config.DATA_DIR / "index"
MATRIX_PATH = INDEX_DIR / "embeddings.npy"
IDS_PATH = INDEX_DIR / "chunk_ids.npy"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


@dataclass
class IndexStats:
    docs_chunked: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    matrix_rows: int = 0


def chunk_documents(conn: sqlite3.Connection, *, limit: int | None = None, log=print) -> tuple[int, int]:
    """Chunk any fetched document that has no chunks yet. Idempotent."""
    sql = """
        SELECT d.doc_key, d.body_text
        FROM document d
        LEFT JOIN chunk c ON c.doc_key = d.doc_key
        WHERE d.fetch_status = 'ok' AND d.body_len > 0 AND c.chunk_id IS NULL
        GROUP BY d.doc_key
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    rows = conn.execute(sql).fetchall()
    log(f"documents needing chunking: {len(rows):,}")

    total = 0
    for i, row in enumerate(rows, 1):
        pieces = chunking.chunk_text(row["body_text"])
        conn.executemany(
            """INSERT OR IGNORE INTO chunk
               (doc_key, ordinal, text, char_start, char_end)
               VALUES (?,?,?,?,?)""",
            [(row["doc_key"], p.ordinal, p.text, p.char_start, p.char_end) for p in pieces],
        )
        total += len(pieces)
        if i % 200 == 0:
            conn.commit()
            log(f"  chunked {i:,}/{len(rows):,} docs -> {total:,} chunks")
    conn.commit()
    return len(rows), total


def embed_chunks(
    conn: sqlite3.Connection, *, limit: int | None = None, batch_size: int = 256, log=print
) -> int:
    """Embed chunks that have no vector yet. Resumable — kill it any time."""
    # Breadth-first: chunk 0 of every document before chunk 1 of any. An
    # interrupted or partial build then covers the whole corpus shallowly
    # rather than a prefix of documents deeply — far more useful, since the
    # crawl is still adding documents and this gets re-run repeatedly.
    sql = """SELECT chunk_id, text FROM chunk
             WHERE embedding IS NULL ORDER BY ordinal, chunk_id"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    pending = conn.execute(sql).fetchall()
    log(f"chunks needing embedding: {len(pending):,}")
    if not pending:
        return 0

    done = 0
    for i in range(0, len(pending), batch_size):
        block = pending[i : i + batch_size]
        vectors = embedding.embed_documents([r["text"] for r in block])
        conn.executemany(
            "UPDATE chunk SET embedding = ?, embed_model = ? WHERE chunk_id = ?",
            [
                (vectors[j].tobytes(), embedding.EMBED_MODEL, block[j]["chunk_id"])
                for j in range(len(block))
            ],
        )
        conn.commit()
        done += len(block)
        log(f"  embedded {done:,}/{len(pending):,}")
    return done


def build_matrix(conn: sqlite3.Connection, *, log=print) -> int:
    """Materialise the searchable matrix from the chunk table."""
    rows = conn.execute(
        "SELECT chunk_id, embedding FROM chunk WHERE embedding IS NOT NULL ORDER BY chunk_id"
    ).fetchall()
    if not rows:
        log("no embedded chunks — nothing to build")
        return 0

    ids = np.asarray([r["chunk_id"] for r in rows], dtype=np.int64)
    matrix = np.vstack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    ).astype(np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MATRIX_PATH, matrix)
    np.save(IDS_PATH, ids)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "model": embedding.EMBED_MODEL,
                "dims": int(matrix.shape[1]),
                "rows": int(matrix.shape[0]),
                "chunk_chars": chunking.CHUNK_CHARS,
                "overlap_chars": chunking.OVERLAP_CHARS,
                "normalised": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"matrix: {matrix.shape} -> {MATRIX_PATH.name} ({matrix.nbytes/1024/1024:,.0f} MB)")
    return int(matrix.shape[0])


def load_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def matrix_drift(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (rows_in_matrix, embedded_chunks_in_db).

    These diverge whenever embedding is interrupted, or new documents are
    embedded without rebuilding. Querying the stale matrix would silently
    search an old, smaller corpus and return confident, wrong results — so
    callers must check rather than assume.
    """
    manifest = load_manifest()
    rows = int(manifest["rows"]) if manifest else 0
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunk WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    return rows, embedded
