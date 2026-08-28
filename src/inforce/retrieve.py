"""M3 — naive retrieval. Top-k dense cosine, and nothing else.

Explicitly absent, by design:
  * no hybrid / BM25            (M9)
  * no reranking                (M9)
  * no as-of-date filtering     (M7 — this is the whole experiment)
  * no query rewriting or decomposition

This module is the control group. Every retrieved chunk is tagged with whether
its source document is withdrawn, which is what M5 counts to produce N.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from . import embedding, entities, index, lexical, temporal


@dataclass
class Hit:
    rank: int
    score: float
    chunk_id: int
    doc_key: str
    text: str
    title: str | None
    status: str            # withdrawn | in_force
    source_kind: str
    doc_date: str | None
    withdrawn_on: str | None
    rbi_ref: str | None
    url: str | None
    valid_from: str | None = None
    valid_to: str | None = None
    validity_certain: bool = True
    known_from: str | None = None

    @property
    def is_withdrawn(self) -> bool:
        """Present-tense status. For 'was this in force on date D', use
        `validity_on` — a document repealed in 2025 was perfectly valid in 2022,
        and reporting it as repealed for a historical query is wrong."""
        return self.status == "withdrawn"

    def validity_on(self, as_of: str) -> str:
        return temporal.validity_at(
            self.valid_from, self.valid_to, self.status,
            self.validity_certain, as_of, self.known_from,
        )


class IndexNotBuilt(RuntimeError):
    pass


class IndexStale(RuntimeError):
    """The on-disk matrix does not match the embedded chunks in the database.

    Happens whenever an embedding run is interrupted, or new documents are
    embedded without rebuilding the matrix. Searching anyway would quietly
    query an older, smaller corpus — the results would look completely normal
    and be wrong, which is worse than an error.
    """


_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def load_index() -> tuple[np.ndarray, np.ndarray]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not index.MATRIX_PATH.exists():
        raise IndexNotBuilt("no index — run: python -m inforce.cli index")
    matrix = np.load(index.MATRIX_PATH, mmap_mode="r")
    ids = np.load(index.IDS_PATH)
    _CACHE = (matrix, ids)
    return _CACHE


def check_fresh(conn: sqlite3.Connection) -> None:
    """Refuse to search a matrix that has drifted from the database."""
    rows, embedded = index.matrix_drift(conn)
    if rows != embedded:
        raise IndexStale(
            f"index matrix has {rows:,} rows but {embedded:,} chunks are embedded "
            f"({embedded - rows:+,}). Rebuild with:\n"
            f"    python -m inforce.cli index --rebuild-matrix\n"
            f"or pass --allow-stale to search the older matrix deliberately."
        )


_MASK_CACHE: dict[str, np.ndarray] = {}
_ENTITY_CACHE: dict[str, np.ndarray] = {}


def entity_mask(conn: sqlite3.Connection, category: str) -> np.ndarray:
    """Mask over index rows: True unless the chunk's document belongs to a
    *different class of regulated entity*.

    Three kinds of document are kept:

    * the target entity's own documents — the obvious case;
    * documents with no category recorded — missing metadata is not evidence of
      a mismatch, and excluding them would let the entity filter quietly double
      as a second time filter (withdrawn circulars carry no category at all);
    * documents filed under a **functional department** rather than an entity
      type (Financial Inclusion, Financial Market, Issuer of Currency…).

    That last case is a fix, not a nicety. RBI's index mixes entity types with
    departments, so a priority-sector question mentioning "commercial bank"
    would otherwise be scoped to Commercial Banks and exclude the Financial
    Inclusion document that actually answers it. Only sibling *entity* documents
    are genuine competitors; department documents are orthogonal.
    """
    if category in _ENTITY_CACHE:
        return _ENTITY_CACHE[category]

    entity_types = set(entities.categories())
    _, ids = load_index()
    doc_cat = {
        r["doc_key"]: r["category"]
        for r in conn.execute("SELECT doc_key, category FROM document")
    }
    chunk_doc = {
        r["chunk_id"]: r["doc_key"]
        for r in conn.execute(
            "SELECT chunk_id, doc_key FROM chunk WHERE embedding IS NOT NULL")
    }

    def keep(c: str | None) -> bool:
        if c is None or c == category:
            return True
        return c not in entity_types  # a department, not a rival entity

    mask = np.fromiter(
        (keep(doc_cat.get(chunk_doc.get(int(cid)))) for cid in ids),
        dtype=bool, count=len(ids),
    )
    _ENTITY_CACHE[category] = mask
    return mask


def in_force_mask(conn: sqlite3.Connection, as_of: str) -> np.ndarray:
    """Boolean mask over index rows: True where the chunk's source document was
    in force on `as_of`.

    This is the whole of M7's mechanism. Documents whose validity is UNKNOWN at
    that date are excluded: citing something that may already have been repealed
    is the failure being fixed, so the conservative choice is the correct one.
    """
    if as_of in _MASK_CACHE:
        return _MASK_CACHE[as_of]

    _, ids = load_index()
    validity = {
        r["doc_key"]: temporal.validity_at(
            r["valid_from"], r["valid_to"], r["status"],
            bool(r["validity_certain"]), as_of, r["known_from"])
        for r in conn.execute(
            """SELECT doc_key, valid_from, valid_to, status, validity_certain,
                      known_from FROM document""")
    }
    chunk_doc = {
        r["chunk_id"]: r["doc_key"]
        for r in conn.execute(
            "SELECT chunk_id, doc_key FROM chunk WHERE embedding IS NOT NULL")
    }
    mask = np.fromiter(
        (validity.get(chunk_doc.get(int(cid)), temporal.UNKNOWN) == temporal.IN_FORCE
         for cid in ids),
        dtype=bool, count=len(ids),
    )
    _MASK_CACHE[as_of] = mask
    return mask


def search(
    conn: sqlite3.Connection,
    question: str,
    *,
    k: int = 8,
    allow_stale: bool = False,
    as_of: str | None = None,
    entity: str | None = None,
    query_vec: np.ndarray | None = None,
    retrieval: str = "dense",
    candidate_pool: int = 100,
) -> list[Hit]:
    """Retrieve top-k.

    `retrieval` is "dense" (the M3 baseline) or "hybrid" (dense + BM25 fused by
    reciprocal rank). Filters apply to both rankers before fusion, so hybrid
    never smuggles a repealed document past the as-of-date filter.

    `as_of`  restricts to documents in force on that date (M7).
    `entity` restricts to one class of regulated entity.

    The two filters are independent and compose; either can be used alone.
    """
    if not allow_stale:
        check_fresh(conn)
    matrix, ids = load_index()
    # `query_vec` lets a caller embed once and run several filter configurations
    # against the same vector — the demo compares two modes per request.
    query = embedding.embed_query(question) if query_vec is None else query_vec

    # Vectors are L2-normalised at write time, so cosine == dot product.
    scores = np.asarray(matrix @ query)

    filtered = False
    mask = None
    if as_of is not None:
        mask = in_force_mask(conn, as_of)
        filtered = True
    if entity is not None:
        em = entity_mask(conn, entity)
        mask = em if mask is None else (mask & em)
        filtered = True

    if filtered:
        if mask is None or not mask.any():
            return []
        # -inf rather than deletion keeps index positions aligned with `ids`.
        scores = np.where(mask, scores, -np.inf)

    available = int(np.isfinite(scores).sum()) if filtered else scores.shape[0]
    if available <= 0:
        return []

    if retrieval == "hybrid":
        # Fail loudly rather than quietly returning dense results. Without this
        # a missing FTS index makes hybrid produce numbers identical to the
        # baseline, with nothing appearing to go wrong — the exact silent
        # degradation this project exists to measure.
        if not lexical.is_built(conn):
            raise IndexNotBuilt(
                "hybrid retrieval requested but the BM25 index is missing — "
                "run: python -m inforce.cli index"
            )
        # Pull a deeper candidate pool from each ranker, then fuse by rank.
        pool = min(candidate_pool, available)
        dense_idx = np.argpartition(-scores, pool - 1)[:pool]
        dense_idx = dense_idx[np.argsort(-scores[dense_idx])]
        dense_ranking = [int(ids[i]) for i in dense_idx]

        sparse_ranking = lexical.search(conn, question, limit=candidate_pool)
        if filtered and sparse_ranking:
            allowed = {int(ids[i]) for i in np.flatnonzero(mask)}
            sparse_ranking = [c for c in sparse_ranking if c in allowed]

        fused = lexical.reciprocal_rank_fusion(
            [dense_ranking, sparse_ranking], limit=min(k, available)
        )
        chunk_ids = [cid for cid, _ in fused]
        score_by_chunk = dict(fused)
    else:
        k = min(k, available)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        chunk_ids = [int(ids[i]) for i in top]
        score_by_chunk = {int(ids[i]): float(scores[i]) for i in top}

    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = {
        r["chunk_id"]: r
        for r in conn.execute(
            f"""SELECT c.chunk_id, c.doc_key, c.text,
                       d.title, d.status, d.source_kind, d.doc_date, d.rbi_ref, d.url,
                       d.valid_from, d.valid_to, d.validity_certain, d.known_from
                FROM chunk c JOIN document d ON d.doc_key = c.doc_key
                WHERE c.chunk_id IN ({placeholders})""",
            chunk_ids,
        ).fetchall()
    }

    # Withdrawal dates live on the annex, joined by RBI document id.
    withdrawn_on = {
        r["doc_key"]: r["withdrawn_on"]
        for r in conn.execute(
            f"""SELECT d.doc_key, MIN(a.withdrawn_on) AS withdrawn_on
                FROM chunk c
                JOIN document d   ON d.doc_key = c.doc_key
                LEFT JOIN annex_entry a ON a.rbi_doc_id = d.rbi_doc_id
                WHERE c.chunk_id IN ({placeholders})
                GROUP BY d.doc_key""",
            chunk_ids,
        ).fetchall()
    }

    hits: list[Hit] = []
    for rank, cid in enumerate(chunk_ids, 1):
        row = rows.get(cid)
        if row is None:
            continue
        hits.append(
            Hit(
                rank=rank,
                score=score_by_chunk.get(cid, 0.0),
                chunk_id=cid,
                doc_key=row["doc_key"],
                text=row["text"],
                title=row["title"],
                status=row["status"],
                source_kind=row["source_kind"],
                doc_date=row["doc_date"],
                withdrawn_on=withdrawn_on.get(row["doc_key"]),
                rbi_ref=row["rbi_ref"],
                url=row["url"],
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
                validity_certain=bool(row["validity_certain"]),
                known_from=row["known_from"],
            )
        )
    return hits
