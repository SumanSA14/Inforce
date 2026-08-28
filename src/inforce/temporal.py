"""M6 — bi-temporal validity.

Two independent time axes, the standard bi-temporal model (and, in warehouse
terms, Slowly Changing Dimension Type 2):

  **valid time**       when the regulation was in force in the real world
                       (`valid_from` .. `valid_to`)
  **transaction time** when *we* learned it (`known_from`), so the system can
                       answer "what did this system believe on date X" — which
                       is what an audit of the system itself requires.

The critical modelling decision is that validity is **three-valued**, not two.
`valid_to IS NULL` is ambiguous: it can mean "still in force" or "withdrawn on
a date RBI did not publish". Seventeen annex rows are in exactly that state,
because RBI appends to the withdrawal annex over time and batch membership does
not imply a single withdrawal date. Collapsing those two cases into a boolean
would silently resurrect dead regulation — the precise failure this project
exists to measure. So the answer to "was this in force on date D" is
IN_FORCE, WITHDRAWN, or UNKNOWN, and UNKNOWN is never quietly rounded to either.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

IN_FORCE = "in_force"
WITHDRAWN = "withdrawn"
UNKNOWN = "unknown"
NOT_YET = "not_yet_issued"

COLUMNS = {
    "valid_from": "TEXT",
    "valid_to": "TEXT",
    "validity_certain": "INTEGER NOT NULL DEFAULT 1",
    "known_from": "TEXT",
}

SUPERSESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS supersession (
    withdrawn_doc_key   TEXT NOT NULL,
    replacement_doc_key TEXT NOT NULL,
    method              TEXT NOT NULL,   -- inferred_similarity | published
    confidence          REAL NOT NULL,
    evidence            TEXT,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (withdrawn_doc_key, replacement_doc_key)
);
CREATE INDEX IF NOT EXISTS idx_supersession_from ON supersession(withdrawn_doc_key);
"""


@dataclass
class CorpusSnapshot:
    as_of: str
    in_force: int = 0
    withdrawn: int = 0
    unknown: int = 0
    not_yet: int = 0

    @property
    def total(self) -> int:
        return self.in_force + self.withdrawn + self.unknown + self.not_yet


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add temporal columns to `document` if absent. Idempotent."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(document)")}
    added = []
    for name, decl in COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE document ADD COLUMN {name} {decl}")
            added.append(name)
    conn.executescript(SUPERSESSION_SCHEMA)
    conn.commit()
    return added


def backfill(conn: sqlite3.Connection, *, log=print) -> dict[str, int]:
    """Populate valid time from the annex labels and document dates.

    `valid_from` prefers the document's own date, falling back to the annex
    circular date. `valid_to` comes from the annex withdrawal date;
    `validity_certain` records whether that date is actually known.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # In-force Master Directions: open-ended validity, certain.
    conn.execute(
        """UPDATE document
           SET valid_from = COALESCE(doc_date, updated_as_on),
               valid_to = NULL,
               validity_certain = 1,
               known_from = COALESCE(known_from, ?)
           WHERE status = 'in_force'""",
        (now,),
    )

    # Withdrawn circulars: close the interval using the annex.
    conn.execute(
        """UPDATE document
           SET valid_from = COALESCE(
                   doc_date,
                   (SELECT MIN(a.circular_date) FROM annex_entry a
                     WHERE a.rbi_doc_id = document.rbi_doc_id)),
               valid_to = (SELECT MIN(a.withdrawn_on) FROM annex_entry a
                            WHERE a.rbi_doc_id = document.rbi_doc_id
                              AND a.withdrawn_on IS NOT NULL),
               known_from = COALESCE(known_from, ?)
           WHERE status = 'withdrawn'""",
        (now,),
    )

    # A withdrawn document with no published withdrawal date is UNCERTAIN, not
    # in force. This flag is what keeps the three-valued logic honest.
    conn.execute(
        """UPDATE document SET validity_certain = 0
           WHERE status = 'withdrawn' AND valid_to IS NULL"""
    )
    conn.commit()

    stats = {
        "in_force": conn.execute(
            "SELECT COUNT(*) FROM document WHERE status='in_force'").fetchone()[0],
        "withdrawn_dated": conn.execute(
            "SELECT COUNT(*) FROM document WHERE status='withdrawn' "
            "AND valid_to IS NOT NULL").fetchone()[0],
        "withdrawn_undated": conn.execute(
            "SELECT COUNT(*) FROM document WHERE status='withdrawn' "
            "AND valid_to IS NULL").fetchone()[0],
        "missing_valid_from": conn.execute(
            "SELECT COUNT(*) FROM document WHERE valid_from IS NULL").fetchone()[0],
    }
    log(f"backfilled: {stats}")
    return stats


def validity_at(
    valid_from: str | None,
    valid_to: str | None,
    status: str,
    certain: bool,
    as_of: str,
    known_from: str | None = None,
) -> str:
    """Three-valued validity. Pure function so it is trivially testable.

    `known_from` is the transaction-time axis — the date we read the annex.
    It resolves most of the undated cases: a document listed as withdrawn in an
    annex read on date K is *certainly* withdrawn on any date at or after K,
    even when RBI never published when it happened. Only queries about dates
    strictly before K remain genuinely UNKNOWN. This is the transaction-time
    axis doing real work rather than being decorative metadata.
    """
    if valid_from and as_of < valid_from:
        return NOT_YET
    if status == IN_FORCE:
        return IN_FORCE

    # status == withdrawn
    if certain and valid_to is not None:
        return WITHDRAWN if as_of >= valid_to else IN_FORCE

    # Withdrawn at some unpublished point.
    if known_from and as_of >= known_from[:10]:
        return WITHDRAWN
    return UNKNOWN


def snapshot(conn: sqlite3.Connection, as_of: str) -> CorpusSnapshot:
    snap = CorpusSnapshot(as_of=as_of)
    for r in conn.execute(
        """SELECT valid_from, valid_to, status, validity_certain, known_from
           FROM document WHERE fetch_status = 'ok'"""
    ):
        verdict = validity_at(
            r["valid_from"], r["valid_to"], r["status"], bool(r["validity_certain"]),
            as_of, r["known_from"],
        )
        setattr(snap, {IN_FORCE: "in_force", WITHDRAWN: "withdrawn",
                       UNKNOWN: "unknown", NOT_YET: "not_yet"}[verdict],
                getattr(snap, {IN_FORCE: "in_force", WITHDRAWN: "withdrawn",
                               UNKNOWN: "unknown", NOT_YET: "not_yet"}[verdict]) + 1)
    return snap


def infer_supersession_chains(
    conn: sqlite3.Connection, *, floor: float = 0.80, candidates: int = 3, log=print
) -> int:
    """Infer replacement edges at DOCUMENT level, allowing multi-hop chains.

    The earlier chunk-level version drew edges only from the confusable-pair
    table, which pairs withdrawn documents against *in-force* ones exclusively.
    Every chain it produced was therefore exactly one hop — not because RBI's
    reality is one hop, but because the inference could not represent anything
    else. Seventeen consolidation-era Directions were themselves later
    withdrawn, so real two-hop chains exist: an old circular is consolidated
    into a 2025 Direction, which is then repealed in favour of a current one.

    This version compares document centroids (mean of a document's chunk
    vectors) and links each withdrawn document to its most similar *later*
    document of any status. Centroids make the whole thing a 3,645 x 3,645
    problem rather than 64,510 x 97,037, and supersession is a document-level
    relation anyway.

    Still INFERRED, not published: RBI stated which circulars were withdrawn,
    never where each one went.
    """
    import numpy as np

    from . import retrieve

    matrix, ids = retrieve.load_index()
    chunk_doc = {
        r["chunk_id"]: r["doc_key"]
        for r in conn.execute(
            "SELECT chunk_id, doc_key FROM chunk WHERE embedding IS NOT NULL")
    }
    meta = {
        r["doc_key"]: (r["status"], r["valid_from"] or r["doc_date"] or "")
        for r in conn.execute(
            "SELECT doc_key, status, valid_from, doc_date FROM document")
    }

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for row, cid in enumerate(ids):
        key = chunk_doc.get(int(cid))
        if key is None:
            continue
        v = np.asarray(matrix[row], dtype=np.float32)
        if key in sums:
            sums[key] += v
            counts[key] += 1
        else:
            sums[key] = v.copy()
            counts[key] = 1

    keys = sorted(sums)
    cent = np.vstack([sums[k] / counts[k] for k in keys]).astype(np.float32)
    norms = np.linalg.norm(cent, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cent /= norms
    log(f"document centroids: {cent.shape}")

    dates = np.array([meta.get(k, ("", ""))[1] for k in keys])
    is_withdrawn = np.array([meta.get(k, ("", ""))[0] == "withdrawn" for k in keys])

    sim = cent @ cent.T
    np.fill_diagonal(sim, -1.0)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for i in np.flatnonzero(is_withdrawn):
        # A replacement must be strictly LATER than what it replaces.
        later = dates > dates[i]
        if not later.any():
            continue
        scores = np.where(later, sim[i], -1.0)
        # Keep several candidates, not just the best. Consolidation produced
        # near-identical documents per entity type, so the single most similar
        # replacement is frequently the right topic under the WRONG entity —
        # a Small Finance Banks question tracing into Urban Co-operative Banks.
        # Storing alternatives lets the tracer choose using the question's
        # entity instead of committing at inference time.
        top = np.argpartition(-scores, min(candidates, len(scores) - 1))[:candidates]
        top = top[np.argsort(-scores[top])]
        for rank, j in enumerate(top, start=1):
            j = int(j)
            if scores[j] < floor:
                continue
            rows.append((keys[i], keys[j], rank, float(scores[j]),
                         f"centroid similarity, replacement dated {dates[j]}", now))

    conn.executescript(SUPERSESSION_SCHEMA)
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(supersession)")}
    if "rank" not in existing:
        conn.execute("ALTER TABLE supersession ADD COLUMN rank INTEGER NOT NULL DEFAULT 1")
    conn.execute("DELETE FROM supersession WHERE method='inferred_centroid'")
    conn.executemany(
        """INSERT OR REPLACE INTO supersession
           (withdrawn_doc_key, replacement_doc_key, rank, method, confidence,
            evidence, created_at)
           VALUES (?,?,?,'inferred_centroid',?,?,?)""",
        rows,
    )
    conn.commit()
    sources = len({r[0] for r in rows})
    log(f"inferred {len(rows):,} centroid edges over {sources:,} documents "
        f"(up to {candidates} candidates each, similarity >= {floor})")
    return len(rows)


def load_chain_maps(conn: sqlite3.Connection) -> tuple[dict, dict]:
    """Load the edge and status maps once, for callers that walk many chains."""
    edges: dict[str, str] = {}
    for r in conn.execute(
        "SELECT withdrawn_doc_key, replacement_doc_key FROM supersession "
        "WHERE method='inferred_centroid' ORDER BY withdrawn_doc_key, rank"
    ):
        edges.setdefault(r["withdrawn_doc_key"], r["replacement_doc_key"])
    status = {r["doc_key"]: r["status"]
              for r in conn.execute("SELECT doc_key, status FROM document")}
    return edges, status


def chain_from(
    conn: sqlite3.Connection,
    doc_key: str,
    *,
    max_hops: int = 6,
    maps: tuple[dict, dict] | None = None,
) -> list[str]:
    """Follow replacement edges until an in-force document, a cycle, or the hop
    limit. The hop limit is a safety net, not the expected exit.

    Pass `maps` from `load_chain_maps` when walking many chains — rebuilding
    them per call turns a sweep over 3,000 documents into an O(n^2) table scan.
    """
    edges, status = maps if maps is not None else load_chain_maps(conn)

    path, seen, cur = [doc_key], {doc_key}, doc_key
    for _ in range(max_hops):
        nxt = edges.get(cur)
        if nxt is None or nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
        if status.get(cur) == IN_FORCE:
            break
    return path
