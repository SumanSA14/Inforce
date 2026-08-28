"""M8 — the demo.

One page, one question, two answers side by side: what a time-blind retriever
returns, and what a time-aware one returns. The point lands before anything is
explained.

Deviation from the plan: this is FastAPI serving a single self-contained HTML
page rather than a React/Vite front end. The UI is one form and two result
columns; a build step and a node dependency would add friction to a demo whose
whole job is to run immediately. Swapping to React later is a rewrite of one
file, not of the API.
"""
from __future__ import annotations

import pathlib
import sqlite3
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from . import config, embedding, entities, retrieve, temporal

STATIC = pathlib.Path(__file__).parent / "static"

app = FastAPI(title="InForce", docs_url="/api/docs")


def _conn() -> sqlite3.Connection:
    """A fresh connection per request — SQLite connections are not safe to
    share across the server's worker threads."""
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


class Query(BaseModel):
    question: str
    # Constrained, not a bare str. Nothing downstream parses this as a date --
    # temporal.validity_at compares it lexically -- so "banana" used to sail
    # through, get echoed back in the response, and permanently allocate a
    # 97k-element mask in retrieve._MASK_CACHE. Rejecting non-dates at the edge
    # closes both the reflected-value path and the unbounded cache key space.
    as_of: str = Field(default="2026-08-08", pattern=r"^\d{4}-\d{2}-\d{2}$")
    k: int = Field(default=5, ge=1, le=50)
    entity_scoping: bool = True

    @field_validator("as_of")
    @classmethod
    def _real_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except ValueError as exc:            # 2026-13-45 matches the pattern
            raise ValueError(f"not a calendar date: {v}") from exc
        return v


def _hit_json(h, as_of: str) -> dict:
    """`status` is present-tense; `validity` is as of the queried date.

    They differ for historical queries, and the difference is the whole point:
    a circular repealed in November 2025 genuinely governed conduct in 2022, so
    labelling it "repealed" on a 2022 query would report correct behaviour as a
    failure."""
    return {
        "rank": h.rank,
        "score": round(h.score, 4),
        "title": h.title or h.doc_key,
        "status": h.status,
        "validity": h.validity_on(as_of),
        "withdrawn_on": h.withdrawn_on or h.valid_to,
        "doc_date": h.doc_date,
        "rbi_ref": h.rbi_ref,
        "url": h.url,
        "snippet": h.text[:320],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/abstention")
def abstention() -> dict:
    """Refusal behaviour, from the saved probe run.

    Served alongside the retrieval metrics because the page must not present
    one without the other: a system that answers precisely and fabricates when
    it has no answer is not a good system, and a refusal rate quoted without a
    false-refusal rate beside it means nothing.
    """
    import json

    path = pathlib.Path(__file__).resolve().parents[2] / "questions" / "abstention-raw.json"
    if not path.exists():
        return {"available": False}
    try:
        rows = [r for r in json.loads(path.read_text(encoding="utf-8"))
                if r.get("class") is not None]
    except Exception:  # noqa: BLE001 - a malformed file must not 500 the page
        return {"available": False}
    if not rows:
        return {"available": False}

    def rate(group, cls, only_retrieved=False):
        g = [r for r in rows if r["group"] == group]
        if only_retrieved:
            g = [r for r in g if r.get("retrieved")]
        return round(100.0 * sum(r["class"] == cls for r in g) / len(g), 1) if g else 0.0

    def count(group, only_retrieved=False):
        g = [r for r in rows if r["group"] == group]
        return len([r for r in g if r.get("retrieved")]) if only_retrieved else len(g)

    out = {
        "available": True,
        "n_unanswerable": count("unanswerable"),
        "n_answerable": count("answerable"),
        "n_answerable_retrieved": count("answerable", True),
        "refused": rate("unanswerable", "refused"),
        "fabricated": rate("unanswerable", "answered"),
        "false_refusal": rate("answerable", "refused", True),
        "gate_enabled": False,
        "gate": None,
    }

    # The gate's operating point, DERIVED from the saved scores rather than
    # typed here. Hand-copying a measured number into a second place is how the
    # earlier reports came to contradict each other.
    spath = path.parent / "abstention-scores.json"
    if spath.exists():
        try:
            sc = json.loads(spath.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return out
        neg = [r for r in rows
               if r["group"] == "unanswerable" and sc.get(r["qid"], -1) >= 0]
        pos = [r for r in rows if r["group"] == "answerable"
               and r.get("retrieved") and sc.get(r["qid"], -1) >= 0]
        if neg and pos:
            best = None
            for t in range(12):
                prevented = sum(1 for r in neg
                                if sc[r["qid"]] < t and r["class"] == "answered")
                lost = sum(1 for r in pos
                           if sc[r["qid"]] < t and r["class"] == "answered")
                if best is None or prevented - lost > best["net"]:
                    fab = sum(1 for r in neg
                              if sc[r["qid"]] >= t and r["class"] == "answered")
                    fr = sum(1 for r in pos
                             if sc[r["qid"]] < t or r["class"] == "refused")
                    best = {"threshold": t, "net": prevented - lost,
                            "prevented": prevented, "lost": lost,
                            "fabricated": round(100.0 * fab / len(neg), 1),
                            "false_refusal": round(100.0 * fr / len(pos), 1)}
            out["gate"] = best
    return out


@app.get("/api/examples")
def examples() -> list[dict]:
    """Questions from the golden set that the naive baseline gets wrong —
    the ones that actually demonstrate the failure."""
    conn = _conn()
    rows = conn.execute(
        """SELECT q.text, q.category
           FROM question q
           JOIN eval_result r ON r.qid = q.qid
           JOIN eval_run  e ON e.run_id = r.run_id AND e.mode = 'naive'
           WHERE q.answerable = 1 AND r.top1_withdrawn = 1
           GROUP BY q.qid ORDER BY RANDOM() LIMIT 8"""
    ).fetchall()
    conn.close()
    return [{"text": r["text"], "category": r["category"]} for r in rows]


@app.get("/api/snapshot")
def snapshot(as_of: str = "2026-08-08") -> dict:
    conn = _conn()
    s = _snapshot(conn, as_of)
    conn.close()
    return {"as_of": as_of, "in_force": s.in_force, "withdrawn": s.withdrawn,
            "unknown": s.unknown, "not_yet": s.not_yet, "total": s.total}


TIMELINE_DATES = [
    "2015-06-30", "2018-06-30", "2020-06-30", "2022-03-14", "2024-06-30",
    "2025-06-30", "2025-11-27", "2025-11-29", "2026-03-31", "2026-08-08",
]

# Events worth marking on the timeline.
EVENTS = {
    "2025-11-29": "RBI/2025-26/100 — 9,445 circulars withdrawn",
    "2026-08-08": "31 Jul 2026 — a further 628 repealed",
}


@app.get("/api/stats")
def stats() -> dict:
    """Live corpus figures for the home page. Nothing hard-coded in the UI."""
    conn = _conn()
    row = conn.execute(
        """SELECT COUNT(*) docs,
                  SUM(status='withdrawn') withdrawn,
                  SUM(status='in_force') in_force
           FROM document WHERE fetch_status='ok'""").fetchone()
    chunks = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunk WHERE embedding IS NOT NULL").fetchone()[0]
    annex = conn.execute("SELECT COUNT(*) FROM annex_entry").fetchone()[0]
    qs = conn.execute(
        "SELECT COUNT(*) t, SUM(answerable) a, COUNT(DISTINCT category) c FROM question"
    ).fetchone()
    edges = conn.execute(
        "SELECT COUNT(*) FROM supersession WHERE method='inferred_centroid'").fetchone()[0]
    pairs = conn.execute("SELECT COUNT(*) FROM confusable_pair").fetchone()[0]
    cats = [dict(r) for r in conn.execute(
        """SELECT category, COUNT(*) n FROM document
           WHERE status='in_force' AND category IS NOT NULL
           GROUP BY category ORDER BY n DESC LIMIT 12""")]
    conn.close()
    return {"documents": row["docs"], "withdrawn": row["withdrawn"],
            "in_force": row["in_force"], "chunks": chunks, "embedded": embedded,
            "annex_rows": annex, "questions": qs["t"], "answerable": qs["a"],
            "categories": qs["c"], "supersession_edges": edges,
            "confusable_pairs": pairs, "in_force_by_category": cats}


@app.get("/api/metrics")
def metrics() -> dict:
    """Every evaluation run, computed live from stored results.

    The dashboard derives its numbers rather than repeating them, so the page
    cannot drift out of step with the database the way a hard-coded table would.
    """
    conn = _conn()
    runs = []
    for r in conn.execute(
        """SELECT run_id, mode, label, k, questions, created_at
           FROM eval_run ORDER BY created_at DESC LIMIT 40"""
    ):
        rid, n = r["run_id"], r["questions"] or 1
        strict = conn.execute(
            "SELECT COUNT(*) FROM eval_result WHERE run_id=? AND expected_rank=1",
            (rid,)).fetchone()[0]
        stale = conn.execute(
            "SELECT COUNT(*) FROM eval_result WHERE run_id=? AND top1_withdrawn=1",
            (rid,)).fetchone()[0]
        # Split the old "wrong entity" bucket. Counting every category
        # disagreement as an entity error conflated "right rule, wrong entity
        # class" with "wrong rule entirely"; adjudicating the 15 flagged cases
        # found 7 of them were subject misses, so the entity mode was roughly
        # doubled. Both still count against lenient P@1 — only the attribution
        # changes.
        mismatches = conn.execute(
            """SELECT ed.title expected_title, d.title actual_title
               FROM eval_result e
               JOIN question q ON q.qid = e.qid
               LEFT JOIN document d  ON d.doc_key = e.top_doc_key
               LEFT JOIN document ed ON ed.doc_key = q.expected_doc_key
               WHERE e.run_id=? AND e.top1_withdrawn=0 AND e.category IS NOT NULL
                 AND d.category IS NOT NULL AND e.category<>d.category""",
            (rid,)).fetchall()
        wrong_entity = sum(
            entities.same_subject(m["expected_title"], m["actual_title"])
            for m in mismatches)
        wrong_subject = len(mismatches) - wrong_entity
        wrong = len(mismatches)
        recall = conn.execute(
            "SELECT SUM(expected_hit) FROM eval_result WHERE run_id=?", (rid,)
        ).fetchone()[0] or 0
        mrr = conn.execute(
            """SELECT AVG(CASE WHEN expected_rank IS NULL THEN 0
                               ELSE 1.0/expected_rank END)
               FROM eval_result WHERE run_id=?""", (rid,)).fetchone()[0] or 0
        runs.append({
            "run_id": rid, "mode": r["mode"], "label": r["label"], "n": n,
            "created_at": r["created_at"],
            "p1_strict": round(100*strict/n, 1),
            "p1_lenient": round(100*(n-stale-wrong)/n, 1),
            "recall": round(100*recall/n, 1),
            "mrr": round(mrr, 3),
            "stale": round(100*stale/n, 1),
            "wrong_entity": round(100*wrong_entity/n, 1),
            "wrong_subject": round(100*wrong_subject/n, 1),
            "wrong_total": round(100*wrong/n, 1),
            "miss": round(100*(n-recall)/n, 1),
        })
    conn.close()
    return {"runs": runs}


@app.get("/api/question-detail")
def question_detail(run_id: str) -> dict:
    """Per-question outcomes for one run, so the dashboard can be drilled into
    rather than only summarised."""
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        """SELECT e.qid, q.text, e.category, e.top1_withdrawn, e.expected_rank,
                  e.expected_hit, d.title top_title, d.status top_status
           FROM eval_result e
           JOIN question q ON q.qid = e.qid
           LEFT JOIN document d ON d.doc_key = e.top_doc_key
           WHERE e.run_id = ? ORDER BY e.qid""", (run_id,))]
    conn.close()
    return {"run_id": run_id, "results": rows}


# temporal.snapshot rescans all 3,645 documents (~385ms). The corpus is static
# for the server's lifetime, so memoise per as_of date. Deliberately kept here
# and not in temporal.py: tests build databases mid-process and must keep seeing
# live counts.
_SNAPS: dict[str, temporal.CorpusSnapshot] = {}


def _snapshot(conn: sqlite3.Connection, as_of: str) -> temporal.CorpusSnapshot:
    if as_of not in _SNAPS:
        if len(_SNAPS) > 512:      # arbitrary as_of values are user-supplied
            _SNAPS.clear()
        _SNAPS[as_of] = temporal.snapshot(conn, as_of)
    return _SNAPS[as_of]


@app.get("/api/timeline")
def timeline() -> dict:
    """Corpus composition at a series of dates. The November 2025 cliff is the
    single clearest statement of the problem, so the UI leads with it."""
    conn = _conn()
    points = []
    for d in TIMELINE_DATES:
        s = _snapshot(conn, d)
        points.append({"date": d, "in_force": s.in_force, "withdrawn": s.withdrawn,
                       "unknown": s.unknown, "event": EVENTS.get(d)})
    conn.close()
    return {"points": points}


# Supersession edges and document metadata are static for the life of the
# process. Rebuilding them per call meant three full table scans per request and
# put ~4s of the ~6s response time into work that never changes.
_GRAPH: dict | None = None


def _graph(conn: sqlite3.Connection) -> dict:
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    edges: dict[str, list[str]] = {}
    for r in conn.execute(
        "SELECT withdrawn_doc_key, replacement_doc_key FROM supersession "
        "WHERE method='inferred_centroid' ORDER BY withdrawn_doc_key, rank"
    ):
        edges.setdefault(r["withdrawn_doc_key"], []).append(r["replacement_doc_key"])
    meta = {
        r["doc_key"]: dict(r)
        for r in conn.execute(
            "SELECT doc_key, title, status, category, doc_date, url FROM document")
    }
    _GRAPH = {
        "edges": edges, "meta": meta,
        "status": {k: v["status"] for k, v in meta.items()},
        "cats": {k: v["category"] for k, v in meta.items()},
        "titles": {k: v["title"] for k, v in meta.items()},
    }
    return _GRAPH


def _trace(conn: sqlite3.Connection, doc_key: str, entity: str | None) -> dict:
    """Follow the supersession chain from a repealed document to what governs
    now. This is the agent's `trace` loop, exposed for the UI."""
    from .agent import MAX_HOPS, choose_next

    g = _graph(conn)
    edges, meta = g["edges"], g["meta"]
    status, cats, titles = g["status"], g["cats"], g["titles"]

    chain = [doc_key]
    for _ in range(MAX_HOPS):
        head = chain[-1]
        if status.get(head) == temporal.IN_FORCE:
            break
        nxt = choose_next(head, chain, entity, edges, cats, status, titles)
        if nxt is None:
            break
        chain.append(nxt)

    # Collapse runs of the same instrument being revised. Traversal walks them
    # (refusing to costs 15 points of resolution), but showing seven rows all
    # titled "Master Circular - Management of Advances" is noise, not lineage.
    from .agent import _norm_title

    steps, revisions = [], 0
    for key in chain:
        m = meta.get(key)
        entry = {
            "doc_key": key,
            "title": (m["title"] if m else key) or key,
            "status": m["status"] if m else "unknown",
            "category": m["category"] if m else None,
            "doc_date": m["doc_date"] if m else None,
            "url": m["url"] if m else None,
            "revisions": 1,
        }
        if steps and _norm_title(steps[-1]["title"]) == _norm_title(entry["title"]) \
                and entry["status"] == steps[-1]["status"]:
            steps[-1]["revisions"] += 1
            # keep the newest of the run as the representative
            steps[-1].update({k: entry[k] for k in
                              ("doc_key", "doc_date", "url")})
            revisions += 1
            continue
        steps.append(entry)

    return {"steps": steps, "hops": len(chain) - 1,
            "collapsed_revisions": revisions,
            "resolved": status.get(chain[-1]) == temporal.IN_FORCE}


@app.post("/api/search")
def search(q: Query) -> dict:
    if not q.question.strip():
        raise HTTPException(400, "empty question")
    conn = _conn()
    try:
        # Embed once, run both configurations against the same vector.
        vec = embedding.embed_query(q.question)
        entity = entities.detect(q.question) if q.entity_scoping else None

        blind = retrieve.search(conn, q.question, k=q.k, query_vec=vec)
        aware = retrieve.search(conn, q.question, k=q.k, as_of=q.as_of,
                                entity=entity, query_vec=vec)
        snap = _snapshot(conn, q.as_of)

        # Trace the repealed documents a time-blind system would have returned,
        # forward to whatever governs now. This is the agent's loop.
        traces, seen = [], set()
        for h in blind:
            if h.validity_on(q.as_of) == temporal.IN_FORCE or h.doc_key in seen:
                continue
            seen.add(h.doc_key)
            traces.append(_trace(conn, h.doc_key, entity))
            if len(traces) >= 3:
                break
    except retrieve.IndexStale as exc:
        raise HTTPException(409, str(exc)) from exc
    except retrieve.IndexNotBuilt as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        conn.close()

    return {
        "question": q.question,
        "as_of": q.as_of,
        "entity_detected": entity,
        "time_blind": {
            "hits": [_hit_json(h, q.as_of) for h in blind],
            "invalid": sum(h.validity_on(q.as_of) != temporal.IN_FORCE for h in blind),
            "top1_invalid": bool(blind and blind[0].validity_on(q.as_of) != temporal.IN_FORCE),
        },
        "time_aware": {
            "hits": [_hit_json(h, q.as_of) for h in aware],
            "invalid": sum(h.validity_on(q.as_of) != temporal.IN_FORCE for h in aware),
            "top1_invalid": bool(aware and aware[0].validity_on(q.as_of) != temporal.IN_FORCE),
        },
        "snapshot": {"in_force": snap.in_force, "withdrawn": snap.withdrawn,
                     "unknown": snap.unknown, "not_yet": snap.not_yet},
        "traces": traces,
    }


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
