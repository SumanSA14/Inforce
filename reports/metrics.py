"""Headline figures, read from the database at build time.

The reports used to carry hand-typed numbers. That drifts: an earlier pair of
PDFs disagreed with each other about how many bugs the project had found,
because the same fact was typed twice. Anything that changes when a measurement
is re-run is queried here instead, so regenerating a report cannot produce a
number the database does not support.

Prose stays in the report. Only figures live here.
"""
from __future__ import annotations

import pathlib
import sqlite3
import subprocess
import sys
from math import sqrt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from inforce import entities, store  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval on a proportion. Wilson rather than normal-approximation:
    at these n the normal interval is wrong near the extremes."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (centre - half), 100 * (centre + half))


def _conn() -> sqlite3.Connection:
    conn = store.connect()
    conn.row_factory = sqlite3.Row
    return conn


def run(label: str) -> dict | None:
    """Every metric for one evaluation run, or None if it has not been run."""
    conn = _conn()
    r = conn.execute(
        "SELECT run_id, questions, mode, created_at FROM eval_run WHERE label = ? "
        "ORDER BY created_at DESC LIMIT 1", (label,)).fetchone()
    if not r:
        return None
    rid, n = r["run_id"], r["questions"] or 1

    def one(sql: str) -> float:
        return conn.execute(sql, (rid,)).fetchone()[0] or 0

    strict = one("SELECT COUNT(*) FROM eval_result WHERE run_id=? AND expected_rank=1")
    recall = one("SELECT SUM(expected_hit) FROM eval_result WHERE run_id=?")
    stale = one("SELECT COUNT(*) FROM eval_result WHERE run_id=? AND top1_withdrawn=1")
    mrr = one("""SELECT AVG(CASE WHEN expected_rank IS NULL THEN 0
                                 ELSE 1.0/expected_rank END)
                 FROM eval_result WHERE run_id=?""")

    # Entity vs subject mismatch, split on the subject half of the RBI title.
    mism = conn.execute(
        """SELECT ed.title expected_title, d.title actual_title
           FROM eval_result e
           JOIN question q ON q.qid = e.qid
           LEFT JOIN document d  ON d.doc_key = e.top_doc_key
           LEFT JOIN document ed ON ed.doc_key = q.expected_doc_key
           WHERE e.run_id=? AND e.top1_withdrawn=0 AND e.category IS NOT NULL
             AND d.category IS NOT NULL AND e.category<>d.category""",
        (rid,)).fetchall()
    wrong_entity = sum(
        entities.same_subject(m["expected_title"], m["actual_title"]) for m in mism)
    wrong_subject = len(mism) - wrong_entity
    conn.close()

    lo, hi = wilson(int(strict), n)
    n_lo, n_hi = wilson(int(stale), n)
    return {
        "label": label, "n": n, "mode": r["mode"], "created_at": r["created_at"],
        "p1_strict": 100 * strict / n,
        "p1_strict_ci": (lo, hi),
        "p1_lenient": 100 * (n - stale - len(mism)) / n,
        "recall": 100 * recall / n,
        "mrr": mrr,
        "stale": 100 * stale / n,
        "stale_ci": (n_lo, n_hi),
        "wrong_entity": 100 * wrong_entity / n,
        "wrong_subject": 100 * wrong_subject / n,
        "miss": 100 * (n - recall) / n,
    }


def corpus() -> dict:
    conn = _conn()
    g = lambda q: conn.execute(q).fetchone()[0] or 0  # noqa: E731
    out = {
        "documents": g("SELECT COUNT(*) FROM document WHERE fetch_status='ok'"),
        "in_force": g("SELECT COUNT(*) FROM document WHERE status='in_force'"),
        "withdrawn": g("SELECT COUNT(*) FROM document WHERE status='withdrawn'"),
        "chunks": g("SELECT COUNT(*) FROM chunk"),
        "annex_rows": g("SELECT COUNT(*) FROM annex_entry"),
        "questions": g("SELECT COUNT(*) FROM question"),
        "answerable": g("SELECT COUNT(*) FROM question WHERE answerable=1"),
        "categories": g("SELECT COUNT(DISTINCT category) FROM question "
                        "WHERE category IS NOT NULL"),
        # Three candidates are stored per withdrawn document (rank 1-3), so the
        # row count is ~3x the number of documents that actually have a
        # replacement chain. The second figure is the meaningful one.
        "edge_rows": g("SELECT COUNT(*) FROM supersession WHERE method='inferred_centroid'"),
        "superseded_docs": g("SELECT COUNT(DISTINCT withdrawn_doc_key) FROM supersession "
                             "WHERE method='inferred_centroid'"),
        "pairs": g("SELECT COUNT(*) FROM confusable_pair"),
    }
    conn.close()
    return out


def question_origin_split(label: str = "v2-naive") -> dict | None:
    """N for hand-written vs generated questions inside one run.

    The check that the expansion did not simply make the benchmark easier.
    """
    conn = _conn()
    r = conn.execute("SELECT run_id FROM eval_run WHERE label=? "
                     "ORDER BY created_at DESC LIMIT 1", (label,)).fetchone()
    if not r:
        conn.close()
        return None
    out = {}
    for name, clause in (("handwritten", "NOT LIKE 'gen-%'"), ("generated", "LIKE 'gen-%'")):
        row = conn.execute(
            f"""SELECT COUNT(*) tot, SUM(top1_withdrawn) st FROM eval_result
                WHERE run_id=? AND qid {clause}""", (r["run_id"],)).fetchone()
        out[name] = {"n": row["tot"], "stale": row["st"] or 0,
                     "pct": 100 * (row["st"] or 0) / row["tot"] if row["tot"] else 0}
    conn.close()
    return out


def stale_by_category(label: str = "v2-naive", limit: int = 12) -> list[tuple[str, int, int]]:
    conn = _conn()
    r = conn.execute("SELECT run_id FROM eval_run WHERE label=? "
                     "ORDER BY created_at DESC LIMIT 1", (label,)).fetchone()
    if not r:
        conn.close()
        return []
    rows = conn.execute(
        """SELECT category, COUNT(*) tot, SUM(top1_withdrawn) st
           FROM eval_result WHERE run_id=? AND category IS NOT NULL
           GROUP BY category HAVING tot >= 3
           ORDER BY 1.0*SUM(top1_withdrawn)/COUNT(*) DESC""", (r["run_id"],)).fetchall()
    conn.close()
    return [(x["category"], x["st"] or 0, x["tot"]) for x in rows][:limit]


def test_count() -> int:
    """Ask pytest rather than trusting a number typed into a report."""
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        for line in reversed(p.stdout.strip().splitlines()):
            if "test" in line and "collected" in line:
                return int(line.split()[0])
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:
        pass
    return 0


def code_stats() -> dict:
    def count(globpat: str) -> tuple[int, int]:
        files = sorted((ROOT / globpat.split("/")[0]).rglob(globpat.split("/")[-1]))
        files = [f for f in files if "__pycache__" not in str(f)]
        return len(files), sum(
            len(f.read_text(encoding="utf-8", errors="replace").splitlines()) for f in files)
    mods, mod_lines = count("src/*.py")
    tests, test_lines = count("tests/*.py")
    return {"modules": mods, "module_lines": mod_lines,
            "tests": tests, "test_lines": test_lines}


if __name__ == "__main__":
    import json
    print(json.dumps({
        "corpus": corpus(),
        "origin_split": question_origin_split(),
        "tests": test_count(),
        "code": code_stats(),
        "runs": {lab: run(lab) for lab in
                 ("naive", "v2-naive", "v2-chunk", "v2-doc", "v2-doc+llm",
                  "v2-doc+expand", "v2-doc+expand+llm", "rank3-chunk", "rank3-doc")},
    }, indent=1, default=str))


# Arms of the ranking ablation, in pipeline order (not quality order).
V2_ARMS = ["v2-chunk", "v2-doc", "v2-doc+llm", "v2-doc+expand", "v2-doc+expand+llm"]


def best_arm(metric: str = "p1_strict") -> dict | None:
    """The arm that actually scored highest, not the most elaborate one.

    Reports previously hardcoded a preference order ending in the
    expansion-plus-rerank arm, on the assumption that more machinery wins. It
    does not: at n=264 both query expansion and LLM reranking make the result
    worse, so that order would have published the weakest arm as the headline.
    """
    runs = [r for r in (run(lab) for lab in V2_ARMS) if r]
    return max(runs, key=lambda r: r[metric]) if runs else None


def abstention(path: str = "questions/abstention-raw.json") -> dict | None:
    """Generation-layer abstention, read from a saved probe run.

    Two rates, always together. A refusal rate alone is meaningless: a system
    that refuses everything scores 100%. The answerable side is additionally
    conditioned on retrieval having surfaced the gold document, because
    declining when the answer was never retrieved is correct behaviour by the
    generator, not a fault of it -- attributing that to generation blames the
    one component that behaved properly.
    """
    import json

    p = ROOT / path
    if not p.exists():
        return None
    rows = [r for r in json.loads(p.read_text(encoding="utf-8"))
            if r.get("class") is not None]
    if not rows:
        return None

    def slice_(group, only_retrieved=False):
        g = [r for r in rows if r["group"] == group]
        return [r for r in g if r.get("retrieved")] if only_retrieved else g

    def pct(rs, cls):
        return (100.0 * sum(r["class"] == cls for r in rs) / len(rs)) if rs else 0.0

    neg = slice_("unanswerable")
    pos = slice_("answerable")
    pos_ret = slice_("answerable", True)
    return {
        "n_unanswerable": len(neg),
        "n_answerable": len(pos),
        "n_answerable_retrieved": len(pos_ret),
        "refused": pct(neg, "refused"),
        "partial": pct(neg, "partial"),
        "fabricated": pct(neg, "answered"),
        "false_refusal_all": pct(pos, "refused"),
        "false_refusal_retrieved": pct(pos_ret, "refused"),
        "answered_retrieved": pct(pos_ret, "answered"),
    }


def gate_sweep(probe: str = "questions/abstention-raw.json",
               scores: str = "questions/abstention-scores.json") -> dict | None:
    """The answerability-gate threshold curve, computed from saved data.

    The expensive parts -- generation and scoring -- are both recorded, so the
    whole curve is derived rather than re-measured. At threshold t a question
    scoring below t is refused by the gate; at or above t whatever the ungated
    system actually did stands, which is faithful because the gate runs before
    generation.
    """
    import json

    pp, sp = ROOT / probe, ROOT / scores
    if not (pp.exists() and sp.exists()):
        return None
    sc = json.loads(sp.read_text(encoding="utf-8"))
    rows = [r for r in json.loads(pp.read_text(encoding="utf-8"))
            if r.get("class") is not None and sc.get(r["qid"], -1) >= 0]
    neg = [r for r in rows if r["group"] == "unanswerable"]
    pos = [r for r in rows if r["group"] == "answerable" and r.get("retrieved")]
    if not neg or not pos:
        return None

    curve = []
    for t in range(0, 12):
        prevented = sum(1 for r in neg
                        if sc[r["qid"]] < t and r["class"] == "answered")
        lost = sum(1 for r in pos
                   if sc[r["qid"]] < t and r["class"] == "answered")
        fab = sum(1 for r in neg
                  if sc[r["qid"]] >= t and r["class"] == "answered")
        fr = sum(1 for r in pos
                 if sc[r["qid"]] < t or r["class"] == "refused")
        curve.append({"t": t, "prevented": prevented, "lost": lost,
                      "net": prevented - lost,
                      "fabricated": 100.0 * fab / len(neg),
                      "false_refusal": 100.0 * fr / len(pos)})
    best = max(curve, key=lambda c: c["net"])
    mean_neg = sum(sc[r["qid"]] for r in neg) / len(neg)
    mean_pos = sum(sc[r["qid"]] for r in pos) / len(pos)
    return {
        "n_unanswerable": len(neg), "n_answerable": len(pos),
        "mean_score_unanswerable": mean_neg,
        "mean_score_answerable": mean_pos,
        "separation": mean_pos - mean_neg,
        "curve": curve, "best": best,
        "useful": best["net"] > 0,
    }
