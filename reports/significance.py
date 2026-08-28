"""Paired significance tests between evaluation runs.

A percentage comparison cannot distinguish a real gain from noise at these
sample sizes: at n=264 a single question moves precision@1 by 0.4 points, and
at n=52 it moved it by 1.9. Because every arm is scored on the SAME questions,
the correct test is paired -- McNemar's exact test on the discordant pairs,
which ignores the questions both arms agree on and asks only whether the
disagreements lean one way.

The discordant counts matter as much as the p-value. An arm that gains 6 and
loses 5 is churning, not improving, even if its headline number rose.

    python reports/significance.py [labelA labelB ...]
"""
from __future__ import annotations

import io
import pathlib
import sqlite3
import sys
from math import comb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from inforce import entities, store  # noqa: E402

DEFAULT_PAIRS = [
    ("v2-chunk", "v2-doc"),
    ("v2-doc", "v2-doc+llm"),
    ("v2-chunk", "v2-doc+llm"),
    ("v2-doc", "v2-doc+expand"),
    ("v2-doc+llm", "v2-doc+expand+llm"),
]


def exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial on the discordant pairs, H0: p = 0.5.

    Exact rather than the chi-square approximation: with a handful of
    discordant pairs the approximation is unreliable in exactly the regime
    where the answer matters.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _conn() -> sqlite3.Connection:
    conn = store.connect()
    conn.row_factory = sqlite3.Row
    return conn


def outcomes(conn: sqlite3.Connection, label: str) -> dict[str, dict] | None:
    r = conn.execute("SELECT run_id FROM eval_run WHERE label=? "
                     "ORDER BY created_at DESC LIMIT 1", (label,)).fetchone()
    if not r:
        return None
    out = {}
    for row in conn.execute(
        """SELECT e.qid, e.expected_rank, e.expected_hit, e.top1_withdrawn,
                  e.category, d.category top_cat, d.title top_title,
                  ed.title expected_title
           FROM eval_result e
           JOIN question q ON q.qid = e.qid
           LEFT JOIN document d  ON d.doc_key = e.top_doc_key
           LEFT JOIN document ed ON ed.doc_key = q.expected_doc_key
           WHERE e.run_id = ?""", (r["run_id"],)
    ):
        mismatch = (not row["top1_withdrawn"] and row["category"] and row["top_cat"]
                    and row["category"] != row["top_cat"])
        out[row["qid"]] = {
            "strict": row["expected_rank"] == 1,
            "recall": bool(row["expected_hit"]),
            "lenient": not row["top1_withdrawn"] and not mismatch,
            "entity_miss": bool(mismatch and entities.same_subject(
                row["expected_title"], row["top_title"])),
        }
    return out


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns which tests survive correction.

    This matters here: the default sweep runs 15 tests, and at 15 tests a
    single p just under 0.05 is exactly what noise produces. Reporting raw
    p-values across a family of comparisons and highlighting the ones below
    0.05 is how a negative result gets published as a positive one. Holm is
    used rather than plain Bonferroni because it is uniformly more powerful at
    the same error rate.
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out = [False] * len(pvals)
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (len(pvals) - rank):
            out[idx] = True
        else:
            break          # step-down: once one fails, all larger p fail too
    return out


def compare(conn, a_label: str, b_label: str, collect: list | None = None) -> None:
    A, B = outcomes(conn, a_label), outcomes(conn, b_label)
    if A is None or B is None:
        print(f"{a_label} vs {b_label}: MISSING RUN")
        return
    qids = sorted(set(A) & set(B))
    print(f"\n{a_label}  ->  {b_label}   (n={len(qids)} paired)")
    for metric in ("strict", "lenient", "recall"):
        lost = sum(1 for q in qids if A[q][metric] and not B[q][metric])
        gained = sum(1 for q in qids if not A[q][metric] and B[q][metric])
        p = exact_p(lost, gained)
        na = sum(A[q][metric] for q in qids)
        nb = sum(B[q][metric] for q in qids)
        verdict = "raw p<0.05" if p < 0.05 else "not significant"
        print(f"   {metric:<8} {100*na/len(qids):5.1f}% -> {100*nb/len(qids):5.1f}%  "
              f"gained {gained:>3}, lost {lost:>3}   p={p:.3f}   {verdict}")
        if collect is not None:
            collect.append((f"{a_label}->{b_label} {metric}", p, gained, lost))


def main() -> None:
    conn = _conn()
    args = sys.argv[1:]
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)] or DEFAULT_PAIRS
    collected: list = []
    for a, b in pairs:
        compare(conn, a, b, collected)

    print()
    print("=" * 72)
    print(f"HOLM-BONFERRONI CORRECTION ({len(collected)} tests in this family)")
    print("=" * 72)
    survives = holm([c[1] for c in collected])
    any_survive = False
    for (name, pv, gained, lost), ok in sorted(
            zip(collected, survives), key=lambda t: t[0][1]):
        if pv < 0.05:
            mark = "SURVIVES" if ok else "fails correction"
            any_survive |= ok
            print(f"  p={pv:.4f}  {mark:<17} {name}  (gained {gained}, lost {lost})")
    if not any_survive:
        print("  Nothing survives correction. Any single result below 0.05 here is")
        print("  what 15 tests produce by chance; do not report one as a finding.")
    print()
    print("Discordant counts carry the signal independently of the p-value:")
    print("a change that gains 6 and loses 5 has a higher headline number and")
    print("has improved nothing, while gained 0 / lost 9 is one-sided whatever")
    print("the correction says.")
    conn.close()


if __name__ == "__main__":
    main()
