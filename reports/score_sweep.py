"""Score every probed question, then sweep the gate threshold offline.

The binary gate collapsed to a constant NO. A calibrated score makes the
operating point tunable -- and because the expensive part (generation) is
already recorded in the saved probe run, the entire threshold curve costs ONE
scoring pass rather than one full run per threshold.

Simulation rule at threshold t: a question scoring below t is refused by the
gate; at or above t, whatever the ungated system actually did stands. That is
exactly what the gate would do, since it runs before generation.

    python reports/score_sweep.py           # score, then sweep
    python reports/score_sweep.py --sweep   # sweep only, reusing saved scores
"""
from __future__ import annotations

import io
import json
import pathlib
import sqlite3
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inforce import answerable, entities, retrieve, store  # noqa: E402

PROBE = ROOT / "questions" / "abstention-raw.json"
SCORES = ROOT / "questions" / "abstention-scores.json"
AS_OF = "2026-08-08"


EVIDENCE = ROOT / "questions" / "sweep-evidence.json"


def do_retrieve() -> None:
    """Pass 1: retrieve for every question, using ONLY the embedding model.

    Split from scoring deliberately. Interleaving retrieval and scoring makes
    every question alternate between nomic-embed-text and qwen2.5:3b, and on a
    4 GB card holding both (2.48 GB plus KV cache for 6k-char prompts) that
    thrashes VRAM: a single embed went from 155 ms to 3.2 s and per-question
    cost rose 24x mid-run. Two passes means each model loads once.
    """
    if EVIDENCE.exists():
        print(f"reusing {EVIDENCE.name}")
        return
    rows = json.loads(PROBE.read_text(encoding="utf-8"))
    conn = store.connect()
    conn.row_factory = sqlite3.Row
    out, t0 = {}, time.time()
    for i, r in enumerate(rows, 1):
        ent = entities.detect(r["question"])
        hits = retrieve.search(conn, r["question"], k=8, as_of=AS_OF,
                               entity=ent, retrieval="hybrid")
        out[r["qid"]] = {
            "question": r["question"],
            "hits": [{"rank": h.rank, "title": h.title or h.doc_key,
                      "text": h.text[:1200], "doc_key": h.doc_key} for h in hits],
        }
        if i % 25 == 0:
            print(f"  retrieved {i}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    EVIDENCE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"retrieved {len(out)} ({time.time()-t0:.0f}s)")


class _Hit:
    def __init__(self, d):
        self.rank, self.title = d["rank"], d["title"]
        self.text, self.doc_key = d["text"], d["doc_key"]


def do_score() -> None:
    """Pass 2: score from cached evidence, using ONLY the chat model."""
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    out, t0 = {}, time.time()
    for i, (qid, d) in enumerate(ev.items(), 1):
        hits = [_Hit(h) for h in d["hits"]]
        out[qid] = answerable.score_per_chunk(d["question"], hits)
        if i % 20 == 0:
            print(f"  scored {i}/{len(ev)}  ({time.time()-t0:.0f}s)", flush=True)
            SCORES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    SCORES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} scores ({time.time()-t0:.0f}s)")


def sweep() -> None:
    rows = json.loads(PROBE.read_text(encoding="utf-8"))
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    rows = [r for r in rows
            if r.get("class") is not None and scores.get(r["qid"], -1) >= 0]

    neg = [r for r in rows if r["group"] == "unanswerable"]
    pos = [r for r in rows if r["group"] == "answerable" and r.get("retrieved")]
    print(f"\nusable: {len(neg)} unanswerable, {len(pos)} answerable "
          f"(gold retrieved)\n")

    # Do the two classes separate at all?
    def mean(rs):
        return sum(scores[r["qid"]] for r in rs) / len(rs) if rs else 0.0
    print(f"mean score  unanswerable {mean(neg):.2f}   answerable {mean(pos):.2f}")
    if mean(pos) - mean(neg) < 0.5:
        print("  -> the classes barely separate; no threshold can do much.")
    print()

    print(f"{'thresh':>7}{'fabricated':>12}{'false refuse':>14}"
          f"{'prevented':>11}{'lost':>7}{'net':>7}")
    print("-" * 58)
    best = None
    for t in range(0, 12):
        # below threshold -> gate refuses; at/above -> keep what actually happened
        fab = sum(1 for r in neg
                  if scores[r["qid"]] >= t and r["class"] == "answered")
        fr = sum(1 for r in pos
                 if scores[r["qid"]] < t or r["class"] == "refused")
        prevented = sum(1 for r in neg
                        if scores[r["qid"]] < t and r["class"] == "answered")
        lost = sum(1 for r in pos
                   if scores[r["qid"]] < t and r["class"] == "answered")
        net = prevented - lost
        mark = ""
        if best is None or net > best[1]:
            best, mark = (t, net), ""
        print(f"{t:>7}{100*fab/len(neg):>11.1f}%{100*fr/len(pos):>13.1f}%"
              f"{prevented:>11}{lost:>7}{net:>+7}")

    t, net = best
    print()
    if net <= 0:
        print(f"NO USEFUL OPERATING POINT. Best is t={t} at net {net:+d}: every")
        print("threshold destroys at least as many correct answers as it saves")
        print("fabrications. The judge cannot separate the two classes, and this")
        print("is a stronger negative result than one failed guess -- it holds")
        print("across the whole curve, not at a single setting.")
    else:
        print(f"BEST OPERATING POINT t={t}: net {net:+d} "
              f"(prevented - lost). Tuned on this data, so treat as an upper bound.")


if __name__ == "__main__":
    if "--sweep" not in sys.argv:
        do_retrieve()
        do_score()
    sweep()
