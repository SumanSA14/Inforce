"""Retrieve top-k evidence for each drafted unanswerable question.

Verification has to judge the QUESTION against what the corpus actually
returns, not against the drafter's own prediction. Asking whether the system
refused would be circular -- it would validate the question by the behaviour
the question exists to measure. Asking whether the answer is present in the
retrieved text is not circular: it is a fact about the corpus.

    python reports/unanswerable_evidence.py
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
from inforce import entities, retrieve, store  # noqa: E402

DRAFT = ROOT / "questions" / "unanswerable-draft.json"
OUT = ROOT / "questions" / "unanswerable-evidence.json"
AS_OF = "2026-08-08"
K = 8


def main() -> None:
    qs = json.loads(DRAFT.read_text(encoding="utf-8"))
    conn = store.connect()
    conn.row_factory = sqlite3.Row
    out, t0 = [], time.time()
    for i, q in enumerate(qs, 1):
        ent = entities.detect(q["text"])
        hits = retrieve.search(conn, q["text"], k=K, as_of=AS_OF, entity=ent,
                               retrieval="hybrid")
        out.append({
            "qid": q["qid"], "text": q["text"], "type": q["type"],
            "why_unanswerable": q["why_unanswerable"],
            "entity_detected": ent,
            "evidence": [{"rank": h.rank,
                          "title": (h.title or h.doc_key)[:110],
                          "text": " ".join(h.text.split())[:900]} for h in hits],
        })
        if i % 10 == 0:
            print(f"  {i}/{len(qs)}  ({time.time()-t0:.0f}s)", flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} questions with evidence to {OUT.name} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
