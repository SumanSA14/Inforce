"""Merge verified unanswerable questions into the golden set.

Only questions an independent verifier confirmed the corpus cannot answer are
merged. A question that the retrieved text DOES answer is not an unanswerable
question -- keeping it would mean scoring the system wrong for answering
correctly, which is the same defect that made 30% of the v1 answerable set
unscoreable.

    python reports/merge_unanswerable.py            # dry run
    python reports/merge_unanswerable.py --write
"""
from __future__ import annotations

import collections
import io
import json
import pathlib
import sqlite3
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inforce import questions, store  # noqa: E402

DRAFT = ROOT / "questions" / "unanswerable-draft.json"
VERDICTS = ROOT / "questions" / "unanswerable-verdicts.json"
GOLDEN = ROOT / "questions" / "golden.jsonl"


def main() -> None:
    write = "--write" in sys.argv
    drafts = {q["qid"]: q for q in json.loads(DRAFT.read_text(encoding="utf-8"))}
    if not VERDICTS.exists():
        print(f"missing {VERDICTS.name} — run the verification step first")
        return
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))

    counts = collections.Counter(v["verdict"] for v in verdicts)
    print(f"verdicts: {dict(counts)}")

    keep = [v for v in verdicts if v["verdict"] == "unanswerable"]
    reject = [v for v in verdicts if v["verdict"] != "unanswerable"]
    print(f"keeping {len(keep)}, rejecting {len(reject)}")
    if reject:
        print("\nrejected (the corpus can answer these, or the verifier was unsure):")
        for v in reject[:12]:
            print(f"   [{v['qid']}] {v['verdict']}: {v.get('reason', '')[:120]}")

    conn = store.connect()
    conn.row_factory = sqlite3.Row
    existing = questions.load_jsonl(GOLDEN)
    have = {q.qid for q in existing}

    new = []
    for v in keep:
        d = drafts.get(v["qid"])
        if not d or v["qid"] in have:
            continue
        new.append({
            "qid": d["qid"], "text": d["text"], "topic": d.get("type"),
            "category": None, "answerable": False,
            "expected_doc_key": None, "expected_quote": None,
            "trap_doc_keys": [], "difficulty": "hard",
            "notes": f"unanswerable ({d['type']}) — {d['why_unanswerable'][:200]}",
        })
    print(f"\n{len(new)} new unanswerable questions to add")

    # Validate the merged set before writing anything.
    merged = existing + [
        questions.Question(
            qid=q["qid"], text=q["text"], topic=q["topic"], category=None,
            answerable=False, expected_doc_key=None, expected_quote=None,
            trap_doc_keys=[], difficulty=q["difficulty"], notes=q["notes"])
        for q in new
    ]
    problems = questions.validate(conn, merged)
    print(f"whole-set validation problems: {len(problems)}")
    for p in problems[:10]:
        print("   ", p[:160])

    total_unans = sum(1 for q in merged if not q.answerable)
    print(f"\nmerged set: {len(merged)} questions, "
          f"{sum(q.answerable for q in merged)} answerable, "
          f"{total_unans} unanswerable")

    if not write:
        print("\n[dry run] pass --write to append")
        return
    if problems:
        print("\nREFUSING to write: validation failed")
        sys.exit(1)
    with GOLDEN.open("a", encoding="utf-8") as fh:
        fh.write("\n# --- v2 unanswerable: verified against retrieved evidence ---\n")
        for q in new:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nappended {len(new)} unanswerable questions to {GOLDEN.name}")


if __name__ == "__main__":
    main()
