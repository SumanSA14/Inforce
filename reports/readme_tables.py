"""Emit the README's results tables as markdown, from the database.

Same reasoning as reports/metrics.py: the README carried hand-typed figures and
drifted from the reports. Run this and paste, or diff it against the README to
catch a number that has gone stale.

    python reports/readme_tables.py
"""
from __future__ import annotations

import io
import pathlib
import sys

# The tables contain en-dashes; the Windows console default codec mangles them.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

ARMS = [
    ("v2-naive", "Naive RAG (dense top-k)"),
    ("v2-chunk", "+ as-of-date, entity scoping, hybrid"),
    ("v2-doc", "+ document-level aggregation"),
    ("v2-doc+llm", "+ LLM listwise rerank"),
    ("v2-doc+expand", "+ query expansion"),
    ("v2-doc+expand+llm", "+ expansion and rerank"),
]


def pct(x: float | None, d: int = 1) -> str:
    return "*pending*" if x is None else f"{x:.{d}f}%"


def main() -> None:
    C = M.corpus()
    split = M.question_origin_split()

    print("## Headline table\n")
    print("| Configuration | Top-1 **repealed** | **Correct doc first** | "
          "Correct doc in top-8 | MRR | n |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, name in ARMS:
        r = M.run(label)
        if not r:
            print(f"| {name} | *pending* | *pending* | *pending* | *pending* | — |")
            continue
        print(f"| {name} | {pct(r['stale'])} | **{pct(r['p1_strict'])}** | "
              f"{pct(r['recall'])} | {r['mrr']:.3f} | {r['n']} |")

    print("\n## Every metric\n")
    print("| Configuration | P@1 strict | 95% CI | P@1 lenient | R@8 | MRR | "
          "Stale | Wrong entity | Wrong subject | Miss |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, name in ARMS:
        r = M.run(label)
        if not r:
            print(f"| {name} | *pending* | | | | | | | | |")
            continue
        lo, hi = r["p1_strict_ci"]
        print(f"| {name} | **{pct(r['p1_strict'])}** | {lo:.1f}–{hi:.1f}% | "
              f"{pct(r['p1_lenient'])} | {pct(r['recall'])} | {r['mrr']:.3f} | "
              f"{pct(r['stale'])} | {pct(r['wrong_entity'])} | "
              f"{pct(r['wrong_subject'])} | {pct(r['miss'])} |")

    print("\n## Corpus\n")
    print(f"- {C['documents']:,} documents ({C['in_force']} in force, "
          f"{C['withdrawn']:,} withdrawn), {C['chunks']:,} chunks")
    print(f"- {C['annex_rows']:,} regulator-supplied labels, "
          f"{C['pairs']:,} confusable pairs")
    print(f"- {C['superseded_docs']:,} withdrawn documents have a replacement chain "
          f"({C['edge_rows']:,} candidate edges at ranks 1–3)")
    print(f"- {C['questions']} questions, {C['answerable']} answerable, "
          f"{C['categories']} categories")
    if split:
        print(f"- N by origin: hand-written {split['handwritten']['pct']:.1f}% "
              f"({split['handwritten']['n']}), generated "
              f"{split['generated']['pct']:.1f}% ({split['generated']['n']})")

    print("\n## Staleness by category (naive baseline)\n")
    print("| Category | N |")
    print("|---|---:|")
    for cat, st, tot in M.stale_by_category():
        print(f"| {cat} | {100*st/tot:.0f}% ({st}/{tot}) |")


if __name__ == "__main__":
    main()
