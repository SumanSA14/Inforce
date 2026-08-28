"""Compare two abstention runs, paired by question.

Adding a way to say no necessarily raises refusals on BOTH sides. The only
question that matters is the trade: how many fabrications were prevented, and
how many correct answers were lost to buy that. A gate that suppresses one of
each has done nothing but move the error around.

    python reports/compare_abstention.py questions/abstention-raw.json \\
                                         questions/abstention-gated.json
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
from math import comb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path: str) -> dict[str, dict]:
    rows = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return {r["qid"]: r for r in rows}


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return
    A, B = load(sys.argv[1]), load(sys.argv[2])
    qids = sorted(set(A) & set(B))
    print(f"paired on {len(qids)} questions\n")

    for group in ("unanswerable", "answerable"):
        g = [q for q in qids if A[q]["group"] == group]
        if not g:
            continue
        print("=" * 68)
        print(f"{group.upper()}  (n={len(g)})")
        print("=" * 68)
        for cls in ("refused", "partial", "answered"):
            a = sum(1 for q in g if A[q]["class"] == cls)
            b = sum(1 for q in g if B[q]["class"] == cls)
            arrow = "->"
            print(f"  {cls:<9} {100*a/len(g):5.1f}% {arrow} {100*b/len(g):5.1f}%"
                  f"   ({a} {arrow} {b})")

        # The trade, stated as counts of individual questions that moved.
        if group == "unanswerable":
            fixed = [q for q in g if A[q]["class"] == "answered"
                     and B[q]["class"] == "refused"]
            broke = [q for q in g if A[q]["class"] == "refused"
                     and B[q]["class"] == "answered"]
            label = ("fabrications prevented", "refusals lost")
        else:
            ret = [q for q in g if A[q].get("retrieved")]
            g = ret or g
            fixed = [q for q in g if A[q]["class"] == "refused"
                     and B[q]["class"] == "answered"]
            broke = [q for q in g if A[q]["class"] == "answered"
                     and B[q]["class"] == "refused"]
            label = ("newly answered", "correct answers lost")
            print(f"  (conditioned on retrieval surfacing the gold document: "
                  f"{len(g)} questions)")
        print(f"\n  {label[0]:<24} {len(fixed)}")
        print(f"  {label[1]:<24} {len(broke)}")
        print(f"  McNemar exact p = {exact_p(len(broke), len(fixed)):.4f}")
        print()

    # The single number that decides it.
    un = [q for q in qids if A[q]["group"] == "unanswerable"]
    an = [q for q in qids if A[q]["group"] == "answerable" and A[q].get("retrieved")]
    prevented = sum(1 for q in un
                    if A[q]["class"] == "answered" and B[q]["class"] == "refused")
    cost = sum(1 for q in an
               if A[q]["class"] == "answered" and B[q]["class"] == "refused")
    print("=" * 68)
    print(f"TRADE: {prevented} fabrications prevented, {cost} correct answers lost")
    if cost:
        print(f"       ratio {prevented/cost:.2f} prevented per answer sacrificed")
    print("A gate that trades one for one has moved the error, not reduced it.")


if __name__ == "__main__":
    main()
