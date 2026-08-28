"""Entity scoping.

The November 2025 consolidation produced roughly eleven near-identical
documents per topic — one per class of regulated entity. That made entity
confusion a failure mode in its own right: at M7, once repealed documents were
removed from competition, 41.7% of top-1 results were the right era but the
wrong entity type (the Small Finance Banks capital rule answering a Commercial
Banks question).

Detection is deliberately deterministic keyword matching, not an LLM call. It
is cheap, auditable, and its errors are inspectable — and it means the entity
filter cannot quietly become another model whose mistakes are invisible.

Patterns are ordered most-specific-first: "regional rural bank" must win over
"rural co-operative bank", and "rural co-operative" over a bare "co-operative".
"""
from __future__ import annotations

import re

# (canonical category, regex). Order matters — first match wins.
PATTERNS: list[tuple[str, str]] = [
    ("Regional Rural Banks", r"\bregional rural bank|\brrbs?\b"),
    ("Rural Co-operative Banks", r"\brural co-?operative|\brural coop|\bstcbs?\b|\bdccbs?\b"),
    ("Urban Co-operative Banks", r"\burban co-?operative|\burban coop|\bucbs?\b"),
    ("Small Finance Banks", r"\bsmall finance bank|\bsfbs?\b"),
    ("Payments Banks", r"\bpayments? bank"),
    ("Local Area Banks", r"\blocal area bank|\blabs?\b"),
    ("Non-Banking Financial Companies", r"\bnon-?banking financial|\bnbfcs?\b"),
    ("Asset Reconstruction Companies", r"\basset reconstruction|\barcs?\b"),
    ("Credit Information Companies", r"\bcredit information compan|\bcics?\b"),
    ("All India Financial Institutions", r"\ball india financial institution|\baifis?\b"),
    ("Commercial Banks", r"\bcommercial bank|\bscheduled commercial"),
]

_COMPILED = [(cat, re.compile(pat, re.IGNORECASE)) for cat, pat in PATTERNS]


def detect(question: str) -> str | None:
    """Infer the regulated-entity category a question is about.

    Returns None when no entity is named — in which case no entity filter is
    applied, which is the correct behaviour: a question about banknote
    recirculation is not scoped to a bank type.
    """
    if not question:
        return None
    for category, pattern in _COMPILED:
        if pattern.search(question):
            return category
    return None


def categories() -> list[str]:
    return [c for c, _ in PATTERNS]


# --- subject families -------------------------------------------------------
#
# The "wrong entity" metric originally counted any top-1 whose category differed
# from the question's. That conflated two unrelated failures: returning the
# right rule for the wrong entity class (Securitisation for SFBs instead of
# NBFCs — benign, the rule is often identical), and returning an entirely
# different rule (Securitisation answered with Capital Adequacy — a real miss).
# An adjudication of all 15 flagged cases found 7 were subject misses filed
# under an entity label, so the metric overstated the benign mode about 2x.
#
# RBI titles carry the split explicitly:
#   Reserve Bank of India (Small Finance Banks - Securitisation of ...) Directions
#                          ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
#                          entity                subject
# Content-word overlap of the subject halves separates the two cleanly: across
# the 15 adjudicated cases every subject miss scored exactly 0.00 and every
# entity miss scored >= 0.33, so any threshold in [0.05, 0.30] reproduces the
# adjudication exactly. 0.20 sits in the middle of that band rather than on
# either edge — the split is categorical, not a tuned cutoff.
SUBJECT_OVERLAP_FLOOR = 0.20

# The separator is a *spaced* dash. Splitting on any dash breaks on the ASCII
# hyphens inside compound entity names — "Non-Banking Financial Companies" and
# "Co-operative" — which silently dragged the entity words into the subject and
# depressed the overlap score.
_DASH = r"\s[-‐-―−]\s"
_PAREN = re.compile(r"\(([^)]*)\)")
_NOISE = re.compile(r"\b(directions?|guidelines?|norms?)\b")
_STOP = {"of", "and", "the", "on", "for", "in", "to", "a"}


def subject_of(title: str | None) -> str | None:
    """The subject half of an RBI title, or None if it has no entity split.

    Master Directions ("Master Direction - Foreign Investment in India") have no
    per-entity siblings, so they return None: an entity mismatch is impossible
    by construction and any miss against them is a subject miss.
    """
    if not title:
        return None
    for inner in _PAREN.findall(title):
        parts = re.split(_DASH, inner, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            s = _NOISE.sub("", parts[1].strip().lower())
            return " ".join(re.sub(r"[^a-z ]", " ", s).split())
    return None


def same_subject(expected_title: str | None, actual_title: str | None) -> bool:
    """Do two documents cover the same subject for (possibly) different entities?"""
    a, b = subject_of(expected_title), subject_of(actual_title)
    if not a or not b:
        return False
    sa = {w for w in a.split() if w not in _STOP}
    sb = {w for w in b.split() if w not in _STOP}
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= SUBJECT_OVERLAP_FLOOR
