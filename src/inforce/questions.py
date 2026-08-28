"""M4 — the golden question set, and the analysis that seeds it.

Writing the questions is human work and stays human work: an LLM-generated
question set graded by an LLM measures nothing. What *can* be automated is
finding where to look — the pairs of documents, one in force and one withdrawn,
that cover the same ground closely enough that retrieval will confuse them.
Those pairs are where N lives.

A question only earns its place in the set if a withdrawn document and a
current Master Direction could *both* plausibly answer it. A question that only
the current document can answer measures nothing about time.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import entities, retrieve

# A pair only counts as confusable if chunks are this close. Tuned to be
# permissive: false positives cost a human glance, false negatives cost a
# question that would have exposed a real failure.
SIMILARITY_FLOOR = 0.75
NEIGHBOURS_PER_CHUNK = 5
BLOCK = 512

# Boilerplate suppression. RBI documents share a standard preamble ("In exercise
# of the powers conferred by Section 35A..."), definitions and repeal clauses, so
# those chunks match almost everything at ~1.0 while carrying no topical signal.
# Left unfiltered they dominate the ranking: one withdrawn document was matching
# 195 distinct in-force documents at similarity 1.000, which is impossible
# topically. A withdrawn chunk selected by more than this many distinct in-force
# documents is treated as boilerplate and dropped.
BOILERPLATE_DOC_SPREAD = 25


@dataclass
class Pair:
    in_force_doc_key: str
    withdrawn_doc_key: str
    max_similarity: float
    mean_similarity: float
    matches: int


@dataclass
class Question:
    qid: str
    text: str
    topic: str | None = None
    category: str | None = None
    answerable: bool = True
    expected_doc_key: str | None = None
    expected_quote: str | None = None
    trap_doc_keys: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    notes: str | None = None


class ValidationError(ValueError):
    pass


_SQUASH = __import__("re").compile(r"\s+")

LEAKAGE_TOKEN_OVERLAP = 0.60

_STOPWORDS = frozenset(
    "what which how the a an of to for in on is are must be and or at by does do "
    "apply applies under with that this from shall its it their there when who".split()
)


def _content_tokens(text: str) -> list[str]:
    import re as _re
    return [w for w in _re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOPWORDS]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()


def _squash(text: str) -> str:
    """Whitespace- and case-insensitive comparison. PDF and HTML extraction
    disagree about spacing, so an exact match would fail on correct quotes."""
    return _SQUASH.sub(" ", text or "").strip().lower()


# Sentences carrying a number, percentage or rupee amount make the best
# questions: unambiguous, checkable, and a stale document typically states a
# *different* value rather than none — the strongest possible demonstration.
_FACTUAL = __import__("re").compile(
    r"[^.]*?(?:\b\d+(?:\.\d+)?\s*(?:per\s*cent|percent|%)"
    r"|₹\s*[\d,]+|\brupees\b|\bnot\s+less\s+than\b|\bminimum\b|\bshall\s+not\s+exceed\b)"
    r"[^.]*\.",
    __import__("re").IGNORECASE,
)


def propose_facts(
    conn: sqlite3.Connection, doc_key: str, *, limit: int = 12, min_len: int = 60
) -> list[str]:
    """Candidate factual sentences from a document, for grounding draft
    questions. This finds material; a human still decides what to ask."""
    row = conn.execute(
        "SELECT body_text FROM document WHERE doc_key = ?", (doc_key,)
    ).fetchone()
    if not row or not row[0]:
        return []
    text = _SQUASH.sub(" ", row[0])
    out, seen = [], set()
    for match in _FACTUAL.finditer(text):
        sentence = match.group(0).strip()
        if len(sentence) < min_len or len(sentence) > 400:
            continue
        key = sentence[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence)
        if len(out) >= limit:
            break
    return out


def _chunk_status_map(conn: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    return {
        r["chunk_id"]: (r["doc_key"], r["status"])
        for r in conn.execute(
            """SELECT c.chunk_id, c.doc_key, d.status
               FROM chunk c JOIN document d ON d.doc_key = c.doc_key
               WHERE c.embedding IS NOT NULL"""
        )
    }


def find_confusable_pairs(
    conn: sqlite3.Connection,
    *,
    floor: float = SIMILARITY_FLOOR,
    neighbours: int = NEIGHBOURS_PER_CHUNK,
    log=print,
) -> list[Pair]:
    """For every in-force chunk, find its nearest withdrawn chunks, then
    aggregate to document pairs.

    Done as a blocked matmul: the full cross product would be ~32k x 64k, which
    is 8 GB as float32. Blocking keeps peak memory in the low hundreds of MB.
    """
    matrix, ids = retrieve.load_index()
    meta = _chunk_status_map(conn)

    rows = np.array([meta.get(int(cid), ("", ""))[1] for cid in ids])
    keys = np.array([meta.get(int(cid), ("", ""))[0] for cid in ids])

    in_force_idx = np.flatnonzero(rows == "in_force")
    withdrawn_idx = np.flatnonzero(rows == "withdrawn")
    if in_force_idx.size == 0 or withdrawn_idx.size == 0:
        log("need both in-force and withdrawn chunks embedded")
        return []

    log(f"in-force chunks: {in_force_idx.size:,}   withdrawn: {withdrawn_idx.size:,}")
    withdrawn_matrix = np.asarray(matrix[withdrawn_idx], dtype=np.float32)

    # Pass 1: collect chunk-level hits. Bounded by (in-force chunks x
    # neighbours), so a few hundred thousand at most — cheap to hold.
    hits: list[tuple[str, int, float]] = []  # (in_force_doc, withdrawn_chunk_col, score)
    for start in range(0, in_force_idx.size, BLOCK):
        block_idx = in_force_idx[start : start + BLOCK]
        block = np.asarray(matrix[block_idx], dtype=np.float32)
        scores = block @ withdrawn_matrix.T  # (block, withdrawn)

        k = min(neighbours, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        for row in range(block.shape[0]):
            src_key = keys[block_idx[row]]
            for col in top[row]:
                score = float(scores[row, col])
                if score >= floor:
                    hits.append((src_key, int(col), score))

        if (start // BLOCK) % 10 == 0:
            log(f"  scanned {min(start+BLOCK, in_force_idx.size):,}/"
                f"{in_force_idx.size:,} in-force chunks, {len(hits):,} hits")

    # Pass 2: suppress boilerplate. A withdrawn chunk that many *different*
    # in-force documents all point at is shared scaffolding, not a topic match.
    spread: dict[int, set[str]] = {}
    for src_key, col, _ in hits:
        spread.setdefault(col, set()).add(src_key)
    boilerplate = {col for col, docs in spread.items() if len(docs) > BOILERPLATE_DOC_SPREAD}
    kept = [h for h in hits if h[1] not in boilerplate]
    log(f"  boilerplate chunks suppressed: {len(boilerplate):,} "
        f"({len(hits) - len(kept):,} of {len(hits):,} hits dropped)")

    agg: dict[tuple[str, str], list[float]] = {}
    for src_key, col, score in kept:
        agg.setdefault((src_key, keys[withdrawn_idx[col]]), []).append(score)

    pairs = [
        Pair(a, b, max(v), float(np.mean(v)), len(v))
        for (a, b), v in agg.items()
    ]
    pairs.sort(key=lambda p: (p.matches, p.max_similarity), reverse=True)
    return pairs


def store_pairs(conn: sqlite3.Connection, pairs: list[Pair]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute("DELETE FROM confusable_pair")
    conn.executemany(
        """INSERT INTO confusable_pair
           (in_force_doc_key, withdrawn_doc_key, max_similarity,
            mean_similarity, matches, computed_at)
           VALUES (?,?,?,?,?,?)""",
        [(p.in_force_doc_key, p.withdrawn_doc_key, p.max_similarity,
          p.mean_similarity, p.matches, now) for p in pairs],
    )
    conn.commit()


def sibling_index(conn: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    """doc_key -> in-force documents on the same subject for other entity classes.

    RBI issues parallel Directions per entity class, so "RBI (Commercial Banks -
    Know Your Customer) Directions" has ten near-identical siblings. Built once
    and reused: the naive form is a full cross product over 3,645 documents.
    """
    docs = [
        (r["doc_key"], r["title"], r["category"])
        for r in conn.execute(
            "SELECT doc_key, title, category FROM document WHERE status='in_force'")
    ]
    buckets: dict[str, list[tuple[str, str]]] = {}
    for key, title, cat in docs:
        subj = entities.subject_of(title)
        if subj:
            buckets.setdefault(subj, []).append((key, cat or ""))
    out: dict[str, list[tuple[str, str]]] = {}
    for key, title, cat in docs:
        subj = entities.subject_of(title)
        if not subj:
            continue
        out[key] = [(k, c) for k, c in buckets[subj] if k != key and c != (cat or "")]
    return out


def validate(conn: sqlite3.Connection, questions: list[Question]) -> list[str]:
    """Structural checks only. Whether a question is *good* is a human call."""
    problems: list[str] = []
    seen: set[str] = set()
    siblings = sibling_index(conn)

    known = {r[0] for r in conn.execute("SELECT doc_key FROM document")}
    in_force = {
        r[0] for r in conn.execute("SELECT doc_key FROM document WHERE status='in_force'")
    }
    withdrawn = {
        r[0] for r in conn.execute("SELECT doc_key FROM document WHERE status='withdrawn'")
    }

    for q in questions:
        if q.qid in seen:
            problems.append(f"{q.qid}: duplicate qid")
        seen.add(q.qid)

        if not q.text.strip():
            problems.append(f"{q.qid}: empty question text")
        if len(q.text) < 20:
            problems.append(f"{q.qid}: suspiciously short question")

        if q.answerable:
            if not q.expected_doc_key:
                problems.append(f"{q.qid}: answerable but no expected_doc_key")
            elif q.expected_doc_key not in known:
                problems.append(f"{q.qid}: expected_doc_key not in corpus")
            elif q.expected_doc_key not in in_force:
                problems.append(
                    f"{q.qid}: expected_doc_key is not in force — the expected "
                    "answer must be current law"
                )
        elif q.expected_doc_key:
            problems.append(f"{q.qid}: marked unanswerable but has an expected_doc_key")

        # The quote is what makes an answer verifiable rather than asserted.
        # If it does not literally appear in the expected document, the question
        # is wrong no matter how plausible it reads.
        if q.expected_quote and q.expected_doc_key:
            row = conn.execute(
                "SELECT body_text FROM document WHERE doc_key = ?", (q.expected_doc_key,)
            ).fetchone()
            if row and row[0]:
                haystack = _squash(row[0])
                if _squash(q.expected_quote) not in haystack:
                    problems.append(
                        f"{q.qid}: expected_quote does not appear in "
                        f"{q.expected_doc_key} — the answer is unverified"
                    )

        # Answer leakage. A question that reuses the source sentence's wording
        # is easier to retrieve than anything a real user would type, which
        # flatters the system and understates N.
        if q.expected_quote and q.text:
            q_tokens = _content_tokens(q.text)
            a_tokens = _content_tokens(q.expected_quote)
            if q_tokens:
                overlap = len(set(q_tokens) & set(a_tokens)) / len(set(q_tokens))
                shared = _ngrams(q_tokens, 4) & _ngrams(a_tokens, 4)
                if overlap > LEAKAGE_TOKEN_OVERLAP or shared:
                    problems.append(
                        f"{q.qid}: leaks the source wording "
                        f"({overlap:.0%} token overlap, {len(shared)} shared 4-grams) "
                        "— rephrase as a user would actually ask it"
                    )

        # Discrimination. The gate that was missing, and the reason roughly a
        # third of the original set could not identify its own gold document.
        #
        # If the gold document has in-force siblings — parallel Directions on
        # the same subject for other entity classes — then a question that does
        # not name its entity cannot distinguish them. "If a bank is wound up,
        # how much of a depositor's money is protected?" was labelled Regional
        # Rural Banks and scored wrong for returning Rural Co-operative Banks,
        # but DICGC cover is Rs 5 lakh either way and the question never says
        # which bank. No system, and no human, could pick the labelled one.
        #
        # Such a question does not measure retrieval; it measures whether the
        # retriever guessed the annotator's arbitrary choice. Both the failures
        # AND the lucky passes are meaningless, so this is checked for every
        # question rather than only the ones that happen to fail.
        if q.answerable and q.expected_doc_key and q.category:
            kin = siblings.get(q.expected_doc_key, [])
            if kin and entities.detect(q.text) != q.category:
                others = sorted({c for _k, c in kin if c})[:3]
                problems.append(
                    f"{q.qid}: under-specified — {len(kin)} in-force Direction(s) "
                    f"cover this subject for other entity classes "
                    f"({', '.join(others)}...), and the question never names "
                    f"{q.category!r}. The gold label is arbitrary; a sibling "
                    "answers it equally well. Name the entity in the question."
                )

        for trap in q.trap_doc_keys:
            if trap not in known:
                problems.append(f"{q.qid}: trap doc {trap} not in corpus")
            elif trap not in withdrawn:
                problems.append(f"{q.qid}: trap doc {trap} is not withdrawn")

        # NOTE: traps are deliberately NOT required.
        #
        # An earlier version of this rule rejected any answerable question with
        # no trap, on the reasoning that it could not expose temporal confusion.
        # That was wrong once traps became *observed* rather than *predicted*:
        # a question where retrieval surfaces no withdrawn document is a case
        # the system handled correctly, and dropping those filters the set down
        # to questions that already exhibit the failure — which inflates N by
        # construction. Successes must stay in the denominator.

    return problems


def load_jsonl(path) -> list[Question]:
    out: list[Question] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"line {lineno}: {exc}") from exc
        out.append(
            Question(
                qid=raw["qid"],
                text=raw["text"],
                topic=raw.get("topic"),
                category=raw.get("category"),
                answerable=bool(raw.get("answerable", True)),
                expected_doc_key=raw.get("expected_doc_key"),
                expected_quote=raw.get("expected_quote"),
                trap_doc_keys=list(raw.get("trap_doc_keys", [])),
                difficulty=raw.get("difficulty", "medium"),
                notes=raw.get("notes"),
            )
        )
    return out


def store_questions(conn: sqlite3.Connection, questions: list[Question]) -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        """INSERT INTO question
           (qid, text, topic, category, answerable, expected_doc_key,
            expected_quote, trap_doc_keys, difficulty, notes, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(qid) DO UPDATE SET
             text=excluded.text, topic=excluded.topic, category=excluded.category,
             answerable=excluded.answerable, expected_doc_key=excluded.expected_doc_key,
             expected_quote=excluded.expected_quote, trap_doc_keys=excluded.trap_doc_keys,
             difficulty=excluded.difficulty, notes=excluded.notes""",
        [
            (q.qid, q.text, q.topic, q.category, 1 if q.answerable else 0,
             q.expected_doc_key, q.expected_quote, json.dumps(q.trap_doc_keys),
             q.difficulty, q.notes, now)
            for q in questions
        ],
    )
    conn.commit()
    return len(questions)


def verify_traps(conn: sqlite3.Connection, run_id: str) -> dict[str, bool]:
    """Which designated traps were actually retrieved by the naive baseline.

    A trap is a *prediction* that a specific withdrawn document will be
    confused with the live one. Validation currently only checks the field is
    populated and points at a withdrawn document — it cannot check the
    prediction was right. Measured against the first naive run, 28 of 36 traps
    were never retrieved, so most designations were unfounded.

    The durable fix is to treat traps as **observed rather than predicted**:
    derive them from what the baseline actually surfaced. `suggest_traps`
    below does that.
    """
    return {
        r["qid"]: bool(r["trap_hit"])
        for r in conn.execute(
            "SELECT qid, trap_hit FROM eval_result WHERE run_id = ?", (run_id,)
        )
    }


def suggest_traps(conn: sqlite3.Connection, run_id: str, qid: str, limit: int = 3) -> list[str]:
    """Withdrawn documents the naive baseline actually retrieved for a question.

    These are evidence-based traps: documents observed competing with the
    correct answer, rather than ones guessed from a similarity table.
    """
    row = conn.execute(
        "SELECT top_doc_key FROM eval_result WHERE run_id=? AND qid=?", (run_id, qid)
    ).fetchone()
    if not row:
        return []
    return [
        r[0] for r in conn.execute(
            """SELECT d.doc_key FROM document d
               WHERE d.status='withdrawn' AND d.doc_key = ?""", (row["top_doc_key"],)
        ).fetchall()
    ][:limit]


def orphans(conn: sqlite3.Connection, questions: list[Question]) -> list[str]:
    """Questions in the database that are no longer in the file.

    The file is the source of truth. Left unnoticed, a dropped question keeps
    scoring — so N would be computed over a set the author no longer endorses.
    """
    in_file = {q.qid for q in questions}
    return [
        r[0] for r in conn.execute("SELECT qid FROM question").fetchall()
        if r[0] not in in_file
    ]


def prune(conn: sqlite3.Connection, qids: list[str]) -> int:
    if not qids:
        return 0
    conn.executemany("DELETE FROM question WHERE qid = ?", [(q,) for q in qids])
    conn.commit()
    return len(qids)
