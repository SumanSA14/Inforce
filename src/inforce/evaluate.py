"""M5 — measure N.

N is the rate at which the baseline surfaces repealed regulation in answer to a
question about current requirements. It is computed by deterministic set
membership — "is this retrieved document on RBI's published withdrawal list?" —
never by an LLM judge. That is the single most important property of this
measurement: the headline number cannot be talked up by a grader model.

Reported at three strengths, weakest to strongest:

  any_withdrawn   at least one retrieved chunk comes from a withdrawn document
  top1_withdrawn  the highest-ranked source is withdrawn
  trap_hit        the *specific* withdrawn document predicted for this question
                  was retrieved — the strongest claim, because the confusion was
                  named in advance rather than found after the fact

Retrieval-level N is an upper bound on citation-level N: an answer can only
cite what retrieval surfaced.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import entities, rerank, retrieve


@dataclass
class QuestionResult:
    qid: str
    category: str | None
    top1_withdrawn: bool
    any_withdrawn: bool
    withdrawn_chunks: int
    total_chunks: int
    trap_hit: bool
    expected_hit: bool
    expected_rank: int | None
    top_doc_key: str | None
    top_title: str | None


@dataclass
class EvalSummary:
    run_id: str
    k: int
    questions: int = 0
    results: list[QuestionResult] = field(default_factory=list)
    entity_accuracy: float | None = None
    # Abstention, measured on the unanswerable questions.
    unanswerable: int = 0
    abstained: int = 0
    false_abstained: int = 0
    neg_max_score: float | None = None
    pos_min_score: float | None = None

    @property
    def abstention_rate(self) -> float:
        """Correct refusals: unanswerable questions scoring below the threshold."""
        return 100.0 * self.abstained / self.unanswerable if self.unanswerable else 0.0

    @property
    def false_abstention_rate(self) -> float:
        """The cost side. Answerable questions the same threshold would also
        refuse. Reporting abstention without this makes any threshold look good
        — push it high enough and every negative is caught, along with most of
        the real questions."""
        return 100.0 * self.false_abstained / self.questions if self.questions else 0.0

    @property
    def separation(self) -> float | None:
        """Gap between the weakest answerable question and the strongest
        unanswerable one. Positive means the two are cleanly separable by score;
        negative means they overlap and abstention cannot be thresholded."""
        if self.neg_max_score is None or self.pos_min_score is None:
            return None
        return self.pos_min_score - self.neg_max_score

    def _rate(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return 100.0 * sum(bool(getattr(r, attr)) for r in self.results) / len(self.results)

    @property
    def n_any(self) -> float:
        return self._rate("any_withdrawn")

    @property
    def n_top1(self) -> float:
        return self._rate("top1_withdrawn")

    @property
    def n_trap(self) -> float:
        return self._rate("trap_hit")

    @property
    def expected_recall(self) -> float:
        return self._rate("expected_hit")

    @property
    def withdrawn_chunk_share(self) -> float:
        total = sum(r.total_chunks for r in self.results)
        if not total:
            return 0.0
        return 100.0 * sum(r.withdrawn_chunks for r in self.results) / total


SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_run (
    run_id      TEXT PRIMARY KEY,
    label       TEXT,
    mode        TEXT NOT NULL,     -- naive | time_aware
    k           INTEGER NOT NULL,
    questions   INTEGER NOT NULL,
    n_any       REAL,
    n_top1      REAL,
    n_trap      REAL,
    expected_recall REAL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_result (
    run_id           TEXT NOT NULL,
    qid              TEXT NOT NULL,
    category         TEXT,
    top1_withdrawn   INTEGER NOT NULL,
    any_withdrawn    INTEGER NOT NULL,
    withdrawn_chunks INTEGER NOT NULL,
    total_chunks     INTEGER NOT NULL,
    trap_hit         INTEGER NOT NULL,
    expected_hit     INTEGER NOT NULL,
    expected_rank    INTEGER,
    top_doc_key      TEXT,
    PRIMARY KEY (run_id, qid)
);
"""


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def run(
    conn: sqlite3.Connection,
    *,
    k: int = 8,
    mode: str = "naive",
    as_of: str | None = None,
    entity_mode: str = "none",
    retrieval: str = "dense",
    # chunk | doc | doc+expand | doc+expand+llm
    ranking: str = "chunk",
    abstention_threshold: float = 0.73,
    label: str | None = None,
    log=print,
) -> EvalSummary:
    retrieve.check_fresh(conn)
    # time_aware restricts retrieval to documents in force on `as_of`;
    # naive sees the whole corpus, exactly as at M3.
    effective_as_of = as_of if mode == "time_aware" else None

    rows = conn.execute(
        """SELECT qid, text, category, expected_doc_key, trap_doc_keys
           FROM question WHERE answerable = 1 ORDER BY qid"""
    ).fetchall()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{mode}-k{k}"
    summary = EvalSummary(run_id=run_id, k=k, questions=len(rows))
    log(f"running {len(rows)} answerable questions, k={k}, mode={mode}")

    entity_correct = 0
    entity_attempted = 0

    for i, q in enumerate(rows, 1):
        # 'oracle' uses the labelled category — it measures the ceiling if
        # detection were perfect. 'inferred' reads the question text, which is
        # what production would actually do. The gap between them is the cost
        # of detection error and must be reported, not hidden.
        entity = None
        if entity_mode == "oracle":
            entity = q["category"]
        elif entity_mode == "inferred":
            entity = entities.detect(q["text"])
            if q["category"]:
                entity_attempted += 1
                entity_correct += int(entity == q["category"])

        query = rerank.expand_query(q["text"]) if "expand" in ranking else q["text"]
        # Every arm gets the same depth. This previously read
        #     pool = k * 6 if ranking != "chunk" else k
        # which confounded the ablation: the chunk baseline was measured over 8
        # candidate chunks while every doc* arm saw 48, so a recall difference
        # could not be attributed to the scoring change rather than the deeper
        # pool. The chunk arm also collapses chunks to unique documents (below),
        # so it needs identical headroom to fill k document slots. The arms now
        # differ only in the scoring function, which is what is under test.
        pool = k * 6
        hits = retrieve.search(conn, query, k=pool, as_of=effective_as_of,
                               entity=entity, retrieval=retrieval)

        if ranking == "chunk":
            ranked_docs = []
            seen_keys = set()
            for h in hits:
                if h.doc_key not in seen_keys:
                    seen_keys.add(h.doc_key)
                    ranked_docs.append((h.doc_key, h.score, [h]))
        else:
            ranked_docs = rerank.aggregate_documents(hits)
            if "llm" in ranking:
                ranked_docs = rerank.llm_rerank(q["text"], ranked_docs)
        ranked_docs = ranked_docs[:k]
        hits = [g[0] for _key, _s, g in ranked_docs]

        traps = set(json.loads(q["trap_doc_keys"] or "[]"))
        doc_keys = [key for key, _s, _g in ranked_docs]

        expected_rank = None
        if q["expected_doc_key"] in doc_keys:
            expected_rank = doc_keys.index(q["expected_doc_key"]) + 1

        summary.results.append(
            QuestionResult(
                qid=q["qid"],
                category=q["category"],
                top1_withdrawn=bool(hits and hits[0].is_withdrawn),
                any_withdrawn=any(h.is_withdrawn for h in hits),
                # Count the underlying chunks, not the collapsed documents.
                # `hits` holds one representative chunk per document after
                # ranking, so len(hits) counted documents while the column,
                # the property and the CLI line all say "chunks" — the unit
                # changed silently when document ranking was introduced.
                withdrawn_chunks=sum(1 for _k, _s, g in ranked_docs
                                     for h in g if h.is_withdrawn),
                total_chunks=sum(len(g) for _k, _s, g in ranked_docs),
                trap_hit=bool(traps & set(doc_keys)),
                expected_hit=expected_rank is not None,
                expected_rank=expected_rank,
                top_doc_key=hits[0].doc_key if hits else None,
                top_title=hits[0].title if hits else None,
            )
        )
        if i % 10 == 0:
            log(f"  {i}/{len(rows)}")

    # --- abstention -------------------------------------------------------
    # Always probed with DENSE cosine, whatever `retrieval` is set to.
    # Reciprocal rank fusion scores are 1/(k+rank) — position only — so every
    # top-1 lands near 0.033 regardless of how good the match is. RRF discards
    # exactly the magnitude abstention depends on, so thresholding it is
    # meaningless. This is a real consequence of hybrid search, not a detail.
    def _top_dense(question: str) -> float:
        hits = retrieve.search(conn, question, k=1, as_of=effective_as_of,
                               retrieval="dense")
        return hits[0].score if hits else 0.0

    neg_rows = conn.execute(
        "SELECT text FROM question WHERE answerable = 0").fetchall()
    if neg_rows:
        neg = [_top_dense(r["text"]) for r in neg_rows]
        pos = [_top_dense(r["text"]) for r in rows]
        summary.unanswerable = len(neg)
        summary.neg_max_score = max(neg)
        summary.pos_min_score = min(pos) if pos else None
        summary.abstained = sum(s < abstention_threshold for s in neg)
        summary.false_abstained = sum(s < abstention_threshold for s in pos)
        log(f"abstention: {summary.abstained}/{len(neg)} negatives refused, "
            f"{summary.false_abstained}/{len(pos)} answerable wrongly refused "
            f"(neg max {max(neg):.3f}, answerable min {min(pos):.3f})")

    if entity_mode == "inferred" and entity_attempted:
        acc = 100.0 * entity_correct / entity_attempted
        log(f"entity detection accuracy: {entity_correct}/{entity_attempted} = {acc:.1f}%")
        summary.entity_accuracy = acc

    tag = mode if entity_mode == "none" else f"{mode}+{entity_mode}"
    if retrieval != "dense":
        tag = f"{tag}+{retrieval}"
    if ranking != "chunk":
        tag = f"{tag}+{ranking}"
    store(conn, summary, mode=tag, label=label)
    return summary


def store(conn: sqlite3.Connection, s: EvalSummary, *, mode: str, label: str | None) -> None:
    init(conn)
    conn.execute(
        """INSERT OR REPLACE INTO eval_run
           (run_id, label, mode, k, questions, n_any, n_top1, n_trap,
            expected_recall, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (s.run_id, label, mode, s.k, s.questions, s.n_any, s.n_top1, s.n_trap,
         s.expected_recall, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.executemany(
        """INSERT OR REPLACE INTO eval_result
           (run_id, qid, category, top1_withdrawn, any_withdrawn, withdrawn_chunks,
            total_chunks, trap_hit, expected_hit, expected_rank, top_doc_key)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [(s.run_id, r.qid, r.category, int(r.top1_withdrawn), int(r.any_withdrawn),
          r.withdrawn_chunks, r.total_chunks, int(r.trap_hit), int(r.expected_hit),
          r.expected_rank, r.top_doc_key) for r in s.results],
    )
    conn.commit()
