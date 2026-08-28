"""InForce CLI.

    python -m inforce.cli m1              # parse from cache (or fetch once)
    python -m inforce.cli m1 --refresh    # re-fetch the annex
    python -m inforce.cli status          # what's in the DB
"""
from __future__ import annotations

import argparse
import io
import sys

from . import (config, crawl, embedding, evaluate, fetch, generate, index,
               lexical, questions, retrieve, store, temporal)
from .annex import parse_annex


def _utf8_stdout() -> None:
    """RBI subjects contain en-dashes and the rupee sign; the default Windows
    console encoding cannot represent them and would raise mid-report."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )


def cmd_m1(args: argparse.Namespace) -> int:
    print(f"source : {config.ANNEX_URL}")
    html, from_cache = fetch.fetch_text(
        config.ANNEX_URL, config.ANNEX_CACHE, refresh=args.refresh
    )
    sha1 = fetch.content_sha1(html)
    print(f"payload: {len(html):,} chars  ({'cache' if from_cache else 'network'})")
    print(f"sha1   : {sha1}")
    print(f"cached : {config.ANNEX_CACHE}")

    rows, report = parse_annex(html)

    print(f"\ntables seen        : {report.tables_seen}")
    print(f"rows parsed        : {report.total_rows:,}")
    print(f"rows skipped       : {report.skipped_rows}  (headers / footnote)")
    print(f"deferred repeal    : {report.deferred_repeal_rows}  (repealed {config.DEFERRED_REPEAL_DATE})")
    print(f"duplicate content  : {report.duplicate_content_rows}  (kept, disambiguated)")
    print(f"unknown withdrawal : {report.unknown_withdrawal_date}  (batch date was impossible)")
    print(f"unparsed dates     : {len(report.unparsed_dates)}")
    if report.unparsed_dates:
        print(f"  samples          : {report.unparsed_dates[:5]}")

    print("\nper batch:")
    for batch, count in report.rows_by_batch.items():
        linked = report.linked_by_batch.get(batch, 0)
        pct = (linked / count * 100) if count else 0.0
        print(f"  {batch:<28} rows={count:>6,}  linked={linked:>5,} ({pct:4.1f}%)")

    total = report.total_rows
    linked_pct = (report.total_linked / total * 100) if total else 0.0
    print(f"  {'TOTAL':<28} rows={total:>6,}  linked={report.total_linked:>5,} ({linked_pct:4.1f}%)")

    if report.warnings:
        print("\nWARNINGS:")
        for w in report.warnings:
            print(f"  ! {w}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    conn = store.connect(args.db)
    store.init_schema(conn)
    inserted, updated = store.upsert_rows(conn, rows)
    store.record_run(
        conn,
        source_url=config.ANNEX_URL,
        from_cache=from_cache,
        content_sha1=sha1,
        rows_parsed=len(rows),
        rows_inserted=inserted,
        rows_updated=updated,
        warnings="; ".join(report.warnings) or None,
    )
    print(f"\ndb     : {args.db or config.DB_PATH}")
    print(f"written: {inserted:,} inserted, {updated:,} updated")
    conn.close()
    return 0


def _report_crawl(stats) -> None:
    print(
        f"\nattempted={stats.attempted:,}  ok={stats.ok:,}  thin={stats.thin}  "
        f"pdf-fallback={stats.from_pdf}  errors={stats.errors}  "
        f"(network={stats.from_network:,}, cache={stats.from_cache:,})"
    )
    for sample in stats.error_samples:
        print(f"  ! {sample}")


def cmd_notifications(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    stats = crawl.crawl_notifications(
        conn, limit=args.limit, delay=args.delay, refresh=args.refresh
    )
    _report_crawl(stats)
    conn.close()
    return 0


def cmd_master_directions(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    stats, report = crawl.crawl_master_directions(
        conn,
        download_pdfs=not args.no_pdfs,
        limit=args.limit,
        delay=args.delay,
        refresh=args.refresh,
    )
    if report.unclassified_single_cells:
        print(f"\nunclassified single-cell rows: {len(report.unclassified_single_cells)}")
        for s in report.unclassified_single_cells[:5]:
            print(f"  ? {s}")
    for w in report.warnings:
        print(f"  ! {w}")
    _report_crawl(stats)
    conn.close()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    ok, msg = embedding.health()
    if not ok:
        print(f"ollama: {msg}")
        return 1

    conn = store.connect(args.db)
    store.init_schema(conn)

    if args.rebuild_matrix:
        rows = index.build_matrix(conn)
        print(f"matrix rebuilt: {rows:,} rows")
        conn.close()
        return 0

    docs, chunks = index.chunk_documents(conn, limit=args.limit_docs)
    print(f"chunked {docs:,} documents -> {chunks:,} new chunks")

    if not args.no_embed:
        n = index.embed_chunks(conn, limit=args.limit_chunks)
        print(f"embedded {n:,} chunks")
        index.build_matrix(conn)

    if not args.no_fts:
        lexical.build(conn)

    conn.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    try:
        hits = retrieve.search(conn, args.question, k=args.k, allow_stale=args.allow_stale)
    except (retrieve.IndexNotBuilt, retrieve.IndexStale) as exc:
        print(exc)
        return 1

    withdrawn = [h for h in hits if h.is_withdrawn]
    print(f"\nQ: {args.question}\n")
    print(f"retrieved {len(hits)} chunks — "
          f"{len(withdrawn)} from WITHDRAWN documents, "
          f"{len(hits)-len(withdrawn)} in force")
    if hits and hits[0].is_withdrawn:
        print("top-1 source is WITHDRAWN")
    print()

    for h in hits:
        flag = "WITHDRAWN" if h.is_withdrawn else "in force "
        extra = f" (withdrawn {h.withdrawn_on})" if h.withdrawn_on else ""
        print(f"[{h.rank}] {h.score:.3f}  {flag}{extra}")
        print(f"     {(h.title or h.doc_key)[:100]}")
        print(f"     {h.doc_date or '?'}  {h.rbi_ref or ''}")
        print(f"     {h.text[:160]}...")
        print()

    if not args.no_llm:
        print("--- answer (naive baseline, no date awareness) ---")
        print(generate.answer(args.question, hits))
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    try:
        retrieve.check_fresh(conn)
    except retrieve.IndexStale as exc:
        print(exc)
        return 1

    pairs = questions.find_confusable_pairs(conn)
    pairs = [p for p in pairs if p.matches >= args.min_matches]
    questions.store_pairs(conn, pairs)
    print(f"\nconfusable pairs found: {len(pairs):,}\n")

    titles = {
        r["doc_key"]: (r["title"], r["doc_date"], r["status"])
        for r in conn.execute("SELECT doc_key, title, doc_date, status FROM document")
    }
    for p in pairs[: args.top]:
        live = titles.get(p.in_force_doc_key, ("?", "?", "?"))
        dead = titles.get(p.withdrawn_doc_key, ("?", "?", "?"))
        print(f"[{p.matches:>3} matches  max={p.max_similarity:.3f}]")
        print(f"   IN FORCE : {(live[0] or '?')[:92]}")
        print(f"              {live[1]}")
        print(f"   WITHDRAWN: {(dead[0] or '?')[:92]}")
        print(f"              {dead[1]}")
        print()
    conn.close()
    return 0


def cmd_abstention(args: argparse.Namespace) -> int:
    """Measure whether the GENERATOR refuses, not whether a score threshold does.

    The existing abstention metric thresholds the top retrieval score, and the
    project's own numbers show that cannot work: answerable and unanswerable
    score ranges overlap (separation -0.038), and under RRF every top-1 lands
    near 0.033 because fusion encodes position, not similarity. So the reported
    metric measures a mechanism that provably does not separate.

    The mechanism that does work is the generator declining when the retrieved
    excerpts do not contain the answer. This measures that, with the
    false-refusal rate alongside it -- a system that refuses everything scores
    100% on abstention and is useless, so reporting one without the other is
    meaningless.
    """
    import pathlib
    import random
    import sqlite3

    from . import abstain, entities, generate, retrieve

    conn = store.connect(args.db)
    conn.row_factory = sqlite3.Row

    neg = [dict(r) for r in conn.execute(
        "SELECT qid, text FROM question WHERE answerable = 0")]
    pos = [dict(r) for r in conn.execute(
        "SELECT qid, text, expected_doc_key FROM question WHERE answerable = 1")]
    random.seed(args.seed)
    if args.sample and args.sample < len(pos):
        pos = random.sample(pos, args.sample)

    if not neg:
        print("no unanswerable questions stored — nothing to measure")
        return 1
    print(f"unanswerable: {len(neg)}   answerable sample: {len(pos)}   "
          f"({len(neg) + len(pos)} generation calls)")

    def probe(q):
        ent = entities.detect(q["text"])
        hits = retrieve.search(conn, q["text"], k=args.k, as_of=args.as_of,
                               entity=ent, retrieval=args.retrieval)
        # Whether retrieval actually surfaced the gold document. Without this
        # the false-refusal rate blames the generator for retrieval's misses:
        # when the answer was never retrieved, declining is the CORRECT
        # behaviour, and counting it as a false refusal would penalise the one
        # component that behaved properly.
        retrieved = (q.get("expected_doc_key") is not None
                     and any(h.doc_key == q["expected_doc_key"] for h in hits))
        try:
            text, _verdict = generate.answer(q["text"], hits, timeout=args.timeout,
                                             gate=not args.no_gate)
        except Exception as exc:  # noqa: BLE001 - a failed call is not a refusal
            return None, f"error: {exc}", retrieved
        return abstain.classify(text), text, retrieved

    rows = []
    for label, group in (("unanswerable", neg), ("answerable", pos)):
        for i, q in enumerate(group, 1):
            cls, text, retrieved = probe(q)
            rows.append({"qid": q["qid"], "group": label, "class": cls,
                         "retrieved": retrieved, "question": q["text"],
                         "text": text or ""})
            if i % 10 == 0:
                print(f"  {label} {i}/{len(group)}", flush=True)

    def rate(group, cls, only_retrieved=False):
        g = [r for r in rows if r["group"] == group and r["class"] is not None]
        if only_retrieved:
            g = [r for r in g if r["retrieved"]]
        return (100.0 * sum(r["class"] == cls for r in g) / len(g)) if g else 0.0

    def count(group, only_retrieved=False):
        g = [r for r in rows if r["group"] == group and r["class"] is not None]
        return len([r for r in g if r["retrieved"]]) if only_retrieved else len(g)

    errors = [r for r in rows if r["class"] is None]
    print()
    print("=" * 62)
    print("GENERATION-LAYER ABSTENTION"
          + ("  [no answerability gate]" if args.no_gate else "  [gated]"))
    print("=" * 62)
    print(f"  unanswerable refused        : {rate('unanswerable', 'refused'):5.1f}%  "
          f"(correct)")
    print(f"  unanswerable partial        : {rate('unanswerable', 'partial'):5.1f}%")
    print(f"  unanswerable answered       : {rate('unanswerable', 'answered'):5.1f}%  "
          f"(fabrication risk)")
    print(f"  answerable wrongly refused  : {rate('answerable', 'refused'):5.1f}%  "
          f"(all {count('answerable')} sampled)")
    print(f"  answerable answered         : {rate('answerable', 'answered'):5.1f}%")
    print()
    print("  Conditioned on retrieval actually surfacing the gold document")
    print("  (declining when the answer was never retrieved is correct, not a fault):")
    print(f"    gold retrieved             : {count('answerable', True)}"
          f"/{count('answerable')}")
    print(f"    of those, wrongly refused  : "
          f"{rate('answerable', 'refused', True):5.1f}%  <- the generator's own"
          f" false-refusal rate")
    print(f"    of those, answered         : "
          f"{rate('answerable', 'answered', True):5.1f}%")
    if errors:
        print(f"  generation errors           : {len(errors)} (excluded from rates)")
    print()
    print("  A system that refuses everything scores 100% on the first line.")
    print("  Read it against the false-refusal rate or it means nothing.")

    # Persist the raw answers. Reclassifying after a detector fix then costs
    # nothing; without this, every change to the refusal patterns meant
    # re-running every generation call, which is the expensive part.
    if args.save:
        import json
        out = pathlib.Path(args.save)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print()
        print(f"  raw answers written to {out}")

    if args.detail:
        print()
        print("--- unanswerable questions that were ANSWERED ---")
        for r in rows:
            if r["group"] == "unanswerable" and r["class"] == "answered":
                print(f"  [{r['qid']}] {' '.join(r['text'].split())[:170]}")
    conn.close()
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    import pathlib

    conn = store.connect(args.db)
    store.init_schema(conn)

    if args.file:
        qs = questions.load_jsonl(pathlib.Path(args.file))
        problems = questions.validate(conn, qs)
        if problems:
            print(f"{len(problems)} validation problem(s):")
            for p in problems:
                print(f"  ! {p}")
            if not args.force:
                print("\nnothing stored — fix these or pass --force")
                return 1
        n = questions.store_questions(conn, qs)
        print(f"stored {n} questions ({len(problems)} problems)")

        stale = questions.orphans(conn, qs)
        if stale:
            if args.prune:
                print(f"pruned {questions.prune(conn, stale)} orphaned: {stale}")
            else:
                print(f"\n! {len(stale)} question(s) in the database but not in the file:")
                print(f"    {stale}")
                print("  they will still be scored — pass --prune to remove them")

    rows = conn.execute(
        """SELECT answerable, difficulty, COUNT(*) n FROM question
           GROUP BY answerable, difficulty ORDER BY answerable DESC, difficulty"""
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM question").fetchone()[0]
    print(f"\nquestion set: {total} total")
    for r in rows:
        kind = "answerable" if r["answerable"] else "unanswerable"
        print(f"  {kind:<14} {r['difficulty']:<8} {r['n']:>4}")
    conn.close()
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    evaluate.init(conn)
    try:
        s = evaluate.run(conn, k=args.k, mode=args.mode, as_of=args.as_of,
                         entity_mode=args.entity, retrieval=args.retrieval,
                         ranking=args.ranking, label=args.label)
    except (retrieve.IndexNotBuilt, retrieve.IndexStale) as exc:
        print(exc)
        return 1

    bar = "=" * 62
    print(f"\n{bar}\n  N — naive baseline over RBI regulation "
          f"({s.questions} questions, k={s.k})\n{bar}")
    print(f"  any retrieved source withdrawn : {s.n_any:5.1f}%")
    print(f"  TOP-1 source withdrawn         : {s.n_top1:5.1f}%")
    print(f"  predicted trap retrieved       : {s.n_trap:5.1f}%")
    print(f"  withdrawn share of all chunks  : {s.withdrawn_chunk_share:5.1f}%")
    print(f"  expected in-force doc in top-{s.k}  : {s.expected_recall:5.1f}%")
    if s.unanswerable:
        sep = s.separation
        print(f"  abstention on unanswerable     : {s.abstention_rate:5.1f}% "
              f"({s.abstained}/{s.unanswerable})")
        print(f"  answerable wrongly refused     : {s.false_abstention_rate:5.1f}% "
              f"({s.false_abstained}/{s.questions})")
        print(f"  score separation               : {sep:+.3f}"
              f"  {'separable' if sep and sep > 0 else 'OVERLAPPING — not thresholdable'}")
    print(bar)

    by_cat: dict[str, list] = {}
    for r in s.results:
        by_cat.setdefault(r.category or "?", []).append(r)
    print("\nby category (top-1 withdrawn):")
    for cat, rs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        hit = sum(r.top1_withdrawn for r in rs)
        print(f"  {cat:<40} {hit}/{len(rs)}")

    if args.detail:
        print("\nper question:")
        for r in sorted(s.results, key=lambda r: r.qid):
            flag = "STALE" if r.top1_withdrawn else "  ok "
            rank = f"exp@{r.expected_rank}" if r.expected_rank else "exp MISS"
            print(f"  [{flag}] {r.qid:<16} withdrawn {r.withdrawn_chunks}/{r.total_chunks}  "
                  f"{rank:<9} trap={'Y' if r.trap_hit else 'n'}")
            print(f"          top: {(r.top_title or r.top_doc_key or '?')[:88]}")

    print(f"\nrun_id: {s.run_id}")
    conn.close()
    return 0


def cmd_temporal(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)

    if args.build:
        added = temporal.migrate(conn)
        print(f"columns added: {added or '(already present)'}")
        temporal.backfill(conn)
        temporal.infer_supersession_chains(conn)

    dates = args.as_of or ["2018-06-30", "2022-03-14", "2025-11-27",
                           "2025-11-29", "2026-08-08"]
    header = f"{'as of':<14}{'in force':>10}{'withdrawn':>11}{'unknown':>9}{'not yet':>9}"
    print(f"\n{header}\n{'-' * len(header)}")
    for d in dates:
        s = temporal.snapshot(conn, d)
        print(f"{d:<14}{s.in_force:>10,}{s.withdrawn:>11,}{s.unknown:>9,}{s.not_yet:>9,}")

    if args.build:
        row = conn.execute(
            """SELECT COUNT(*) n, COUNT(DISTINCT withdrawn_doc_key) d,
                      AVG(confidence) c FROM supersession
               WHERE method='inferred_centroid'"""
        ).fetchone()
        print(f"\nsupersession edges: {row['n']:,} over {row['d']:,} documents "
              f"(mean confidence {row['c'] or 0:.3f})")
        print("  NOTE: inferred from text similarity. RBI published which "
              "circulars were withdrawn,\n        never where each one went — "
              "these are not published fact.")
    conn.close()
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from . import agent

    conn = store.connect(args.db)
    store.init_schema(conn)
    try:
        state = agent.run(conn, args.question, today=args.today)
    except (retrieve.IndexNotBuilt, retrieve.IndexStale) as exc:
        print(exc)
        return 1

    titles = {r["doc_key"]: r["title"]
              for r in conn.execute("SELECT doc_key, title FROM document")}
    status = {r["doc_key"]: r["status"]
              for r in conn.execute("SELECT doc_key, status FROM document")}

    print(f"\nQ: {args.question}\n")
    for n in state["notes"]:
        print(f"  · {n}")

    print(f"\nIN-FORCE ANSWER ({len(state['hits'])} sources)")
    for h in state["hits"][:4]:
        print(f"  [{h.rank}] {(h.title or h.doc_key)[:88]}")

    chains = [c for c in state["chains"] if len(c) > 1]
    if chains:
        print(f"\nSUPERSESSION TRACE ({len(chains)} chain(s), {state['hops']} hops)")
        for chain in chains[: args.max_chains]:
            for depth, key in enumerate(chain):
                flag = "LIVE " if status.get(key) == "in_force" else "dead "
                arrow = "    " if depth == 0 else " -> "
                print(f"  {arrow}[{flag}] {(titles.get(key) or key)[:80]}")
            print()

    if state.get("conflicts"):
        print(f"AMBIGUOUS: {len(state['conflicts'])} different live documents reached")
        for k in state["conflicts"]:
            print(f"  - {(titles.get(k) or k)[:88]}")
    conn.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server
    print(f"InForce demo -> http://{args.host}:{args.port}")
    server.serve(host=args.host, port=args.port)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    rows = store.summary(conn)
    if not rows:
        print("empty — run `python -m inforce.cli m1` first")
        return 0
    header = f"{'batch':<28}{'rows':>8}{'linked':>8}{'circ#':>8}{'no date':>9}{'defer':>7}  {'range'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        rng = f"{r['earliest'] or '?'} .. {r['latest'] or '?'}"
        print(
            f"{r['batch']:<28}{r['rows']:>8,}{r['linked'] or 0:>8,}"
            f"{r['with_circ_no'] or 0:>8,}{r['unparsed_dates'] or 0:>9,}"
            f"{r['deferred'] or 0:>7,}  {rng}"
        )
    tot = conn.execute(
        "SELECT COUNT(*) c, SUM(rbi_doc_id IS NOT NULL) l FROM annex_entry"
    ).fetchone()
    print("-" * len(header))
    print(f"{'TOTAL':<28}{tot['c']:>8,}{tot['l'] or 0:>8,}")

    docs = store.document_summary(conn)
    if docs:
        print("\ndocuments fetched:")
        dh = f"{'kind':<18}{'status':<14}{'fetch':<8}{'n':>8}{'avg chars':>12}"
        print(dh)
        print("-" * len(dh))
        for r in docs:
            avg = f"{r['avg_chars']:,.0f}" if r["avg_chars"] else "-"
            print(
                f"{r['source_kind']:<18}{r['status']:<14}{r['fetch_status']:<8}"
                f"{r['n']:>8,}{avg:>12}"
            )
        pending = len(store.pending_notification_ids(conn))
        print(f"\nnotifications still pending: {pending:,}")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(prog="inforce")
    parser.add_argument("--db", type=str, default=None, help="override DB path")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("m1", help="scrape and label the RBI withdrawn-circulars annex")
    p1.add_argument("--refresh", action="store_true", help="re-fetch instead of using cache")
    p1.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    p1.set_defaults(func=cmd_m1)

    p3 = sub.add_parser("notifications", help="M2: fetch withdrawn circulars (HTML)")
    p3.add_argument("--limit", type=int, default=None)
    p3.add_argument("--delay", type=float, default=config.CRAWL_DELAY_SECONDS)
    p3.add_argument("--refresh", action="store_true")
    p3.set_defaults(func=cmd_notifications)

    p4 = sub.add_parser("master-directions", help="M2: index + PDFs of in-force MDs")
    p4.add_argument("--limit", type=int, default=None)
    p4.add_argument("--delay", type=float, default=config.CRAWL_DELAY_SECONDS)
    p4.add_argument("--refresh", action="store_true")
    p4.add_argument("--no-pdfs", action="store_true", help="index metadata only")
    p4.set_defaults(func=cmd_master_directions)

    p5 = sub.add_parser("index", help="M3: chunk + embed + build the vector index")
    p5.add_argument("--limit-docs", type=int, default=None)
    p5.add_argument("--limit-chunks", type=int, default=None)
    p5.add_argument("--no-embed", action="store_true", help="chunk only")
    p5.add_argument("--no-fts", action="store_true", help="skip the BM25 index")
    p5.add_argument("--rebuild-matrix", action="store_true",
                    help="rebuild the search matrix from already-embedded chunks")
    p5.set_defaults(func=cmd_index)

    p6 = sub.add_parser("ask", help="M3: naive retrieval (the control group)")
    p6.add_argument("question")
    p6.add_argument("-k", type=int, default=8)
    p6.add_argument("--no-llm", action="store_true", help="retrieval only, no generation")
    p6.add_argument("--allow-stale", action="store_true",
                    help="search even if the matrix has drifted from the database")
    p6.set_defaults(func=cmd_ask)

    p7 = sub.add_parser("candidates",
                        help="M4: find in-force/withdrawn pairs that cover the same ground")
    p7.add_argument("--top", type=int, default=25)
    p7.add_argument("--min-matches", type=int, default=3)
    p7.set_defaults(func=cmd_candidates)

    p8 = sub.add_parser("questions", help="M4: load and inspect the golden question set")
    p8.add_argument("--file", type=str, default=None, help="JSONL to load")
    p8.add_argument("--force", action="store_true", help="store despite validation problems")
    p8.add_argument("--prune", action="store_true",
                    help="delete questions in the database that are absent from the file")
    p8.set_defaults(func=cmd_questions)

    p9 = sub.add_parser("measure", help="M5: run the question set and compute N")
    p9.add_argument("-k", type=int, default=8)
    p9.add_argument("--mode", default="naive", choices=["naive", "time_aware"])
    p9.add_argument("--as-of", default="2026-08-08",
                    help="date for time_aware retrieval (YYYY-MM-DD)")
    # doc+llm is the reranker without query expansion. evaluate.py always
    # supported it -- it tests for "expand" and "llm" independently -- but the
    # choice list omitted it, so the one arm that isolates the reranker from an
    # expansion step that measurably hurt could not be run.
    p9.add_argument("--ranking", default="chunk",
                    choices=["chunk", "doc", "doc+llm", "doc+expand", "doc+expand+llm"],
                    help="chunk position, or document-level aggregation with optional "
                         "query expansion and listwise LLM reranking")
    p9.add_argument("--retrieval", default="dense", choices=["dense", "hybrid"],
                    help="dense top-k, or dense+BM25 fused by reciprocal rank")
    p9.add_argument("--entity", default="none", choices=["none", "oracle", "inferred"],
                    help="entity scoping: oracle uses the label, inferred reads the question")
    p9.add_argument("--label", default=None)
    p9.add_argument("--detail", action="store_true", help="per-question breakdown")
    p9.set_defaults(func=cmd_measure)

    p10 = sub.add_parser("temporal", help="M6: bi-temporal validity and snapshots")
    p10.add_argument("--build", action="store_true",
                     help="migrate, backfill valid time, infer supersession")
    p10.add_argument("--as-of", action="append", help="date(s) to snapshot, YYYY-MM-DD")
    p10.set_defaults(func=cmd_temporal)

    p12 = sub.add_parser("agent", help="M11: resolve -> retrieve -> trace loop -> adjudicate")
    p12.add_argument("question")
    p12.add_argument("--today", default="2026-08-08")
    p12.add_argument("--max-chains", type=int, default=3)
    p12.set_defaults(func=cmd_agent)

    p11 = sub.add_parser("serve", help="M8: run the side-by-side demo")
    p11.add_argument("--host", default="127.0.0.1")
    p11.add_argument("--port", type=int, default=8000)
    p11.set_defaults(func=cmd_serve)

    p2 = sub.add_parser("status", help="summarise what is in the database")
    p13 = sub.add_parser("abstention",
                         help="measure whether the generator refuses on unanswerable questions")
    p13.add_argument("-k", type=int, default=8)
    p13.add_argument("--as-of", default="2026-08-08")
    p13.add_argument("--retrieval", default="hybrid", choices=["dense", "hybrid"])
    p13.add_argument("--sample", type=int, default=60,
                     help="answerable questions to sample for the false-refusal rate")
    p13.add_argument("--seed", type=int, default=20260813)
    p13.add_argument("--timeout", type=int, default=300)
    p13.add_argument("--no-gate", action="store_true",
                     help="skip the pre-generation answerability check, to "
                          "measure what grounding verification alone catches")
    p13.add_argument("--save", type=str, default=None,
                     help="write raw answers to JSON so they can be reclassified "
                          "without re-running generation")
    p13.add_argument("--detail", action="store_true",
                     help="list unanswerable questions the system answered anyway")
    p13.set_defaults(func=cmd_abstention)

    p2.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    if args.db:
        import pathlib

        args.db = pathlib.Path(args.db)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
