# InForce — Build Plan

*Working name. "Was this rule in force on date X?" Alternatives: AsOf, Praman.*

---

## The thesis

Retrieval systems have no model of time. An embedding cannot distinguish "is true" from "was true" — a repealed circular and the Master Direction that replaced it are near-identical semantically, so they rank near-identically too.

In India this is unusually severe. On 28 November 2025, RBI withdrew **9,445 circulars** in one notification (RBI/2025-26/100), consolidating them into 244 Master Directions. A further 628 were repealed on 31 July 2026. All of those documents are still on rbi.org.in, still indexed, still perfectly retrievable. A pipeline that scrapes "all RBI circulars" builds a corpus where withdrawn documents outnumber live ones roughly forty to one.

And you cannot simply delete them. SEBI and RBI savings clauses state that repealed instructions still govern conduct from the period when they applied — any act done, penalty incurred, or proceeding pending under a rescinded circular remains enforceable. So:

- **Delete the repealed documents** → legally wrong. You cannot answer an audit, an inspection, or an enforcement question about past conduct.
- **Keep them without a time index** → factually wrong. You cite dead law as current.

The only correct design keeps both and separates them by time. That configuration does not exist off the shelf.

---

## The decision gate

This project is built **measurement first**. The single number that justifies everything downstream:

> **N** = how often a naive RAG pipeline over the RBI archive cites a withdrawn circular when asked about current requirements.

You get N at **M5**, roughly a third of the way in.

- **N > 20%** → the premise holds. Build the fix (M6 onward).
- **N in 5–20%** → real but modest. Continue, but lead the write-up with the audit/as-of-date case rather than the staleness case.
- **N < 5%** → the premise is wrong. **Stop.** You will have learned that in weeks rather than months, and that is the point of sequencing it this way.

Designing an experiment that can falsify your own idea is the strongest single signal in this project. Do not skip the gate, and do not quietly move the goalposts if the number disappoints.

---

## Ground rules

1. **Labels before documents.** M1 comes before M2. If the ground truth doesn't parse, nothing else matters.
2. **The naive baseline is a permanent feature, not a phase.** Keep it behind a `--naive` flag forever. It is your control group, and you will re-run it every time you change anything.
3. **Resist improving the baseline.** Its badness is the finding. A good baseline destroys the result you are trying to produce.
4. **Cache every fetch to local disk, permanently.** Never let a demo or an eval run depend on a live request to rbi.org.in.
5. **Scope v1 to the 3,516 annex rows that carry stable document IDs.** Skip fuzzy matching on the first pass — it is not on the critical path to N. Note that M1 found **93.4% of rows also carry a departmental circular number** (`DOR.CRE.REC.402/07-01-001/2025-26`), so the eventual matching job for the remaining rows is far smaller than originally feared.

---

## Milestones

Effort is in **focused days**. Convert to your own calendar: at ~12 hrs/week a focused day is roughly half a week; at full time it is a day.

### M0 — Setup — *0.5 days*

**Learn:** Python virtualenvs, `.env` handling, Docker basics for a single service.

**Build:** Python 3.13 project skeleton. Postgres 16 + pgvector running via `docker compose`. Ollama installed with a model pulled. One script that connects to the DB and prints a row.

**You now have:** a skeleton that runs.

---

### M1 — The labels — *1–2 days*  ← **do this first**

**Learn:** `requests` with a real User-Agent; parsing large HTML tables with `lxml` (not the default parser — the page is 3.6 MB).

**Build:** Scrape `https://www.rbi.org.in/scripts/NotificationUserWithdrawnCircular.aspx`. One GET, HTTP 200, static HTML, no JS, no auth. ~10,807 `<tr>` rows, columns `Date | Subject | Department`. Extract `NotificationUser.aspx?Id=NNNN` where present.

Write to a `documents` table: `rbi_doc_id`, `circular_date`, `subject`, `department`, `status='withdrawn'`, `has_stable_id`.

**Verify:** roughly 3,339 rows should carry an ID. If that count is wildly off, RBI changed the page — re-check before continuing.

**You now have:** ~3,339 regulator-labelled withdrawn documents. **Zero annotation cost.** This is the asset that makes the whole project cheap.

---

### M2 — The documents — *2–3 days*

**Learn:** polite scraping (rate limiting, retry with backoff, permanent disk cache); PDF text extraction with PyMuPDF and why naive extraction mangles tables.

**Build:** Fetch the 3,516 ID-linked circulars, then `BS_ViewMasDirections.aspx` for the in-force Master Directions, marked `status='in_force'`.

Cache every response to disk keyed by URL. Make re-runs idempotent — you will run this many times.

**What the build actually found (corrections to the assumptions above):**

- **Circular text is inline in the notification HTML**, inside `#pnlDetails`. No PDF parsing needed for most of the withdrawn corpus — this removes the whole table-extraction problem the original plan budgeted for.
- **But roughly a quarter of pages are title-plus-PDF only.** Master-Direction-style documents (PDF filenames like `271MD….PDF` rather than `NT21….PDF`) carry no inline body. These need a PDF fallback, and they matter disproportionately — they're the documents most easily confused with live Master Directions.
- **Notification pages carry two reference numbers**, `RBI/2025-26/211` and `DOR.CRE.REC.402/07-01-001/2025-26`. Extra join keys, free.
- **Every page links two boilerplate PDFs** (an Utkarsh brochure and an accessibility notice) under `/rdocs/content/pdfs/`. Matching on `.pdf` alone grabs the wrong file — filter to `/rdocs/notification/PDFs/`.
- **The MD index holds 410 documents, not 244.** It spans all departments and years; the November 2025 consolidation is a subset, identifiable by its `Nov 28, 2025` date row. ~240 MB of PDFs, median 487 kb, max 2.8 MB.
- **The MD index is a flat, stateful table**: a category heading row, then a date row, then the documents belonging to both. Category and date must be carried down while walking, and a new category must reset the date — otherwise documents get silently mis-dated.

**You now have:** a real, labelled, two-class corpus. No AI yet. Do not shortcut this step; silent data problems poison everything downstream and surface as mysterious model behaviour weeks later.

---

### M3 — The naive baseline — *3–4 days*

**Learn:** what an embedding is, cosine similarity, pgvector column types and HNSW indexing, fixed-size chunking.

**Build:** The dumbest defensible RAG. Fixed-size chunks with overlap → embed with `bge-small-en-v1.5` locally → store in pgvector → top-k cosine retrieval → stuff into a local LLM prompt → answer with sources.

No reranking. No hybrid search. No date filtering. No cleverness.

**You now have:** the control group. Put it behind `--naive` and never delete it.

---

### M4 — The question set — *4–6 days*  ← **the grind**

**Learn:** how to write eval questions that are answerable from the corpus, how to avoid leakage (don't quote the source text in the question), why unanswerable questions belong in the set.

**Build:** 50–60 questions about current regulatory requirements, chosen so that **both** a withdrawn circular and a current Master Direction plausibly match. Use the `Department` column to stratify — pick topics where consolidation demonstrably happened.

For each: the question, the expected current source, and the withdrawn document(s) that would be the wrong answer. Add ~10 deliberately unanswerable questions to measure abstention later.

**You now have:** your eval set — the most valuable artifact in the repo, and the one nobody else will bother to build.

This milestone is unglamorous and it is where the project is most likely to die. Budget honestly and do not rush it.

---

### M5 — Measure N — *1–2 days*  ← **THE PAYLOAD**

**Learn:** nothing new. This is counting.

**Build:** Run M4's questions through M3's baseline. For each answer compute:

- % of answers citing **at least one** withdrawn document
- % where the **top-1** retrieved source is withdrawn
- % of total retrieved chunks that come from withdrawn documents

**You now have: N.** Push the repo public with this number in the README even if nothing else is built. From here on the project has a defensible claim.

**→ Apply the decision gate before continuing.**

---

### M6 — Bi-temporal store — *3–4 days*

**Learn:** valid-time vs transaction-time; this is Slowly Changing Dimension Type 2 from data warehousing, wearing a new hat. The concept sounds heavier than the code.

**Build:** Add `valid_from` / `valid_to` to documents. Withdrawal dates already come from M1 — the Annex is three separate tables, one per batch, each with its own effective date, plus 16 `@`-marked rows deferred to 2026-01-01. Add a `supersession` edge table (`withdrawn_doc → replacing_master_direction`) where derivable.

**~~Wayback Machine CDX diffing~~ — no longer required.** The plan originally assumed the Annex was one cumulative list with no per-row withdrawal date, requiring snapshot diffs to recover batch membership. M1 disproved that. Saves roughly a day.

**Known limitation to document honestly:** withdrawal dates are batch-level, not document-level. Within a batch, every row shares one effective date.

**You now have:** the data model that makes as-of-date queries possible.

---

### M7 — As-of-date retrieval + the delta — *2–3 days*

**Build:** Retrieval filters by an as-of date, defaulting to today. Re-run M4's question set. Compute **M** — the same metrics, with time-awareness on.

**You now have: N → M.** The before/after. This single line is the entire pitch.

---

### M8 — The demo — *3–4 days*

**Learn:** SSE streaming from FastAPI (skip if it fights you — a non-streaming response is fine for a demo).

**Build:** One page. A question box, a date picker, and a toggle. Same question rendered side by side: time-blind answer citing a repealed circular, time-aware answer citing the live Master Direction. Citations link to the actual RBI document.

**You now have:** the 30-second hook. This is what a founder or recruiter sees before they read anything.

---

### M9 — Real model + quality pass — *2 days*

**Build:** Swap the local model for `claude-sonnet-5` on the generation step. Re-run the full eval. Publish both sets of numbers — the local-model results are a genuine finding in their own right ("here is what this costs to run on a laptop").

Add a cross-encoder reranker (`bge-reranker-base`) here if retrieval quality is the bottleneck. Measure before and after; if it doesn't move the number, drop it and say so.

**You now have:** shipping-quality numbers.

---

### M10 — Package — *2–3 days* — **done**

**Built:** README leading with the results table. MIT licence with an explicit note that RBI documents are public records, not redistributed, and that this is not regulatory advice. CONTRIBUTING.md. GitHub Actions across Python 3.11–3.13. Honest "Known limitations" section.

**Two corrections to this milestone as originally written:**

**CI cannot run the eval suite.** The plan assumed it would. It can't: the eval needs a 254 MB corpus and a local Ollama model, neither of which belongs in CI. What CI *does* run is the 91-test suite plus three checks that a working checkout cannot catch — SQLite FTS5 availability, every module importing, and the demo page actually shipping as package data. Claiming "CI runs my evals" when it runs unit tests would be the kind of overstatement this project criticises elsewhere.

**Docker was dropped, on purpose.** `docker compose up` would not work from a clean clone: the pipeline needs Ollama as a second service with GPU passthrough, the corpus is not in git so a fresh container has nothing to retrieve over, and Docker is not installed on the development machine so anything written here would be untested. Shipping an untested compose file that *looks* like it works is precisely the README-ware pattern the competitive analysis identified. The reproducibility story is `pip install -e .` plus the CLI — verified by building the wheel and inspecting it.

**Building the wheel found real bugs:** `numpy`, `pymupdf`, `fastapi` and `uvicorn` were imported but undeclared, and `static/index.html` was not packaged — so an installed copy's demo would have served a 404 while the repo worked fine.

**You now have:** the shippable repo. **This is the stop line.**

---

### M11 — The agent layer — *5–7 days*  ← **required, not optional**

**Why this is not optional:** hiring research is unambiguous that agents get a dedicated interview round, and that candidates who have shipped orchestration price above those who have shipped only RAG. "I deliberately chose not to use an agent framework" is a senior answer — but it only lands if you can demonstrate that you could have.

**Why it isn't decorative here.** The supersession trace is genuinely agentic:

> *"What are the current KYC requirements for NBFCs?"* → retrieval hits circular X → X withdrawn Nov 2025 → subsumed into Master Direction Y → Y amended by a circular in July 2026 → is that one still live?

Iterative graph walk, a decision at each hop, **variable depth, explicit termination condition**. A single LLM call structurally cannot do this.

**Learn:** LangGraph state, nodes, conditional edges, and cycles specifically. Anthropic's "Building Effective Agents" *before* you design, so you don't overbuild.

**Build:** three nodes, each with a genuinely distinct job:

1. **Temporal resolver** — infer the as-of date from the question ("when we onboarded this customer in 2021"). Distinct extraction task.
2. **Supersession tracer** — the iterative loop above, with a hop limit and a termination condition.
3. **Conflict adjudicator** — when two documents are both valid at date D with overlapping scope, decide which governs. A different reasoning mode from retrieval.

**Measure:** re-run M4's question set with the agent layer on. Report the delta, the added latency, and the added token cost. If it doesn't improve the number, **say so** — a measured negative result is a stronger signal than an unmeasured feature.

**You now have:** a defensible answer to "why multi-agent instead of one well-prompted call" that comes from the problem rather than from a job description.

---

### M12 — Prompt injection threat model — *1 day* — **done**

Named explicitly in hiring research as a screening signal: *"anyone who hasn't thought about prompt injection hasn't shipped a real system."* Your corpus is third-party documents, so the threat is real rather than theoretical.

**Build:** a short README section on the threat model, plus a handful of tests that plant injection strings in ingested documents and confirm they don't alter routing or citations. One day, high signal.

---

## Stop here

That's roughly **28–38 focused days** through M12. The hiring-signal curve flattens hard after this — building further costs months and returns very little.

**Optional extensions, strictly in this order, only if time remains:**

1. **Abstention measurement** — how often does it correctly refuse the unanswerable questions from M4?
2. **SEBI as a second corpus** — Master Circular rescission annexures work the same way. Proves the approach generalises across regulators.
3. **Fuzzy matching for the remaining ~69%** of annex rows without stable IDs.
4. **Clause-level lineage** — "what specifically changed between the old circular and the new Master Direction." **Genuinely hard.** RBI published *which* circulars were withdrawn, not *where each one went*. This can eat three months. Do not start it before M12 is done.

---

## Open risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| N comes back low | Medium | That's the gate working. You lose weeks, not months. |
| M4 question set is rushed and weak | **High** | The most likely failure. It's the boring middle. Slow down here. |
| Motivation dies in M1–M2 (scraping, no AI) | High | Keep M5 visible — the payload is close. |
| RBI changes the annex page | Low | Permanent disk cache from M2 means a change can't break you retroactively. |
| Scope creep into clause-level lineage | Medium | It is item 5 on the optional list for a reason. |
