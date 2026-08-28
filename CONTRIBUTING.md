# Contributing

## Setup

```bash
pip install -e ".[dev]"
pytest
```

136 tests, all self-contained — no network, no corpus, no Ollama. If a test needs
any of those, it is written wrong.

## The rules that matter here

This project measures retrieval quality, so the code that produces the number
has to be harder to fool than the number itself.

**Validate by content, never by status code, length, or the absence of an
exception.** Every serious bug found while building this was a *silent success*:
an HTTP 200 serving a bot interstitial that was stored as 50 documents of
"content"; a stale index answering questions over 2% of the corpus; hybrid
retrieval degrading to dense and reporting the baseline's numbers as an
improvement. None raised. All were caught by checking output against something
independent — declared file size, database row count, corpus composition.

**Prefer failing loudly to degrading quietly.** `retrieve.search` raises when the
index has drifted or the BM25 table is missing rather than returning something
plausible. New code should do the same.

**Do not make the baseline artificially bad.** The naive pipeline is a control
group. It is allowed to be simple; it is not allowed to be broken, because a
crippled baseline inflates N and makes the headline result dishonest.

**Golden questions are written by hand.** An LLM-generated question set graded by
an LLM measures the grader. Automation is welcome for *finding* candidate topics
(`inforce candidates`) and for validation, not for authoring.

**Report the cost side of any metric.** Abstention is reported with its
false-refusal rate; entity detection with its miss rate; hybrid retrieval with
the MRR regression it causes on the unfiltered baseline. A metric quoted without
its tradeoff is marketing.

## Changing the question set

```bash
python -m inforce.cli questions --file questions/golden.jsonl --prune
```

Validation refuses on: an expected source that is withdrawn or absent, a quote
that does not appear verbatim in its document, question wording that overlaps
its own answer, and traps that are not actually withdrawn. `--prune` removes
questions deleted from the file, which would otherwise keep being scored.

Re-run `inforce measure` after any change. **N is expected to move**, and the
direction is informative — removing answer leakage raised it from 36.1% to
38.9%, and broadening coverage took it to 44.2%.

## Rebuilding from scratch

```bash
python -m inforce.cli m1                 # labels from RBI's published annex
python -m inforce.cli notifications      # ~3,235 withdrawn circulars
python -m inforce.cli master-directions  # 410 in-force Master Directions
python -m inforce.cli index              # chunk, embed, FTS
python -m inforce.cli temporal --build   # bi-temporal validity
```

Every stage is resumable and idempotent; every fetch is cached permanently.
Be polite to rbi.org.in — the default is one request per second, and the full
crawl takes a few hours. It only has to be paid once.

## Reporting a data problem

RBI changes its pages. If counts drift, say what you observed against what the
code expected — `inforce m1` prints both, and mismatches are warnings rather
than silent adjustments by design.
