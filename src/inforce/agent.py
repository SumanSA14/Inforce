"""M11 — the agent layer.

Three nodes, only one of which is genuinely an agent in the sense that matters
(the model directing its own control flow rather than following a fixed path):

  resolve      infer the as-of date and entity from the question. One pass,
               deterministic. A workflow step, not an agent.
  retrieve     time-aware + entity-scoped + hybrid retrieval. One pass.
  trace        **the loop.** Walk the supersession chain from each stale
               document until it reaches something in force, revisits a node,
               or hits the hop limit. Depth is data-dependent: 82.4% of chains
               in this corpus need more than one hop, and they run to six.
  adjudicate   when several live documents cover the same ground, decide which
               governs. A different reasoning mode, fixed path.

Being able to say "one of these is a real agent and the other two are workflow
nodes" is the point. Anthropic's "Building Effective Agents" draws exactly this
line, and claiming four agents where there is one loop is the decorative-agent
pattern this project criticises elsewhere.

LangGraph earns its place here only because of `trace`: a cycle with a
data-dependent exit is what conditional edges are for. Had the chains turned out
to be uniformly one hop — as the first, cruder supersession inference made them
appear — a plain function call would have been the honest choice, and this
module would not exist.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Annotated, TypedDict

from . import entities, retrieve, temporal  # noqa: F401  (temporal used in choose_next)

MAX_HOPS = 6

# Dates a practitioner would actually write.
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_YEAR = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(\d{4})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)(\d{2})\b")
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def _keep_last(a: list, b: list) -> list:
    return b


class AgentState(TypedDict, total=False):
    question: str
    today: str
    as_of: str
    entity: str | None
    hits: list
    stale: list          # documents a time-blind retriever would have surfaced
    frontier: list       # chain heads still being walked
    chains: Annotated[list, _keep_last]
    hops: int
    conflicts: list
    notes: list


def resolve_date(question: str, today: str) -> tuple[str, str]:
    """Return (as_of, how). Explicit dates win over months, months over years."""
    m = _ISO.search(question)
    if m:
        return m.group(0), "explicit date"
    m = _MONTH_YEAR.search(question)
    if m:
        return f"{m.group(2)}-{_MONTHS[m.group(1).lower()]:02d}-28", "month and year"
    m = _YEAR.search(question)
    if m:
        year = m.group(0)
        if year != today[:4]:
            # A bare past year means "as it stood then" — take year end.
            return f"{year}-12-31", "bare year"
    return today, "defaulted to today"


def _norm_title(t: str | None) -> str:
    """Titles differing only in punctuation, case or an '(Updated as on …)'
    suffix are the same instrument revised, not a supersession to something new."""
    t = re.sub(r"\(updated as on[^)]*\)", "", (t or ""), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def choose_next(
    head: str,
    chain: list[str],
    entity: str | None,
    edges: dict[str, list[str]],
    doc_cat: dict[str, str | None],
    status: dict[str, str] | None = None,
    titles: dict[str, str | None] | None = None,
) -> str | None:
    """Pick the next hop from the candidate set.

    This is where the loop makes a decision rather than following a fixed edge.
    Candidates are scored on three things, in order:

    1. **Already in force.** A chain exists to answer "what governs now", so a
       live candidate ends it. Without this the walker wandered through further
       repealed documents and 3,003 chains hit the hop cap without resolving.
    2. **Matching entity.** A Small Finance Banks question must not trace into
       Urban Co-operative Banks — the wrong-entity failure reappearing inside
       the supersession graph.
    3. **Similarity rank**, as the tie-break.

    Candidates whose title is a normalised duplicate of something already in the
    chain are skipped: a run of seven documents all called "Change in Bank Rate"
    is one instrument being revised, and walking it burns the hop budget without
    moving toward an answer.

    Pure function, so it is testable without LangGraph's internals.
    """
    status = status or {}
    titles = titles or {}
    seen_titles = {t for t in (_norm_title(titles.get(k)) for k in chain) if t}

    options = [c for c in edges.get(head, []) if c not in chain]
    if not options:
        return None

    # Prefer a candidate that is not simply another revision of something
    # already in the chain — but do NOT refuse to walk one when that is all
    # there is. Refusing cost 15 points of chain resolution (86.3% -> 71.5%)
    # while only tidying the output; runs of near-identical revisions are
    # collapsed at display time instead, which gets both.
    # A blank title carries no evidence of duplication, so it never blocks.
    fresh = [c for c in options
             if not _norm_title(titles.get(c))
             or _norm_title(titles.get(c)) not in seen_titles]
    pool = fresh or options

    def rank(item: tuple[int, str]) -> tuple[int, int, int]:
        idx, c = item
        live = status.get(c) == temporal.IN_FORCE
        match = bool(entity) and doc_cat.get(c) == entity
        return (0 if live else 1, 0 if match else 1, idx)

    return min(enumerate(pool), key=rank)[1]


def build(conn: sqlite3.Connection, *, today: str = "2026-08-08"):
    """Compile the graph. Returns an object with .invoke(state)."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - install-time guidance
        raise ImportError(
            "the agent layer needs LangGraph, which is an optional extra:\n"
            "    pip install -e \".[agent]\""
        ) from exc

    # Several candidate replacements per document, best first. The tracer picks
    # among them using the question's entity rather than taking whatever was
    # most similar at inference time.
    edges: dict[str, list[str]] = {}
    for r in conn.execute(
        "SELECT withdrawn_doc_key, replacement_doc_key FROM supersession "
        "WHERE method='inferred_centroid' ORDER BY withdrawn_doc_key, rank"
    ):
        edges.setdefault(r["withdrawn_doc_key"], []).append(r["replacement_doc_key"])

    status = {r["doc_key"]: r["status"]
              for r in conn.execute("SELECT doc_key, status FROM document")}
    titles = {r["doc_key"]: r["title"]
              for r in conn.execute("SELECT doc_key, title FROM document")}
    doc_cat = {r["doc_key"]: r["category"]
               for r in conn.execute("SELECT doc_key, category FROM document")}

    def node_resolve(state: AgentState) -> AgentState:
        as_of, how = resolve_date(state["question"], today)
        entity = entities.detect(state["question"])
        return {"as_of": as_of, "entity": entity, "hops": 0, "chains": [],
                "notes": [f"as-of {as_of} ({how})",
                          f"entity {entity or 'not specified'}"]}

    def node_retrieve(state: AgentState) -> AgentState:
        hits = retrieve.search(conn, state["question"], k=6,
                               as_of=state["as_of"], entity=state["entity"],
                               retrieval="hybrid")
        # What a time-blind retriever would have surfaced instead: these are the
        # chain heads worth tracing, because they are what the user might have
        # been handed by a system with no model of time.
        blind = retrieve.search(conn, state["question"], k=6, retrieval="hybrid")
        stale = [h.doc_key for h in blind
                 if h.validity_on(state["as_of"]) != temporal.IN_FORCE]
        seen, frontier = set(), []
        for key in stale:
            if key not in seen:
                seen.add(key)
                frontier.append(key)
        return {"hits": hits, "stale": frontier, "frontier": list(frontier),
                "chains": [[k] for k in frontier],
                "notes": state["notes"] + [
                    f"{len(hits)} in-force results; {len(frontier)} stale documents "
                    "a time-blind system would have returned"]}

    def node_trace(state: AgentState) -> AgentState:
        """One hop for every chain still open. The loop body."""
        chains, frontier = [], []
        for chain in state["chains"]:
            head = chain[-1]
            if status.get(head) == temporal.IN_FORCE:
                chains.append(chain)
                continue
            nxt = choose_next(head, chain, state.get("entity"), edges, doc_cat,
                              status, titles)
            if nxt is None:                      # dead end, or only cycles left
                chains.append(chain)
                continue
            chains.append(chain + [nxt])
            frontier.append(nxt)
        return {"chains": chains, "frontier": frontier,
                "hops": state["hops"] + 1}

    def route_after_trace(state: AgentState) -> str:
        """The conditional edge. Depth is data-dependent, so this is the part a
        straight-line pipeline could not express."""
        if not state["frontier"]:
            return "adjudicate"
        if state["hops"] >= MAX_HOPS:
            return "adjudicate"
        return "trace"

    def node_adjudicate(state: AgentState) -> AgentState:
        """Where several chains land on different live documents, the answer is
        ambiguous and should say so rather than pick silently."""
        landed, unresolved = {}, []
        for chain in state["chains"]:
            end = chain[-1]
            if status.get(end) == temporal.IN_FORCE:
                landed.setdefault(end, []).append(chain[0])
            else:
                unresolved.append(chain[0])

        notes = list(state["notes"])
        notes.append(f"traced {len(state['chains'])} chains in {state['hops']} hops")
        if unresolved:
            notes.append(f"{len(unresolved)} chain(s) did not reach a live document")
        conflicts = []
        if len(landed) > 1:
            conflicts = sorted(landed)
            notes.append(
                f"{len(landed)} different live documents were reached — "
                "the replacement is ambiguous, flagging rather than choosing")
        return {"conflicts": conflicts, "notes": notes,
                "chains": state["chains"]}

    g = StateGraph(AgentState)
    g.add_node("resolve", node_resolve)
    g.add_node("retrieve", node_retrieve)
    g.add_node("trace", node_trace)
    g.add_node("adjudicate", node_adjudicate)
    g.set_entry_point("resolve")
    g.add_edge("resolve", "retrieve")
    g.add_edge("retrieve", "trace")
    g.add_conditional_edges("trace", route_after_trace,
                            {"trace": "trace", "adjudicate": "adjudicate"})
    g.add_edge("adjudicate", END)

    compiled = g.compile()
    compiled._titles = titles     # for rendering
    compiled._status = status
    return compiled


def run(conn: sqlite3.Connection, question: str, *, today: str = "2026-08-08") -> dict:
    app = build(conn, today=today)
    return app.invoke({"question": question, "today": today, "notes": []})
