"""M11 — agent tests.

The loop is the only part that is genuinely an agent, so it is the part worth
pinning: variable depth, and three distinct exits (reached a live document, ran
out of candidates, hit the hop cap).
"""
from __future__ import annotations

import pytest

from inforce import store, temporal
from inforce.agent import MAX_HOPS, resolve_date

TODAY = "2026-08-08"


@pytest.mark.parametrize("q,expected", [
    ("What applied on 2022-03-14?", "2022-03-14"),
    ("What were the rules in March 2022?", "2022-03-28"),
    ("What did banks have to do in 2019?", "2019-12-31"),
])
def test_resolve_date_reads_the_question(q, expected):
    assert resolve_date(q, TODAY)[0] == expected


def test_explicit_date_beats_a_bare_year():
    """'as of 2022-03-14 under the 2016 Direction' must resolve to the date."""
    assert resolve_date("as of 2022-03-14 under the 2016 Direction", TODAY)[0] == "2022-03-14"


def test_no_date_defaults_to_today():
    as_of, how = resolve_date("What capital must a bank hold?", TODAY)
    assert as_of == TODAY
    assert "defaulted" in how


def test_current_year_is_not_treated_as_historical():
    """A question mentioning this year means now, not 31 December."""
    assert resolve_date("what changed in 2026?", TODAY)[0] == TODAY


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "a.db")
    store.init_schema(c)
    temporal.migrate(c)
    docs = [
        ("notif:old", "withdrawn", None),
        ("notif:mid", "withdrawn", None),
        ("md:sfb", "in_force", "Small Finance Banks"),
        ("md:ucb", "in_force", "Urban Co-operative Banks"),
        ("notif:loopA", "withdrawn", None),
        ("notif:loopB", "withdrawn", None),
    ]
    for key, status, cat in docs:
        c.execute(
            """INSERT INTO document (doc_key, source_kind, status, url, category,
                                     fetch_status, fetched_at)
               VALUES (?,?,?,'http://x',?,'ok','t')""",
            (key, "notification", status, cat))
    edges = [
        ("notif:old", "notif:mid", 1),
        ("notif:mid", "md:ucb", 1),     # most similar, wrong entity
        ("notif:mid", "md:sfb", 2),     # correct entity, ranked lower
        ("notif:loopA", "notif:loopB", 1),
        ("notif:loopB", "notif:loopA", 1),   # a cycle
    ]
    c.execute("ALTER TABLE supersession ADD COLUMN rank INTEGER NOT NULL DEFAULT 1")
    for a, b, rank in edges:
        c.execute(
            """INSERT INTO supersession (withdrawn_doc_key, replacement_doc_key,
                                         rank, method, confidence, created_at)
               VALUES (?,?,?, 'inferred_centroid', 0.9, 't')""", (a, b, rank))
    c.commit()
    return c


def test_chain_walks_multiple_hops_and_stops_at_a_live_document(conn):
    maps = temporal.load_chain_maps(conn)
    chain = temporal.chain_from(conn, "notif:old", maps=maps)
    assert chain[0] == "notif:old"
    assert len(chain) > 2, "must traverse the intermediate document"
    assert chain[-1].startswith("md:"), "must terminate on something in force"


def test_cycle_does_not_hang(conn):
    """A -> B -> A. Without cycle detection this loops until the hop cap, and
    with a bad cap it never terminates at all."""
    maps = temporal.load_chain_maps(conn)
    chain = temporal.chain_from(conn, "notif:loopA", maps=maps)
    assert chain == ["notif:loopA", "notif:loopB"]
    assert len(chain) <= MAX_HOPS + 1


def test_hop_limit_is_respected(conn):
    maps = temporal.load_chain_maps(conn)
    chain = temporal.chain_from(conn, "notif:old", max_hops=1, maps=maps)
    assert len(chain) == 2, "one hop means two nodes"


def test_tracer_prefers_the_matching_entity_over_the_closest_match(conn):
    """`notif:mid` has two candidates: Urban Co-operative Banks ranked first by
    similarity, Small Finance Banks second. A Small Finance Banks question must
    follow the second — otherwise the wrong-entity failure reappears inside the
    supersession graph, which is exactly what happened before this fix.
    """
    from inforce.agent import choose_next

    edges = {"notif:mid": ["md:ucb", "md:sfb"]}
    cats = {"md:ucb": "Urban Co-operative Banks", "md:sfb": "Small Finance Banks"}

    assert choose_next("notif:mid", ["notif:mid"], "Small Finance Banks",
                       edges, cats) == "md:sfb"
    assert choose_next("notif:mid", ["notif:mid"], None,
                       edges, cats) == "md:ucb", "with no entity, rank order wins"
    assert choose_next("notif:mid", ["notif:mid"], "Payments Banks",
                       edges, cats) == "md:ucb", "no match falls back to rank order"


def test_chooser_never_revisits_a_node():
    """Cycle protection lives in the chooser, not just the walker."""
    from inforce.agent import choose_next

    edges = {"a": ["b", "c"]}
    assert choose_next("a", ["a", "b"], None, edges, {}) == "c"
    assert choose_next("a", ["a", "b", "c"], None, edges, {}) is None


def test_chooser_prefers_a_live_candidate():
    """A chain exists to answer 'what governs now', so a live candidate ends it.
    Without this the walker wandered on and 3,003 chains hit the hop cap."""
    from inforce.agent import choose_next

    edges = {"a": ["dead1", "live1"]}
    status = {"dead1": "withdrawn", "live1": "in_force"}
    titles = {"dead1": "Something Else", "live1": "A Live Direction"}
    assert choose_next("a", ["a"], None, edges, {}, status, titles) == "live1"


def test_chooser_prefers_a_different_instrument_over_another_revision():
    """Seven documents all called 'Change in Bank Rate' are one instrument being
    revised. Given a genuine alternative, take it."""
    from inforce.agent import choose_next

    edges = {"a": ["dup", "other"]}
    titles = {"a": "Change in Bank Rate", "dup": "Change in  Bank Rate!",
              "other": "Liquidity Adjustment Facility"}
    status = {"dup": "withdrawn", "other": "withdrawn"}
    assert choose_next("a", ["a"], None, edges, {}, status, titles) == "other"


def test_chooser_still_walks_a_duplicate_when_it_is_the_only_option():
    """Refusing cost 15 points of chain resolution while only tidying output.
    Repeat runs are collapsed at display time instead."""
    from inforce.agent import choose_next

    edges = {"a": ["dup"]}
    titles = {"a": "Change in Bank Rate", "dup": "Change in Bank Rate"}
    assert choose_next("a", ["a"], None, edges, {}, {"dup": "withdrawn"}, titles) == "dup"


def test_missing_titles_never_block_traversal():
    """A blank title is absence of evidence, not evidence of duplication."""
    from inforce.agent import choose_next

    edges = {"a": ["b"]}
    assert choose_next("a", ["a"], None, edges, {}, {}, {}) == "b"


def test_updated_as_on_suffix_does_not_make_a_new_instrument():
    from inforce.agent import _norm_title

    assert _norm_title("KYC Directions, 2025 (Updated as on June 24, 2026)") == \
           _norm_title("KYC Directions, 2025")


def test_agent_runs_end_to_end_on_a_fixture(conn, monkeypatch):
    """The whole graph, with retrieval stubbed — this pins the wiring
    (entry point, conditional edge, termination) rather than retrieval quality."""
    from inforce import agent, retrieve

    class FakeHit:
        def __init__(self, key):
            self.doc_key, self.rank, self.title = key, 1, key
        def validity_on(self, as_of):
            return temporal.WITHDRAWN if self.doc_key.startswith("notif") else temporal.IN_FORCE

    monkeypatch.setattr(retrieve, "search",
                        lambda conn, q, **kw: [FakeHit("md:sfb"), FakeHit("notif:old")])
    state = agent.run(conn, "small finance bank KYC rules", today=TODAY)

    assert state["entity"] == "Small Finance Banks"
    assert state["as_of"] == TODAY
    assert state["hops"] >= 1, "the loop must have executed at least once"
    assert any(len(c) > 1 for c in state["chains"]), "a chain must have advanced"
    assert any("traced" in n for n in state["notes"])
