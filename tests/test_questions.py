"""M4 validation tests.

The question set is the gate on N. A structurally broken question set produces
a number that looks fine and means nothing, so validation refuses rather than
warns.
"""
from __future__ import annotations

import sqlite3

import pytest

from inforce import store
from inforce.questions import Question, ValidationError, load_jsonl, validate


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    store.init_schema(c)
    now = "2026-08-08T00:00:00+00:00"
    for key, status, title in [
        ("md:live1", "in_force", "RBI (Commercial Banks - Capital Adequacy) Directions, 2025"),
        ("md:live2", "in_force", "RBI (SFB - Licensing) Guidelines, 2025"),
        ("notif:900", "withdrawn", "Operating Guidelines for Small Finance Banks"),
        ("notif:901", "withdrawn", "Capital Adequacy Norms 2016"),
    ]:
        c.execute(
            """INSERT INTO document (doc_key, source_kind, status, url, title,
                                     fetch_status, fetched_at)
               VALUES (?,?,?,?,?,'ok',?)""",
            (key, "master_direction" if status == "in_force" else "notification",
             status, "http://x", title, now),
        )
    c.commit()
    return c


def good(**over) -> Question:
    base = dict(
        qid="q1",
        text="What minimum capital adequacy ratio must small finance banks maintain?",
        expected_doc_key="md:live1",
        trap_doc_keys=["notif:901"],
    )
    base.update(over)
    return Question(**base)


def test_valid_question_passes(conn):
    assert validate(conn, [good()]) == []


def test_expected_document_must_be_in_force(conn):
    """The expected answer has to be current law. If the expected source is
    itself withdrawn, the question is scoring the wrong thing entirely."""
    problems = validate(conn, [good(expected_doc_key="notif:900")])
    assert any("not in force" in p for p in problems)


def test_expected_document_must_exist(conn):
    problems = validate(conn, [good(expected_doc_key="md:nope")])
    assert any("not in corpus" in p for p in problems)


def test_answerable_question_needs_an_expected_document(conn):
    problems = validate(conn, [good(expected_doc_key=None)])
    assert any("no expected_doc_key" in p for p in problems)


def test_question_without_traps_is_accepted(conn):
    """Traps are OBSERVED, not predicted — derived from what the baseline
    actually retrieves. A question with no trap is one the system handled
    correctly. Rejecting it would filter the set down to questions that already
    exhibit the failure, inflating N by construction. Successes must stay in the
    denominator.

    (An earlier version of this rule rejected such questions. Walking into that
    selection bias, and backing out of it, is why the rule is now inverted.)
    """
    assert validate(conn, [good(trap_doc_keys=[])]) == []


def test_trap_documents_must_actually_be_withdrawn(conn):
    problems = validate(conn, [good(trap_doc_keys=["md:live2"])])
    assert any("not withdrawn" in p for p in problems)


def test_unanswerable_question_must_not_claim_a_source(conn):
    q = Question(qid="q2", text="What is the capital ratio for llama farming co-ops?",
                 answerable=False, expected_doc_key="md:live1")
    problems = validate(conn, [q])
    assert any("marked unanswerable" in p for p in problems)


def test_unanswerable_question_without_source_is_fine(conn):
    q = Question(qid="q2", text="What is the capital ratio for llama farming co-ops?",
                 answerable=False)
    assert validate(conn, [q]) == []


def test_duplicate_qids_are_caught(conn):
    problems = validate(conn, [good(), good()])
    assert any("duplicate qid" in p for p in problems)


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '# a comment line\n'
        '{"qid":"q1","text":"What minimum capital adequacy ratio applies?",'
        '"expected_doc_key":"md:live1","trap_doc_keys":["notif:901"]}\n'
        '\n',
        encoding="utf-8",
    )
    qs = load_jsonl(path)
    assert len(qs) == 1
    assert qs[0].qid == "q1"
    assert qs[0].trap_doc_keys == ["notif:901"]
    assert qs[0].answerable is True


def test_malformed_jsonl_raises_with_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"qid": "q1", oops}\n', encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        load_jsonl(path)
    assert "line 1" in str(exc.value)
