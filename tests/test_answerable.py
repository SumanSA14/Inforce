"""The answerability gate.

Covers the gap grounding verification cannot see. verify.py checks that every
figure in an answer appears in a retrieved source, which stops the model
inventing numbers. It cannot catch a number that is genuinely present, quoted
correctly, and attached to the wrong obligation -- the failure that dominated
on unanswerable questions.
"""
from __future__ import annotations

from unittest import mock

import pytest

from inforce import answerable, generate


class FakeHit:
    def __init__(self, rank, text, title="RBI (X - Y) Directions", doc_key="md:1"):
        self.rank, self.text, self.title, self.doc_key = rank, text, title, doc_key


HITS = [FakeHit(1, "A regulated entity shall not contribute more than 10 percent "
                   "of the corpus of an AIF scheme.")]


def _reply(content):
    m = mock.Mock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"message": {"content": content}}
    return m


def test_no_evidence_is_not_answerable():
    """No retrieval means nothing to ground an answer in; that must not reach
    the generator at all."""
    ok, why = answerable.can_answer("anything", [])
    assert ok is False
    assert "no evidence" in why


@pytest.mark.parametrize("verdict,expected", [
    ("NO\nNONE", False),
    ("no\nNONE", False),
    ("YES\nA regulated entity shall not contribute more than 10 percent.", True),
    ("yes\nsome quote", True),
])
def test_verdict_is_parsed(verdict, expected):
    with mock.patch.object(answerable.SESSION, "post", return_value=_reply(verdict)):
        ok, _ = answerable.can_answer("q", HITS)
    assert ok is expected


def test_transport_failure_fails_open():
    """A refusal caused by an outage would masquerade as excellent abstention.
    The gate must never turn a transport error into a decline."""
    with mock.patch.object(answerable.SESSION, "post",
                           side_effect=RuntimeError("connection reset")):
        ok, why = answerable.can_answer("q", HITS)
    assert ok is True
    assert "unavailable" in why


def test_unparsed_verdict_fails_open():
    with mock.patch.object(answerable.SESSION, "post",
                           return_value=_reply("I think maybe possibly")):
        ok, why = answerable.can_answer("q", HITS)
    assert ok is True
    assert "unparsed" in why


def test_gate_is_off_by_default():
    """It prevented 29 fabrications and destroyed 33 correct answers when
    measured; it refuses 98% of answerable questions. Shipping it on would
    make the system strictly worse, so the default must stay off until the
    judge discriminates."""
    import inspect
    assert inspect.signature(generate.answer).parameters["gate"].default is False


def test_gate_declines_before_generating():
    """When the gate says no, no generation call is made at all -- the point is
    to decline before an answer exists to defend."""
    with mock.patch.object(answerable, "can_answer", return_value=(False, "nope")), \
         mock.patch.object(generate.SESSION, "post") as post:
        text, verdict = generate.answer("q", HITS, gate=True, strict=False)
    post.assert_not_called()
    assert "can't answer" in text
    assert verdict.supported is True


def test_gate_can_be_disabled_for_measurement():
    """The two defences must be separable or their contributions cannot be
    told apart."""
    with mock.patch.object(answerable, "can_answer") as gate, \
         mock.patch.object(generate.SESSION, "post",
                           return_value=_reply("An answer. [1]")):
        generate.answer("q", HITS, gate=False, strict=False)
    gate.assert_not_called()


def test_default_path_does_not_call_the_gate():
    """Belt and braces: the shipped default must not pay for a check that
    makes results worse."""
    with mock.patch.object(answerable, "can_answer") as gate,          mock.patch.object(generate.SESSION, "post",
                           return_value=_reply("An answer. [1]")):
        generate.answer("q", HITS, strict=False)
    gate.assert_not_called()
