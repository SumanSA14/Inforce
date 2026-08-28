"""Refusal detection.

Deterministic phrase matching rather than an LLM judge, for the same reason N
has no judge: a grader drawn from the same model family that wrote the text
would fail invisibly. The cases below are real generated output, not invented
strings -- each was produced by asking the running system an out-of-scope
question.
"""
from __future__ import annotations

import pytest

from inforce.abstain import classify, refuses


# Verbatim output from the deployed generator.
OBSERVED_REFUSALS = [
    "The question about the capital of France is not addressed in the provided excerpts. [4]",
    "The provided excerpts do not contain any information regarding the cap on "
    "mutual fund total expense ratio. [1], [2], [3], [4], [5]",
    'The provided excerpts do not contain any information regarding a "purple '
    'elephant compliance sandwich". Therefore, no citation is needed as the '
    "information is not present in the given documents.",
    "According to [4], an NBFC-MFI shall adhere to the provisions of Reserve Bank "
    "of India (Non-Banking Financial Companies - Prudential Norms on Capital "
    "Adequacy) Directions, 2025. However, the exact net worth requirement for "
    "NBFC-MFI is not specified in the provided excerpts.",
]

ANSWERS = [
    "A payments bank shall maintain a minimum CRAR of 15 per cent. [2]",
    "The card issuer must seek OTP consent after 30 days of non-use. [1]",
    "Deposit insurance covers up to Rs 5 lakh per depositor. [3]",
]


@pytest.mark.parametrize("text", OBSERVED_REFUSALS)
def test_observed_refusals_are_detected(text):
    assert refuses(text) is True
    assert classify(text) == "refused"


@pytest.mark.parametrize("text", ANSWERS)
def test_substantive_answers_are_not_refusals(text):
    assert refuses(text) is False
    assert classify(text) == "answered"


def test_partial_refusal_is_not_credited_as_abstention():
    """A disclaimer followed by a claim answered something. Counting it as a
    clean refusal would credit the system for withholding what it supplied."""
    text = ("The excerpts do not state the fee, but they do require review "
            "within 30 days. [1]")
    assert refuses(text) is False
    assert classify(text) == "partial"


def test_single_sentence_partial_is_caught():
    """The per-sentence version of this check missed partials that put the
    disclaimer and the claim in one sentence across a contrastive conjunction."""
    assert refuses("No provision specifies this, though 15 per cent applies "
                   "generally.") is False


def test_withheld_output_counts_as_refusal():
    """verify.py suppresses an answer whose figures are absent from the
    sources; withholding is the behaviour being measured."""
    assert refuses("") is True
    assert refuses(None) is True
    assert refuses("   ") is True


def test_year_alone_is_not_a_substantive_claim():
    """A citation year must not be mistaken for a regulatory figure, or every
    refusal that names a Direction would be downgraded to a partial."""
    assert refuses("The Directions, 2025 do not specify this requirement.") is True


def test_domain_negation_is_not_a_refusal():
    """Regression: a correct, complete answer was scored as a refusal because
    "card not present" is standard payments terminology and matched the
    declining-language pattern. Refusal language must be tied to the SOURCE --
    regulatory prose is full of negations that are part of the rule."""
    text = ("[1] Card issuers shall put in place a mechanism to validate "
            "non-recurring, cross-border card not present (CNP) transactions "
            "by October 01, 2026, where request for authentication is raised "
            "by an overseas merchant.")
    assert refuses(text) is False
    assert classify(text) == "answered"


def test_source_word_plural_is_matched():
    """The commonest real phrasing is plural. An earlier pattern matched
    "excerpt" inside "excerpts" and then failed on the trailing s."""
    assert refuses("The provided excerpts do not contain that requirement.") is True


def test_declining_about_the_world_is_not_a_refusal():
    """A rule that says something is not permitted is an ANSWER, not a
    declination to answer."""
    text = "A payments bank shall not undertake lending activities. [2]"
    assert refuses(text) is False


def test_withheld_answer_is_a_refusal():
    """verify.py suppresses an answer whose figures are absent from the
    sources and substitutes a fixed message. That is a refusal by construction
    -- the output was withheld -- but it was scored as an ANSWER, because
    "do not support" was missing from the declining-language list and the
    suppressed draft's own figures could still match the substantive pattern.
    On unanswerable questions that inflated the apparent fabrication rate."""
    from inforce import verify
    msg = verify.REFUSAL.format(
        reason="figures not found in any retrieved source: 18 (percent)")
    assert refuses(msg) is True
    assert classify(msg) == "refused"


def test_withheld_detection_tracks_verify():
    """Imported, not retyped, so the two cannot drift apart."""
    from inforce.abstain import WITHHELD
    from inforce import verify
    assert verify.REFUSAL.startswith(WITHHELD)
