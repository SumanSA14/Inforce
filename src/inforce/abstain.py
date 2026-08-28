"""Did the generated answer refuse?

Abstention was previously measured only by thresholding the top retrieval
score. That measurement is a documented negative result: the answerable and
unanswerable score ranges overlap (separation -0.038), so no threshold
separates them, and under RRF every top-1 lands near 0.033 because fusion
scores encode position rather than similarity.

Meanwhile the mechanism that does work went unmeasured. The generator is
instructed to answer only from retrieved excerpts and to say so plainly when
they do not contain the answer, and it does: asked the capital of France, it
says the question is not addressed in the excerpts. This module detects that
refusal so the behaviour can be scored.

Deterministic phrase matching, not an LLM judge -- the same reasoning that
keeps N free of a grader model. A judge here would be scoring the output of the
same model family that produced the text, and its errors would be invisible.

Two failure modes shaped the design:

1. PARTIAL refusals. "The excerpts do not state the fee, but they do require
   annual review" answers something, just not what was asked. Counting it as a
   clean refusal would credit the system for withholding what it supplied.

2. DOMAIN PHRASES that look like refusals. A correct answer about "cross-border
   card not present (CNP) transactions" was scored as a refusal because
   "not present" matched. Refusal language must therefore be tied to the
   SOURCE -- the excerpts, the documents, the information provided -- and not
   float free anywhere in the text. Regulatory prose is full of negations that
   are part of the rule, not a declination to state it.
"""
from __future__ import annotations

import re

# Words naming the evidence. A refusal is about the SOURCE, not about the world.
# The trailing \w* absorbs plurals -- without it "excerpt" matches inside
# "excerpts" and the following \W then fails against the "s", so the commonest
# phrasing in real output ("the provided excerpts do not contain") was missed.
# "direction" and "circular" are here because in this corpus they name the
# source document: "the Directions do not specify" is a declination.
_SOURCE = (r"(?:excerpt|document|source|context|passage|text|information"
           r"|material|direction|circular|annex|provided|given|retrieved"
           r"|reference)\w*")

# Declining language. Each was observed in real output.
_DECLINE = (
    r"do(?:es)? not (?:contain|include|specify|state|mention|address|provide)"
    r"|not (?:addressed|specified|stated|mentioned|contained|explicitly mentioned)"
    r"|no (?:information|mention|reference)"
    r"|cannot (?:be )?(?:determined|answered|found)"
    r"|unable to (?:answer|determine|find)"
    r"|is not (?:covered|found|available|present|provided)"
    r"|do(?:es)? not support"
    r"|silent on"
    r"|outside the scope"
)

# The declining phrase must sit within ~80 characters of a word naming the
# evidence, in either order. That window covers "the provided excerpts do not
# contain" and "is not mentioned in the given documents" while rejecting
# "card not present", where no source word is anywhere near.
# A plain bounded character gap rather than a word-by-word one. The word-based
# version could not cross the space after a comma ("Directions, 2025 do not
# specify"), because it required word characters immediately after each
# separator.
REFUSAL = re.compile(
    rf"(?:(?:{_SOURCE}).{{0,70}}?(?:{_DECLINE}))"
    rf"|(?:(?:{_DECLINE}).{{0,70}}?(?:{_SOURCE}))",
    re.IGNORECASE | re.DOTALL,
)

# A concrete regulatory claim: a figure with a unit, or a money amount.
# A bare year is deliberately excluded -- "the Directions, 2025" is a citation,
# not a rule, and treating it as one would downgrade every refusal that names a
# document.
SUBSTANTIVE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:per cent|percent|%|bps|basis points))"
    r"|(?:₹|Rs\.?|INR)\s*[\d,]+"
    r"|\b\d+\s*(?:lakh|crore)\b"
    r"|\b\d+\s*(?:days?|months?|years?|hours?|working days?|weeks?)\b",
    re.IGNORECASE,
)


# The exact text verify.py substitutes when it suppresses an unsupported
# answer. Imported rather than retyped so the two cannot drift apart -- and
# checked explicitly, because it is a refusal by construction: the output was
# withheld. It was previously scored as an ANSWER, since "do not support" was
# missing from the declining-language list and the suppressed draft's figures
# could still match the substantive pattern. That inflated the apparent
# fabrication rate on unanswerable questions.
def _withheld_prefix() -> str:
    from .verify import REFUSAL as _R
    return _R.split("{")[0].strip()


WITHHELD = _withheld_prefix()


def refuses(answer: str | None) -> bool:
    """True when the answer declines for lack of evidence and asserts nothing.

    Whole-answer, not per-sentence: the per-sentence version missed partials
    that put the disclaimer and the claim in one sentence across a contrastive
    conjunction ("...do not state the fee, but they do require review within
    30 days").

    The bias is deliberately conservative -- a refusal that mentions a figure
    in passing is scored as a partial, so this UNDER-counts refusals rather
    than over-counting them. For a metric whose failure mode is flattering the
    system, erring toward under-credit is the safe direction.
    """
    if not answer or not answer.strip():
        # An empty answer is a withheld one -- verify.py suppresses output whose
        # figures are absent from the sources -- and withholding is the
        # behaviour being measured.
        return True
    if answer.strip().startswith(WITHHELD):
        return True
    return bool(REFUSAL.search(answer)) and not SUBSTANTIVE.search(answer)


def classify(answer: str | None) -> str:
    """refused | partial | answered -- for reporting rather than scoring."""
    if not answer or not answer.strip():
        return "refused"
    if answer.strip().startswith(WITHHELD):
        return "refused"
    has_refusal = bool(REFUSAL.search(answer))
    has_claim = bool(SUBSTANTIVE.search(answer))
    if has_refusal and not has_claim:
        return "refused"
    if has_refusal and has_claim:
        return "partial"
    return "answered"
