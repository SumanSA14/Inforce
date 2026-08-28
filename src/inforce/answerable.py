"""Does the retrieved evidence actually answer the question?

A separate judgment, made BEFORE generation.

The existing defence (verify.py) checks that every figure in an answer appears
in a retrieved source. That stops the model INVENTING numbers, and it is
effective at that. It cannot see the failure that actually dominates on
unanswerable questions: a number that is genuinely present in the sources,
correctly quoted, and attached to the wrong obligation.

The measured example: asked what minimum continuing interest a SPONSOR must
keep in a Category II AIF -- a SEBI requirement absent from this corpus -- the
system answered "10 percent of the corpus". That figure is real, and it is in
the retrieved chunk. It is RBI's ceiling on a REGULATED ENTITY's contribution
to an AIF. Correct number, correct document, wrong obligation-holder. Grounding
verification passes it, because the failure is relevance, not grounding.

Asking the same model to answer and to police its own relevance in one pass
does not work: by the time it is writing an answer it has already committed to
having one. Splitting the judgment out gives it a question it can decline
cheaply, before any answer exists to defend.

This raises false refusals -- it must, since it adds a way to say no. Both
directions are reported by `inforce abstention`, because a refusal rate without
a false-refusal rate beside it means nothing.
"""
from __future__ import annotations

import re

from . import security
from .embedding import OLLAMA_BASE, SESSION

CHAT_MODEL = "qwen2.5:3b"

SYSTEM = (
    "You decide whether a set of excerpts from Reserve Bank of India documents "
    "contains the answer to a question. You do NOT answer the question.\n\n"
    "Reply with exactly one word on the first line: YES or NO.\n"
    "On the second line, quote the sentence that answers it, or write NONE.\n\n"
    "Say YES only if an excerpt states the specific fact asked for, about the "
    "same subject and the same kind of entity the question is about. RBI "
    "publishes near-identical rules for different entity classes and different "
    "obligations; a number that looks right but governs a different party or a "
    "different activity is NOT an answer.\n"
    "Say NO if the excerpts are merely on a related topic, if they give a "
    "figure for a different obligation, or if answering would need anything "
    "beyond what is written.\n"
    "When uncertain, say NO."
)

_VERDICT = re.compile(r"^\s*(YES|NO)\b", re.IGNORECASE)


def can_answer(
    question: str, hits, *, model: str = CHAT_MODEL, timeout: int = 300,
    max_chars: int = 6000,
) -> tuple[bool, str]:
    """(answerable, supporting quote or reason).

    Fails OPEN — returns True on any error. A transport failure must not
    silently turn into a refusal, or an outage would masquerade as excellent
    abstention.
    """
    if not hits:
        return False, "no evidence retrieved"

    blocks, used = [], 0
    for hit in hits:
        block = (f"[{hit.rank}] {hit.title or hit.doc_key}\n"
                 f"{security.wrap(hit.text[:1200])}")
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)

    prompt = ("Excerpts:\n\n" + "\n\n".join(blocks)
              + f"\n\nQuestion: {question}\n\nDoes an excerpt state the answer?")
    try:
        resp = SESSION.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "stream": False,
                  "options": {"temperature": 0, "num_predict": 120},
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": prompt}]},
            timeout=timeout)
        resp.raise_for_status()
        raw = (resp.json().get("message", {}) or {}).get("content", "") or ""
    except Exception:  # noqa: BLE001 - see docstring: fail open
        return True, "answerability check unavailable"

    m = _VERDICT.search(raw)
    if not m:
        return True, "unparsed answerability verdict"
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    quote = lines[1] if len(lines) > 1 else ""
    return m.group(1).upper() == "YES", quote


_SCORE = re.compile(r"\b(10|[0-9])\b")

SCORE_SYSTEM = (
    "You rate how well a set of excerpts from Reserve Bank of India documents "
    "answers a question. You do NOT answer the question.\n\n"
    "Reply with a single integer 0-10 on the first line and nothing else.\n"
    "  10 - an excerpt states exactly the fact asked for, for the same subject "
    "and the same kind of regulated entity\n"
    "   5 - an excerpt is about the right subject but states a different fact, "
    "or the same fact for a different kind of entity\n"
    "   0 - nothing in the excerpts bears on the question\n\n"
    "RBI publishes near-identical rules for different entity classes and "
    "different obligations. A figure that looks right but governs a different "
    "party or a different activity is a 3, not a 9."
)


def score(question: str, hits, *, model: str = CHAT_MODEL, timeout: int = 300,
          max_chars: int = 6000) -> int:
    """0-10 relevance score, or -1 if unavailable.

    A calibrated score rather than a YES/NO verdict. The binary version
    collapsed: told to default to NO when uncertain, qwen2.5:3b answered NO to
    almost everything -- 100% of unanswerable questions and 98% of answerable
    ones -- which is a stuck switch, not a judgment.

    A score also makes the operating point tunable, and lets the whole
    threshold sweep be done offline from one scoring pass. -1 (unavailable) is
    distinct from 0 (irrelevant) so a transport failure can be excluded rather
    than silently read as a refusal.
    """
    if not hits:
        return 0
    blocks, used = [], 0
    for hit in hits:
        block = (f"[{hit.rank}] {hit.title or hit.doc_key}\n"
                 f"{security.wrap(hit.text[:1200])}")
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    prompt = ("Excerpts:\n\n" + "\n\n".join(blocks)
              + f"\n\nQuestion: {question}\n\nScore 0-10:")
    try:
        resp = SESSION.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "stream": False,
                  "options": {"temperature": 0, "num_predict": 8},
                  "messages": [{"role": "system", "content": SCORE_SYSTEM},
                               {"role": "user", "content": prompt}]},
            timeout=timeout)
        resp.raise_for_status()
        raw = (resp.json().get("message", {}) or {}).get("content", "") or ""
    except Exception:  # noqa: BLE001 - unavailable, not irrelevant
        return -1
    m = _SCORE.search(raw)
    return int(m.group(1)) if m else -1


CHUNK_SYSTEM = (
    "Rate 0-10 how directly the excerpt answers the question. "
    "10 = it states exactly the fact asked for, for the same kind of entity. "
    "5 = right subject, different fact. 0 = unrelated. "
    "Reply with one integer only."
)


def score_per_chunk(question: str, hits, *, model: str = CHAT_MODEL,
                    timeout: int = 120, top: int = 8) -> int:
    """Best per-chunk relevance score, 0-10.

    One small call per chunk rather than one large call over all of them.
    `score()` above rates the whole 6,000-character context at once and returns
    0 for everything -- including questions the system went on to answer
    correctly. That is not the model being degenerate: on a hundred-character
    context it returns 10 for an obviously relevant pair and 0 for an obviously
    irrelevant one. It is a context-length limit, and the binary gate's constant
    NO was the same failure wearing a different output format.

    Scoring chunk by chunk keeps every judgment inside the regime where the
    3b model actually discriminates. It is also ~8x cheaper per call than the
    whole-context version despite making eight times as many calls, because
    prompt processing dominates and each prompt is a fraction of the size.

    The excerpt stays FENCED and capped at 900 characters. Both were challenged
    and both survived measurement. Inspecting six wrongly-refused questions
    suggested the fence was the problem -- on one, an unfenced chunk scored 10
    where the fenced version scored 0 -- but rescoring all 139 showed that
    generalised in the wrong direction: unfencing at 1800 characters fixed
    ZERO false refusals (still 13, still 56.9%) while raising the unanswerable
    mean score 0.19 -> 0.70 and letting three times as many fabrications
    through (2.5% -> 7.6%), net +14 -> +10. A six-case diagnostic is a
    hypothesis, not a result.

    Returns -1 only if every chunk failed, so an outage stays distinguishable
    from a genuine zero.
    """
    if not hits:
        return 0
    scores = []
    for hit in hits[:top]:
        try:
            resp = SESSION.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": model, "stream": False,
                      "options": {"temperature": 0, "num_predict": 6},
                      "messages": [
                          {"role": "system", "content": CHUNK_SYSTEM},
                          {"role": "user", "content":
                           f"Excerpt:\n{security.wrap(hit.text[:900])}\n\n"
                           f"Question: {question}\n\nScore:"}]},
                timeout=timeout)
            resp.raise_for_status()
            raw = (resp.json().get("message", {}) or {}).get("content", "") or ""
        except Exception:  # noqa: BLE001
            continue
        m = _SCORE.search(raw)
        if m:
            scores.append(int(m.group(1)))
    return max(scores) if scores else -1
