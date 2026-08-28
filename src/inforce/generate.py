"""M3 — answer generation with a local model, plus source listing.

The prompt deliberately says nothing about currency, repeal or dates. That
omission is the experiment: the baseline is given whatever retrieval returns
and has no way to know a circular was withdrawn. M7 adds the missing dimension.
"""
from __future__ import annotations

from . import answerable, security, verify
from .embedding import OLLAMA_BASE, SESSION
from .retrieve import Hit

CHAT_MODEL = "qwen2.5:3b"  # 1.9 GB — fits the 4 GB card; 7b would spill to CPU

SYSTEM = (
    "You answer questions about Indian banking regulation using only the "
    "provided excerpts from Reserve Bank of India documents. Cite the source "
    "number in square brackets, like [2], for every claim. If the excerpts do "
    "not contain the answer, say so plainly.\n\n" + security.SYSTEM_GUARD
)


def build_prompt(
    question: str, hits: list[Hit], *, max_chars: int = 6000
) -> tuple[str, list[str]]:
    """Return (prompt, warnings).

    Retrieved text is third-party content, so each chunk is fenced and its own
    delimiters neutralised before it enters the prompt — otherwise a chunk could
    close the fence early and continue as apparent instruction. Chunks matching
    injection patterns are reported rather than dropped: silently discarding a
    genuine regulatory sentence is a worse failure than surfacing a suspect one.
    """
    blocks, warnings, used = [], [], 0
    for hit in hits:
        snippet = hit.text[:1200]
        for finding in security.scan(snippet):
            warnings.append(f"[{hit.rank}] {hit.doc_key} - {finding}")
        block = (f"[{hit.rank}] {hit.title or hit.doc_key}\n"
                 f"{security.wrap(snippet)}")
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    context = "\n\n".join(blocks)
    return (f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\n"
            f"Answer with citations:"), warnings


def answer(
    question: str, hits: list[Hit], *, model: str = CHAT_MODEL, timeout: int = 600,
    log=None, strict: bool = True, gate: bool = False,
) -> tuple[str, "verify.Verdict"]:
    """Generate an answer and verify it against the sources it was given.

    Returns (answer_to_show, verdict). With `strict`, an answer containing a
    figure absent from every retrieved chunk — or citing a source that was never
    retrieved — is withheld. That is what closes the injection gap at the
    generation boundary: fencing asks the model to behave, this checks whether
    it did, and the check does not care how it was persuaded.

    `gate` runs an answerability judgment FIRST and declines outright if the
    evidence does not address the question. It targets the gap `strict` cannot
    see: a figure genuinely present in the sources but governing a different
    obligation passes every grounding check, because the failure is relevance
    rather than grounding.

    IT IS OFF BY DEFAULT BECAUSE IT DOES NOT WORK. Measured on 79 unanswerable
    and 60 answerable questions, it prevented 29 fabrications and destroyed 33
    correct answers — a ratio of 0.88, worse than one for one. It does not
    discriminate; it very nearly always says no, refusing 100% of unanswerable
    questions and 98% of answerable ones. "Refuse everything" scores perfectly
    on the abstention rate and is useless, which is why that rate is never
    reported without the false-refusal rate beside it.

    The likely cause is the judge model: qwen2.5:3b, told to default to NO when
    uncertain, collapses to a constant NO rather than a judgment. A larger judge,
    or a calibrated score instead of a binary verdict, is where to look next.
    The code and its harness stay because the negative result is reproducible
    and worth keeping — `inforce abstention --no-gate` measures the other side.
    """
    if gate:
        ok, why = answerable.can_answer(question, hits, timeout=timeout)
        if not ok:
            if log:
                log(f"answerability gate declined: {why}")
            return (
                "I can't answer that from the retrieved sources. The excerpts "
                "do not state the specific requirement asked about.",
                verify.Verdict(supported=True, unsupported=[],
                               invalid_citations=[], cited=[], checked=0),
            )

    prompt, warnings = build_prompt(question, hits)
    if warnings and log:
        for w in warnings:
            log(f"injection-shaped text in retrieved source: {w}")
    resp = SESSION.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 400},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    draft = resp.json().get("message", {}).get("content", "").strip()

    shown, verdict = verify.gate(draft, hits, strict=strict)
    if not verdict.supported and log:
        log(f"answer withheld — {verdict.reason}")
    return shown, verdict
