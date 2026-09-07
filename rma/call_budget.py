"""Call-budget policy: construct first, verify with what is left.

Why this exists. Metered against Danus on First Proof batch-2 problem 1 with an
eight-call ceiling, RMA served one proposal call, five verification calls and one
literature call, then emitted nothing. The cause is structural, not incidental:
one RMA verification is FOUR model calls (``completeness_gate`` once plus
``enumerate_gaps`` at ``RMA_GAP_SAMPLES=3``), it runs every round, and it runs
whether or not a proof exists yet. Two rounds of that is eight calls -- the whole
budget -- spent auditing a document that was never written.

Danus, measured on the same backend and budget, does not have this failure. Its
verifier is cheap per unit and strictly downstream: one claim at a time, and only
after a worker has submitted something. The policy here borrows that shape
without borrowing its machinery:

  1. CONSTRUCTION FLOOR. A fixed share of the budget is reserved for producing a
     proof. Verification cannot draw on it.
  2. NOTHING TO VERIFY, NO VERIFICATION. A stub, an empty document, or a backend
     refusal is not a proof, and auditing it is pure waste.
  3. SAMPLES SCALE WITH WHAT IS LEFT. The three-sample gap ensemble is a variance
     reduction, not a correctness requirement; under a tight budget one sample
     that runs beats three that cannot.

Set ``RMA_CALL_BUDGET`` to enable. Unset (the default) means unlimited and every
check below returns permissive, so normal operation is unchanged.
"""
from __future__ import annotations

import os
import re
import threading

_LOCK = threading.Lock()
_SPENT = 0

#: Share of the budget reserved for construction; verification may use the rest.
CONSTRUCTION_FLOOR = float(os.environ.get("RMA_CONSTRUCTION_FLOOR", "0.5"))

#: A document shorter than this is a stub, not a proof worth auditing.
MIN_PROOF_CHARS = int(os.environ.get("RMA_MIN_PROOF_CHARS", "1500"))

# Phrases a metering layer or a provider returns INSTEAD of mathematics. Writing
# one of these into a solution file is the failure mode this catches: the run
# produced a refusal and reported it as a proof.
_REFUSAL_RE = re.compile(
    r"budget exhausted|call budget for this run is spent|rate.?limit(ed)?"
    # The subscription's real wording is "You've hit your session limit · resets
    # 12pm (America/Chicago)" -- 61 characters that an eight-call arm swallowed
    # eight times while this pattern only knew the phrasing "you have hit".
    r"|you(?:'?ve| ?have)? ?hit your (?:usage|session) limit|quota exceeded"
    r"|session limit . resets|usage limit reached"
    r"|service is (currently )?unavailable",
    re.I,
)


def budget() -> int:
    """Total calls allowed, or 0 for unlimited."""
    try:
        return max(0, int(os.environ.get("RMA_CALL_BUDGET", "0")))
    except ValueError:
        return 0


def enabled() -> bool:
    return budget() > 0


def spend(n: int = 1) -> None:
    global _SPENT
    with _LOCK:
        _SPENT += n


def spent() -> int:
    return _SPENT


def remaining() -> int:
    b = budget()
    return (b - _SPENT) if b else 10**9


def is_refusal(text: str | None) -> bool:
    """True when the text is an infrastructure message rather than mathematics."""
    if not text:
        return False
    head = text.strip()[:400]
    return bool(_REFUSAL_RE.search(head))


def looks_substantive(solution_text: str | None) -> bool:
    """Is there something here worth spending verification calls on?"""
    if not solution_text:
        return False
    body = solution_text.strip()
    if is_refusal(body):
        return False
    return len(body) >= MIN_PROOF_CHARS


def may_verify(solution_text: str | None) -> tuple[bool, str]:
    """Should a verification pass run now? Returns (allowed, reason)."""
    if not enabled():
        return True, "no budget set"
    if not looks_substantive(solution_text):
        return False, "no substantive proof to verify yet"
    floor = int(round(budget() * CONSTRUCTION_FLOOR))
    if _SPENT < floor:
        return False, f"construction floor not met ({_SPENT}/{floor} calls)"
    if remaining() <= 1:
        return False, "budget spent; keeping the proof rather than auditing it"
    return True, "ok"


def gap_samples(default: int) -> int:
    """Shrink the gap-finder ensemble to what the remaining budget can pay for."""
    if not enabled():
        return default
    # Leave at least one call for anything downstream of the gap finder.
    return max(1, min(default, remaining() - 1))


def reset() -> None:
    global _SPENT
    with _LOCK:
        _SPENT = 0


_AUDIT_RE = re.compile(
    r"\b(the (submitted|given|current|presented) (document|proof|draft))"
    r"|\b(as (presented|written|submitted))\b"
    r"|\breferee\b|\bthe author(s)?\b"
    r"|\b(I|we) (have )?(reviewed|audited|checked) the\b"
    r"|\bthis (review|audit|assessment)\b", re.I)


def is_regression(seed: str, out: str) -> tuple[bool, str]:
    """Did an improvement pass hand back something worse than what it was given?

    Measured on First Proof batch-2: asked to improve an audited proof, RMA
    returned a referee report about it. Three blind judges said the same thing in
    three different words -- "a referee-style endorsement rather than an actual
    proof", "an audit of an unseen document", "meta-commentary that asserts an
    unseen proof closes" -- and completeness fell from 8.0 to 1.0 on one problem.
    The cause is structural: a round begins with the critic, so critique is the
    loop's dominant output, and the delivery path writes whatever the store's
    latest proof revision holds, which by then is the critique.

    Two symptoms are cheap and reliable to detect: the document collapsed to a
    fraction of its input, and it talks ABOUT a proof instead of containing one.
    Either means the pass regressed, and the honest delivery is the document we
    were given -- an improvement step must never be able to score below its own
    input.
    """
    seed_t, out_t = (seed or "").strip(), (out or "").strip()
    if not seed_t:
        return False, ""
    if not out_t:
        return True, "produced nothing"
    ratio = len(out_t) / len(seed_t)
    if ratio < 0.6:
        return True, f"collapsed to {ratio:.0%} of the document it was given"
    hits = len(_AUDIT_RE.findall(out_t))
    if hits >= 3 and ratio < 1.2:
        return True, f"reads as a review of the proof, not a proof ({hits} markers)"
    return False, ""
