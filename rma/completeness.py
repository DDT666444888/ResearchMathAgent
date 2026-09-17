"""Completeness-driven solving.

The verifier historically gated on *structural proxies* for completeness (proof
word count, presence of lemma/proof environments, boundary-case keywords). Those
reward the appearance of completeness, which is why proofs came back
clarity-high / completeness-low. This module adds the semantic machinery that
makes *completeness itself* a driver of the solve loop:

  #1 semantic completeness gate     — score the current proof's completeness
                                       (LLM rubric dimension) and, below a
                                       threshold, emit the judge's gap notes as
                                       hard verifier issues so refinement must
                                       close them.
  #2 gap-enumeration verifier       — multi-sample structured gap finder
                                       (unproved lemma / cited-black-box /
                                       logical leap / unchecked-finite), majority
                                       voted, with locations.
  #3 lemma-DAG completeness         — parse the proof into a claim dependency
                                       graph and measure the proved-leaf fraction.
  #4 reduction / partial-credit     — prompt slot + fallback that commits to the
                                       strongest *completely* provable sub-result.
  #5 per-gap refinement targeting   — pick the single highest-priority open gap so
                                       the refiner fixes it instead of rewriting
                                       the whole proof.
  #6 code-discharge of finite steps — require finite/numeric claims to be closed
                                       with an attached runnable check.
  #7 telemetry                      — per-round completeness trajectory,
                                       gaps opened/closed, question-shape tag.
  #8 fidelity gate                  — is the chosen READING of the problem the one
                                       its own citations point to? #1-#3 all grade
                                       a proof given whatever interpretation it
                                       committed to; none asks whether that
                                       interpretation is the intended one. A proof
                                       can be complete, gap-free, and answering a
                                       trivialized version of a citation-backed hard
                                       problem (see fidelity_gate's docstring for the
                                       measured case that motivated this).

Everything degrades gracefully: with no LLM backend (rma-skeleton / offline) the
semantic passes return no issues and the structural DAG/telemetry still work, so
the pipeline never hard-fails on account of this module.

See documents/EVALUATION.md and the paper's Completeness section.
"""
from __future__ import annotations

import json
import os
import re
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from .budget import count_tokens
from .models import (
    call_anthropic,
    call_claude_code,
    should_use_anthropic,
    should_use_claude_code,
)

# Completeness dimension is 0..10; below this the proof is treated as unfinished
# and its gaps become hard (error-severity) verifier issues. Override via env.
COMPLETENESS_THRESHOLD = float(os.environ.get("RMA_COMPLETENESS_MIN", "8"))
# How many independent gap-enumeration samples to draw; a gap must appear in a
# strict majority to be gated on (keeps single-shot judge noise from blocking).
GAP_SAMPLES = max(1, int(os.environ.get("RMA_GAP_SAMPLES", "3")))

# Issue codes this module introduces. All are error-severity so _verify_solution
# (which gates on any error) fails until they are resolved.
INCOMPLETE_STEP = "incomplete_step"          # #1 semantic completeness gate
UNPROVED_LEMMA = "unproved_lemma"            # #2 named sub-result asserted, not proved
CITED_BLACKBOX = "cited_blackbox_crux"       # #2 crux hidden behind a citation
LOGICAL_LEAP = "logical_leap"                # #2 "it follows that" with no argument
UNCHECKED_FINITE = "unchecked_finite_claim"  # #6 finite/numeric claim with no code check
MISREAD_STATEMENT = "misread_problem_statement"  # #8 wrong reading, not an incomplete one
COMPLETENESS_ISSUE_CODES = {
    INCOMPLETE_STEP, UNPROVED_LEMMA, CITED_BLACKBOX, LOGICAL_LEAP, UNCHECKED_FINITE,
    MISREAD_STATEMENT,
}

# Real ceiling on how much of the proof a grading call may see, in TOKENS —
# matches the round loop's own DEFAULT_CONTEXT_BUDGET (rma/config.py, 60,000)
# minus headroom for the problem statement/system prompt/JSON schema, not the
# ~24,000/16,000-CHARACTER limits this replaces below. Those were 4-6x tighter
# than even a token-based reading of the same budget, and on a real solve run
# (prob-09, first_proof_2, 2026-09-02) cut a 44,271-char delivered proof before
# its own theorem-main/theorem-closed sections -- the critic then reported
# those sections as "entirely absent" for 2 rounds (~$27 of otherwise-wasted
# spend) even though they were present later in the same document.
_MAX_PROOF_TOKENS_FOR_GRADING = int(os.environ.get("RMA_GRADING_PROOF_TOKEN_BUDGET", "40000"))


def _prepare_proof_for_grading(solution_text: str) -> tuple[str, bool]:
    """Fit ``solution_text`` for a grading prompt without silently cutting off
    the part of the proof that matters most: its own conclusion.

    A plain prefix slice can -- and, on a real run, did -- cut a long proof
    before it ever reaches the theorem it is proving, then have the critic
    report the missing tail as the proof being incomplete. Two changes fix
    that: the ceiling is a real token budget instead of an arbitrary character
    count picked before proofs this long existed, and when a proof genuinely
    exceeds it, the cut keeps BOTH ends (setup and conclusion) with the middle
    elided and explicitly marked, instead of only the head.

    Returns (text_to_grade, was_truncated).
    """
    if count_tokens(solution_text) <= _MAX_PROOF_TOKENS_FOR_GRADING:
        return solution_text, False
    # ~4 chars/token, the same fallback rate budget.count_tokens uses when no
    # tokenizer is installed -- good enough for a split point; not scored.
    char_budget = _MAX_PROOF_TOKENS_FOR_GRADING * 4
    head = solution_text[: char_budget * 3 // 5]
    tail = solution_text[-(char_budget * 2 // 5):]
    omitted = len(solution_text) - len(head) - len(tail)
    stitched = (
        f"{head}\n\n"
        f"...[{omitted} characters omitted here -- the proof continues below]...\n\n"
        f"{tail}"
    )
    return stitched, True


_TRUNCATION_NOTE = (
    "NOTE: this proof was too long to include in full; you are seeing its "
    "beginning and end only, with a middle section omitted and marked as such "
    "above. Do NOT report content in the omitted region as 'missing', 'absent', "
    "or 'never proved' -- you cannot see it, so you cannot make that claim. Only "
    "report a gap if it is visibly missing WITHIN the text actually shown to you."
)


# ─────────────────────────────────────────────────────────────────────────────
# model plumbing (mirrors rma.solve._model_verify_proof so the backend choice is
# identical to whatever the solve run is using)
# ─────────────────────────────────────────────────────────────────────────────
def _backend_available(args: Namespace) -> bool:
    model_name = getattr(args, "model_name", "rma-skeleton")
    provider = getattr(args, "model_provider", "auto")
    return bool(should_use_anthropic(model_name, provider)
                or should_use_claude_code(model_name, provider))


def _model_call(args: Namespace, system: str, prompt: str, max_tokens: int = 2048) -> str:
    """One-shot judge call. Uses the lightweight subscription helper
    (``webapp.llm.complete`` → single-turn ``claude -p``, no agent loop) that the
    rubric/GenRM evaluators use — NOT the heavy 12-turn proof-generation driver.
    Falls back to the Anthropic API when an API model is configured."""
    model_name = getattr(args, "model_name", "rma-skeleton")
    provider = getattr(args, "model_provider", "auto")
    # Preferred: one-shot subscription completion.
    try:
        from webapp.llm import complete as _complete  # repo root on sys.path under `python -m rma`
        out = _complete(prompt, system=system)
        if out and out.strip():
            return out.strip()
    except Exception:
        pass
    # Fallback: pay-per-token API when explicitly using a claude-* API model.
    try:
        if should_use_anthropic(model_name, provider):
            resp = call_anthropic(model=model_name, system=system, prompt=prompt,
                                  max_tokens=max_tokens, temperature=0.0)
            return (resp.text or "").strip()
    except Exception:
        pass
    return ""


def _parse_json_block(raw: str):
    raw = (raw or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    # Clean parse first.
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Otherwise decode the FIRST complete JSON value (object or array) starting at
    # the earliest '{' or '['; raw_decode ignores any trailing prose the model
    # appended (e.g. a "Note: ..." after the JSON). Earliest-start matters: a
    # completeness object {"completeness":..,"missing":[...]} must not be mistaken
    # for its inner array.
    starts = sorted(i for i in (raw.find("{"), raw.find("[")) if i >= 0)
    dec = json.JSONDecoder()
    for start in starts:
        try:
            value, _ = dec.raw_decode(raw[start:])
            return value
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# #3  lemma-DAG completeness
# ─────────────────────────────────────────────────────────────────────────────
_CLAIM_ENV = r"(?:theorem|lemma|claim|proposition|corollary)"


def lemma_dag(solution_text: str) -> dict:
    """Claim-graph summary for the current proof.

    Delegates to :mod:`rma.claims`, which parses claims AND their dependency
    edges. The previous implementation matched only the literal environment
    names theorem/lemma/claim/proposition/corollary, so on the repo's real
    proofs — which declare `\\newtheorem{qsixlemma}{Lemma}` and friends so
    per-problem proofs can be concatenated without counter clashes — it
    reported nodes=0, completeness_fraction=0.0 for a document with 17 claims.
    The proved-leaf fraction the paper reports was identically zero as a result.

    Keeps the historical keys so existing readers (verification reports,
    completeness_trajectory.json) keep working, and adds the edge-derived ones.
    """
    from .claims import parse_claims

    graph = parse_claims(solution_text)
    nodes = len(graph)
    proved = sum(1 for c in graph.claims if c.proved)
    return {
        "nodes": nodes,
        "proved": proved,
        "unproved_titles": [c.title for c in graph.claims if not c.proved],
        "completeness_fraction": graph.proved_fraction(),
        # New, edge-derived: the paper's structural checker measures the
        # TERMINAL claims, not all claims (main.tex:368).
        "edges": len(graph.edges),
        "terminal_nodes": len(graph.terminal_claims()),
        "proved_terminal_fraction": graph.proved_terminal_fraction(),
        "cycles": graph.cycles(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# #6  code-discharge of finite / numeric claims
# ─────────────────────────────────────────────────────────────────────────────
_FINITE_CLAIM_RE = re.compile(
    r"(by (?:a |direct )?computation|numerical(?:ly)? (?:check|verif)|"
    r"one (?:can )?(?:verif|check)|it is (?:easy|straightforward) to (?:check|verify)|"
    r"exhaustive(?:ly)?|finite (?:check|verification)|for all \$?n ?\\?le)",
    re.IGNORECASE)
_CODE_EVIDENCE_RE = re.compile(
    r"\\begin\{(?:verbatim|lstlisting|minted)\}|sympy|numpy|"
    r"verified (?:by|via) (?:code|computation)|computational appendix|"
    r"the following (?:script|code)", re.IGNORECASE)


def unchecked_finite_issues(solution_text: str) -> list[dict[str, str]]:
    """#6: flag finite/numeric claims that assert a computation without attaching
    a runnable check. A step backed by code counts as complete; an asserted
    finite claim does not."""
    if not _FINITE_CLAIM_RE.search(solution_text):
        return []
    if _CODE_EVIDENCE_RE.search(solution_text):
        return []
    hit = _FINITE_CLAIM_RE.search(solution_text)
    return [{
        "code": UNCHECKED_FINITE,
        "severity": "error",
        "message": ("A finite/numeric claim is asserted ('...{}...') but no runnable "
                    "check is attached. Discharge it with a short verification "
                    "(sympy/numeric) and include the code + its output, or replace "
                    "it with a symbolic argument.").format(hit.group(0)),
        "detail": hit.group(0),
    }]


# ─────────────────────────────────────────────────────────────────────────────
# #2  gap-enumeration verifier (multi-sample, majority voted)
# ─────────────────────────────────────────────────────────────────────────────
_GAP_SYSTEM = (
    "You are a ruthless completeness referee for research mathematics. Your only "
    "job is to find every place the proof is NOT actually finished: a named "
    "lemma/claim stated but not proved, a crux step hidden behind a citation "
    "without instantiating it here, a logical leap ('it follows that', 'clearly', "
    "'one can show') with no argument, or a finite/numeric claim asserted without "
    "a check. Do not reward clarity or length. Be specific and location-anchored."
)
_GAP_PROMPT = (
    "Enumerate the completeness gaps in this LaTeX proof. Return ONLY a JSON array "
    "(no prose, no fences). Each element:\n"
    '{"code": "unproved_lemma"|"cited_blackbox_crux"|"logical_leap"|'
    '"unchecked_finite_claim", "location": "<lemma name / section / the phrase>", '
    '"claim": "<the specific unproved statement>", "critical": true|false}\n'
    "critical=true means the overall result does NOT hold without closing this gap. "
    "If the proof is genuinely complete, return exactly: []\n\n"
    "PROOF:\n{proof}"
)


def enumerate_gaps(solution_text: str, args: Namespace,
                   samples: int = GAP_SAMPLES) -> list[dict[str, str]]:
    """#2: multi-sample structured gap finder. A gap is gated on only when it
    appears in a strict majority of samples (dampens single-shot noise)."""
    if not _backend_available(args):
        return []
    # Under a call budget the three-sample ensemble is a variance reduction we
    # may not be able to afford; one sample that runs beats three that cannot.
    from . import call_budget as _budget
    samples = _budget.gap_samples(samples)
    from collections import Counter
    tally: Counter = Counter()
    detail: dict[tuple, dict] = {}
    valid = 0
    proof_for_grading, truncated = _prepare_proof_for_grading(solution_text)
    gap_prompt = _GAP_PROMPT.replace("{proof}", proof_for_grading)
    if truncated:
        gap_prompt = f"{_TRUNCATION_NOTE}\n\n{gap_prompt}"
    for _ in range(max(1, samples)):
        raw = _model_call(args, _GAP_SYSTEM, gap_prompt)
        parsed = _parse_json_block(raw)
        if not isinstance(parsed, list):
            continue
        valid += 1
        seen = set()
        for it in parsed:
            if not isinstance(it, dict):
                continue
            code = str(it.get("code", "")).strip()
            if code not in COMPLETENESS_ISSUE_CODES:
                continue
            key = (code, str(it.get("location", ""))[:60].lower().strip())
            if key in seen:
                continue
            seen.add(key)
            tally[key] += 1
            detail[key] = it
    if valid == 0:
        return []
    threshold = valid // 2 + 1  # strict majority
    issues: list[dict[str, str]] = []
    for key, count in tally.items():
        if count < threshold:
            continue
        it = detail[key]
        crit = bool(it.get("critical", True))
        issues.append({
            "code": key[0],
            # non-critical gaps are warnings so they inform without blocking
            "severity": "error" if crit else "warning",
            "message": f"{it.get('claim', 'Unproved step')} (at {it.get('location','?')})",
            "detail": f"agreement {count}/{valid}",
            "input_truncated": truncated,
        })
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# #1  semantic completeness gate (rubric completeness dimension → issues)
# ─────────────────────────────────────────────────────────────────────────────
_COMPLETE_SYSTEM = (
    "You are grading ONLY proof completeness for a research-level mathematics "
    "proof: are all essential steps explicitly present with no unjustified leaps? "
    "Ignore clarity and style. Be calibrated and skeptical of citations used as "
    "black boxes."
)
_COMPLETE_PROMPT = (
    "Score the completeness of this proof on 0-10 (10 = every step justified, "
    "0 = sketch). List the specific missing steps. Return ONLY JSON:\n"
    '{"completeness": <0-10>, "missing": ["<specific missing step>", ...]}\n\n'
    "PROBLEM:\n{problem}\n\nPROOF:\n{proof}"
)


def completeness_gate(problem_text: str, solution_text: str,
                      args: Namespace) -> tuple[float | None, list[dict[str, str]]]:
    """#1: returns (completeness_score, issues). Below COMPLETENESS_THRESHOLD the
    judge's ``missing`` notes become hard issues that block verification."""
    if not _backend_available(args):
        return None, []
    proof_for_grading, truncated = _prepare_proof_for_grading(solution_text)
    prompt = (_COMPLETE_PROMPT.replace("{problem}", problem_text[:6000])
                              .replace("{proof}", proof_for_grading))
    if truncated:
        prompt = f"{_TRUNCATION_NOTE}\n\n{prompt}"
    raw = _model_call(args, _COMPLETE_SYSTEM, prompt)
    parsed = _parse_json_block(raw)
    if not isinstance(parsed, dict) or "completeness" not in parsed:
        return None, []
    try:
        score = float(parsed["completeness"])
    except Exception:
        return None, []
    if score >= COMPLETENESS_THRESHOLD:
        return score, []
    missing = parsed.get("missing") or []
    issues = []
    for m in missing[:8]:
        issues.append({
            "code": INCOMPLETE_STEP,
            "severity": "error",
            "message": f"Completeness {score:.0f}/10 (<{COMPLETENESS_THRESHOLD:.0f}); "
                       f"missing step: {str(m)[:300]}",
            "detail": str(m)[:300],
            "input_truncated": truncated,
        })
    if not issues:  # low score but no itemized notes
        issues.append({
            "code": INCOMPLETE_STEP,
            "severity": "error",
            "message": f"Proof completeness scored {score:.0f}/10, below the "
                       f"{COMPLETENESS_THRESHOLD:.0f}/10 gate; essential steps are missing.",
            "detail": str(score),
            "input_truncated": truncated,
        })
    return score, issues


# ─────────────────────────────────────────────────────────────────────────────
# #8  fidelity gate — right problem, not just a complete proof of *some* problem
# ─────────────────────────────────────────────────────────────────────────────
_FIDELITY_SYSTEM = (
    "You are checking whether a proof engages with the SUBSTANCE of the problem it "
    "claims to solve, not whether its individual steps are valid -- a different "
    "question from completeness. A proof can be internally airtight and still answer "
    "an easier problem than the one posed, by reading an ambiguous or loosely-worded "
    "condition in whatever way makes the problem collapse: treating a strict "
    "inequality as unbounded, discarding a stated identification, or exploiting a "
    "literal reading that lets a free parameter be pushed to a degenerate limit. This "
    "is most likely exactly when the problem names a specific paper, theorem, "
    "conjecture, or author+year: a trivializing misreading is how a hard, "
    "citation-backed problem quietly gets 'solved' by accident. Be skeptical of any "
    "proof whose answer collapses a problem that cites specific, nontrivial prior "
    "work into something easy -- but do not invent a citation or a mismatch that "
    "is not really there."
)
_FIDELITY_PROMPT = (
    "PROBLEM (note any cited papers, named theorems, authors, or years):\n{problem}\n\n"
    "PROOF / ANSWER:\n{proof}\n\n"
    "Does this proof's approach genuinely engage with the difficulty the problem's own "
    "citations point to, or does it reach its answer via a reading of some condition "
    "that trivializes the problem relative to what those citations establish? If the "
    "problem cites no specific prior result, or the proof's approach clearly engages "
    "the real difficulty, say so and return no issue. Return ONLY JSON:\n"
    '{"trivializes": true|false, "cited_result": "<what the problem cites, or empty>", '
    '"mismatch": "<one sentence naming the specific misreading, or empty>"}'
)


def fidelity_gate(problem_text: str, solution_text: str,
                  args: Namespace) -> list[dict[str, str]]:
    """#8: is the proof's chosen READING of the problem the one intended, not just
    complete given whatever reading it picked?

    Motivated by a real, measured failure, not a hypothetical: on First Proof
    batch-2's flat-Moebius-band problem, three independently built systems (RMA,
    Danus, and RMA-through-Danus's-own-verify-gate) each produced a complete,
    internally valid proof of inf=0 by reading a "strictly shortens every boundary
    subarc" condition as an unbounded one-sided contraction -- when the problem's
    own citation (Schwartz, Annals of Math. 2025) is for the isometric threshold
    sqrt(3). ``completeness_gate`` (#1) never catches this class of error: its own
    system prompt scopes it to "ONLY proof completeness... ignore clarity and
    style", which a wrong-but-complete proof of the wrong problem satisfies fully.
    This gate asks the prior question instead, and is deliberately independent of
    #1-#3 so an ablation can isolate its effect.
    """
    if not _backend_available(args):
        return []
    proof_for_grading, truncated = _prepare_proof_for_grading(solution_text)
    prompt = (_FIDELITY_PROMPT.replace("{problem}", problem_text[:6000])
                              .replace("{proof}", proof_for_grading))
    if truncated:
        prompt = f"{_TRUNCATION_NOTE}\n\n{prompt}"
    raw = _model_call(args, _FIDELITY_SYSTEM, prompt)
    parsed = _parse_json_block(raw)
    if not isinstance(parsed, dict) or not parsed.get("trivializes"):
        return []
    mismatch = str(parsed.get("mismatch") or "").strip()
    if not mismatch:
        return []
    cited = str(parsed.get("cited_result") or "").strip()
    message = ("Proof may trivialize the problem relative to its own cited result"
               f"{f' ({cited})' if cited else ''}: {mismatch}")
    return [{
        "code": MISREAD_STATEMENT,
        "severity": "error",
        "message": message[:400],
        "detail": mismatch[:400],
        "input_truncated": truncated,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# #5  per-gap refinement targeting
# ─────────────────────────────────────────────────────────────────────────────
_GAP_PRIORITY = {
    CITED_BLACKBOX: 5, UNPROVED_LEMMA: 4, LOGICAL_LEAP: 3,
    INCOMPLETE_STEP: 2, UNCHECKED_FINITE: 1,
}


def select_priority_gap(issues: list[dict[str, str]]) -> dict | None:
    """#5: choose the single highest-priority completeness gap so the refiner can
    focus its whole budget on closing it rather than rewriting everything."""
    comp = [i for i in issues
            if i.get("code") in COMPLETENESS_ISSUE_CODES and i.get("severity") == "error"]
    if not comp:
        return None
    return max(comp, key=lambda i: _GAP_PRIORITY.get(i.get("code"), 0))


def focus_directive(gap: dict | None) -> str:
    """Prompt fragment steering refinement at one gap while preserving the rest."""
    if not gap:
        return ""
    return (
        "\n## PRIORITY GAP (close this fully before anything else)\n"
        f"- {gap.get('message','')}\n"
        "Keep every already-correct part of the proof VERBATIM. Spend your effort "
        "proving exactly this one gap in full detail. If it genuinely cannot be "
        "closed, invoke REDUCTION MODE below instead of hand-waving it.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# #4  reduction / partial-credit mode
# ─────────────────────────────────────────────────────────────────────────────
def reduction_directive() -> str:
    return (
        "\n## REDUCTION MODE (honesty valve)\n"
        "A completely proved partial result is worth far more than a full but "
        "hand-waved proof. If you cannot finish the full statement, explicitly: "
        "(1) state the STRONGEST sub-result (special case, one direction, a "
        "reduction to a named theorem, or a counterexample) that you CAN prove "
        "with zero gaps, prove it completely, and (2) in a clearly labelled "
        "'Remaining' subsection list exactly what is left and why. Do NOT pad the "
        "unproved part with 'it follows that' — name it as open.\n"
    )


def code_discharge_directive() -> str:
    return (
        "\n## FINITE CLAIMS MUST BE CHECKED\n"
        "Any finite or numerical claim ('by computation', 'one can verify', "
        "exhaustive check, a specific value) must be discharged with a short "
        "runnable script (sympy/numeric) whose code AND output you include in a "
        "verbatim block — or replaced by a symbolic argument. An asserted finite "
        "claim does not count as proved.\n"
    )


def solve_directives() -> str:
    """All prompt directives appended to the proposer/refiner user prompt."""
    return code_discharge_directive() + reduction_directive()


# ─────────────────────────────────────────────────────────────────────────────
# #7  telemetry
# ─────────────────────────────────────────────────────────────────────────────
def classify_question_shape(statement: str) -> str:
    """Coarse question-shape tag so analytics can slice completeness by shape."""
    s = (statement or "").lower()
    shapes = []
    if re.search(r"counterexample|does there exist|is it true|prove or disprove|whether", s):
        shapes.append("decide/counterexample")
    if re.search(r"compute|explicit|closed form|formula|evaluate|determine the value", s):
        shapes.append("compute/explicit")
    if re.search(r"necessary and sufficient|characteriz|classify|conditions on", s):
        shapes.append("characterize")
    if re.search(r"\bbound\b|asymptotic|growth|rate|estimate", s):
        shapes.append("bound/asymptotic")
    return ",".join(shapes) or "other"


def _gap_keys(issues: list[dict]) -> set:
    return {(i.get("code"), (i.get("detail") or i.get("message", ""))[:60])
            for i in issues if i.get("code") in COMPLETENESS_ISSUE_CODES}


def record_round(output_dir: Path, problem_id: str, round_idx: int,
                 completeness_score: float | None, dag: dict,
                 issues: list[dict], prev_gaps: set | None) -> set:
    """Append one round to completeness_trajectory.json and return this round's
    gap-key set (so the caller can compute opened/closed next round)."""
    cur = _gap_keys(issues)
    prev = prev_gaps or set()
    opened = sorted(cur - prev)
    closed = sorted(prev - cur)
    rec = {
        "round": round_idx,
        "ts": datetime.now(timezone.utc).isoformat(),
        "completeness_score": completeness_score,
        "dag_completeness_fraction": dag.get("completeness_fraction"),
        "dag_nodes": dag.get("nodes"),
        "dag_proved": dag.get("proved"),
        "open_gaps": len(cur),
        "gaps_opened": len(opened),
        "gaps_closed": len(closed),
    }
    try:
        path = output_dir / problem_id / "artifacts" / "completeness_trajectory.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = []
        data.append(rec)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return cur
