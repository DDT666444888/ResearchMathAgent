"""Critic — issue discovery, the paper's four complementary analyses.

main.tex:366-368:

    "The critic operation applies four complementary analyses to the current
     proof. First, an LM-based critic identifies logical gaps and returns them
     as typed issue records. Second, a deterministic structural checker parses
     the proof into claims and dependencies, measures the fraction of terminal
     claims supported by completed arguments, and flags finite or numerical
     claims that lack executable verification. Third, a semantic completeness
     judge assesses whether the proof establishes the target statement as a
     whole and, when its score falls below the completeness threshold, converts
     its itemized missing steps into severity-rated issues. Those three all
     grade the proof given whatever reading of the problem it adopted, so a
     fourth, independent fidelity analysis asks whether that reading is the
     intended one or one that trivializes the problem relative to its own
     citations."

The output is the ranked queue Q (main.tex:370): open issues ordered by
severity, then by dependency impact. This operation produces Q; the round loop
then repairs Q[:b].

Each analysis is individually disable-able (``--ablate critic.lm |
critic.structural | critic.semantic | critic.fidelity``) so the paper's
per-analysis ablation (4.3 / 3.8 / 4.1 / 4.5 vs the full 6.0) is a runnable
configuration.

The three model-backed analyses are delegated to ``rma.completeness`` — the
existing, tested seam — rather than re-implemented here, so this file makes no
direct model call. They are injectable (``analyses=``) so the whole critic runs
offline in tests with canned findings.
"""
from __future__ import annotations

from .. import claims as _claims
from .. import completeness as _completeness
from ..orchestrator import Query
from ..ranking import Issue, assign_severity, compute_impact, rank
from .base import Operation, register


# ─────────────────────────────────────────────────────────────────────────────
# the four analyses (each returns a list of rma.ranking.Issue)
# ─────────────────────────────────────────────────────────────────────────────
def structural_analysis(proof_text: str) -> list[Issue]:
    """Deterministic. No model. Parses the claim graph and flags:
    unproved terminal claims, dependency cycles, and per-claim finite/numeric
    assertions with no runnable check."""
    graph = _claims.parse_claims(proof_text)
    issues: list[Issue] = []

    proved = {c.id for c in graph.claims if c.proved}
    for claim in graph.terminal_claims():
        if claim.id not in proved:
            issues.append(Issue(
                id=f"struct-unproved-{claim.id}",
                code="unproved_lemma",
                message=f"Terminal claim {claim.display} is stated but not proved.",
                claim_id=claim.id,
                detail=claim.title,
                meta={"analysis": "structural"},
            ))

    for cycle in graph.cycles():
        anchor = cycle[0]
        issues.append(Issue(
            id=f"struct-cycle-{'-'.join(cycle)}",
            code="circular_reasoning",
            message=f"Circular dependency among claims: {' -> '.join(cycle)}.",
            claim_id=anchor,
            detail=",".join(cycle),
            meta={"analysis": "structural"},
        ))

    for claim, phrase in graph.unchecked_finite_claims():
        issues.append(Issue(
            id=f"struct-finite-{claim.id}-{abs(hash(phrase)) % 10000}",
            code="unchecked_finite_claim",
            message=f"{claim.display} asserts a finite/numeric check ('{phrase}') "
                    f"with no runnable verification attached.",
            claim_id=claim.id,
            detail=phrase,
            meta={"analysis": "structural"},
        ))
    return issues


def lm_gap_analysis(proof_text: str, args) -> list[Issue]:
    """LM critic — typed logical-gap issues. Delegates to the multi-sample,
    majority-voted gap enumerator in rma.completeness."""
    raw = _completeness.enumerate_gaps(proof_text, args)
    return _issues_from_completeness(raw, "lm")


def semantic_analysis(problem_text: str, proof_text: str, args) -> tuple[float | None, list[Issue]]:
    """Semantic completeness judge — score + itemized missing steps as issues."""
    score, raw = _completeness.completeness_gate(problem_text, proof_text, args)
    return score, _issues_from_completeness(raw, "semantic")


def fidelity_analysis(problem_text: str, proof_text: str, args) -> list[Issue]:
    """Fidelity critic — is the proof's reading of the problem the intended one,
    not just complete given whatever reading it picked? Delegates to
    rma.completeness.fidelity_gate; see that function's docstring for the
    measured failure (three independent systems, one shared misreading) that
    motivated adding this as a fourth, independent analysis."""
    raw = _completeness.fidelity_gate(problem_text, proof_text, args)
    return _issues_from_completeness(raw, "fidelity")


def _issues_from_completeness(raw: list[dict], analysis: str) -> list[Issue]:
    out = []
    for i, item in enumerate(raw or []):
        meta = dict(item)
        meta["analysis"] = analysis
        out.append(Issue(
            id=f"{analysis}-{i}",
            code=str(item.get("code", "incomplete_step")),
            message=str(item.get("message", "")),
            severity=str(item.get("severity") or "P2"),
            detail=str(item.get("detail", "")),
            meta=meta,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# dedup against issues already open in the store
# ─────────────────────────────────────────────────────────────────────────────
def _dedup_key(issue: Issue) -> tuple:
    """Two findings are the same obligation if they hit the same claim with the
    same defect kind; when no claim is anchored, fall back to a normalized
    message so the critic does not reopen the same gap every round."""
    code = (issue.code or "").lower()
    if issue.claim_id:
        return (code, issue.claim_id)
    return (code, " ".join((issue.message or "").lower().split())[:80])


def dedup_against_open(found: list[Issue], open_issues: list[Issue]) -> list[Issue]:
    seen = {_dedup_key(i) for i in open_issues}
    fresh, local = [], set()
    for issue in found:
        key = _dedup_key(issue)
        if key in seen or key in local:
            continue
        local.add(key)
        fresh.append(issue)
    return fresh


@register
class CriticOperation(Operation):
    """Run(Critic, p, S, B) -> ranked queue Q."""

    name = "critic"
    include_proof = True

    def instructions(self, ctx) -> str:
        return ("Critic: find every place the proof is not actually finished — "
                "unproved claims, logical gaps, missing justifications, unchecked "
                "computations. Return typed, location-anchored issues.")

    def query(self, ctx) -> Query:
        return Query(text=f"Discover open proof obligations for {ctx.store.problem_id}.",
                     id=f"{ctx.store.problem_id}-critic-r{ctx.round}")

    def run(self, ctx, *, analyses: dict | None = None, telemetry_path=None):
        """The critic is a compound operation: four analyses over the current
        proof, merged into one ranked queue. It still goes through Run(u,q,S,B)
        — so it gets a budget-compiled observation and a telemetry line — but
        its "invoke" is the four analyses rather than one model call. Every
        model-backed analysis is delegated to rma.completeness (or injected via
        ``analyses=``), so the operation itself makes no direct model call.
        """
        analyses = analyses or {}

        def _analyses_invoke(unit: str, observation: str):
            return self._collect(ctx, analyses)

        result = super().run(ctx, invoke=_analyses_invoke, telemetry_path=telemetry_path)
        # The parsed queue (result.artifact) now gets the store ids write-back
        # assigned, so downstream ops can link to the issues by record id.
        # write_back assigned store ids to the FRESH issues (result.written is in
        # the same order as the writeback list = the fresh issues). Propagate
        # those ids onto the fresh Issue objects, which are also in the queue.
        for issue, record in zip(ctx.extra.get("_fresh", []), result.written):
            issue.id = record.id
        # The queue (persistent + fresh) is the real artifact, not just the fresh
        # write-list that result.artifact holds.
        result.artifact = ctx.extra.get("queue", result.artifact)
        return result

    # ── the four analyses, collected into one issue list ────────────────────
    def _collect(self, ctx, analyses: dict) -> list[Issue]:
        proof_text = ctx.current_proof_text()
        args = ctx.args_namespace()
        cfg = ctx.config
        found: list[Issue] = []

        if not cfg.ablated("critic.structural"):
            found += analyses.get("structural", structural_analysis)(proof_text)
        if not cfg.ablated("critic.lm"):
            found += analyses.get("lm", lm_gap_analysis)(proof_text, args)
        if not cfg.ablated("critic.semantic"):
            comp_score, sem_issues = analyses.get("semantic", semantic_analysis)(
                ctx.problem_text, proof_text, args)
            ctx.extra["completeness_score"] = comp_score
            found += sem_issues
        # Independent of completeness on purpose: #1-#3 all grade the proof given
        # whatever reading of the problem it committed to; this asks whether that
        # reading is the intended one (see fidelity_analysis's docstring).
        if not cfg.ablated("critic.fidelity"):
            found += analyses.get("fidelity", fidelity_analysis)(
                ctx.problem_text, proof_text, args)
        return found

    # ── parse: dedup + severity + rank into the queue Q ─────────────────────
    def parse(self, found: list[Issue], ctx):
        """Q is the ranked set of ALL currently-open obligations, not only this
        round's fresh findings. A gap discovered in round r but not yet closed
        must stay in Q so the solver can re-attempt it in round r+1 — otherwise
        the repair loop only ever sees new issues and never finishes old ones.
        Deduplication applies to WRITES (don't reopen the same issue), not to
        the queue."""
        proof_text = ctx.current_proof_text()
        graph = _claims.parse_claims(proof_text)

        persistent = _open_issue_objects(ctx.store)     # already carry store ids
        fresh = dedup_against_open(found, persistent)    # the ones to write

        for issue in list(persistent) + list(fresh):
            issue.severity = assign_severity(issue, graph)
            issue.impact = compute_impact(issue, graph)

        queue = rank(list(persistent) + list(fresh), graph,
                     mode=ctx.config.order_mode, assign=False)
        ctx.extra["queue"] = queue
        ctx.extra["_fresh"] = fresh
        return fresh   # writeback writes only the fresh issues

    # ── writeback: the fresh issues -> I records with provenance ─────────────
    def writeback(self, fresh: list[Issue], ctx) -> list[dict]:
        return [{
            "component": "I", "kind": "issue", "body": issue.message,
            "meta": {"title": issue.message[:80] or issue.code, "code": issue.code,
                     "severity": issue.severity, "issue_type": _issue_type(issue.code),
                     "claim_id": issue.claim_id, "detail": issue.detail,
                     "impact": issue.impact, "analysis": issue.meta.get("analysis")},
        } for issue in fresh]


def _open_issue_objects(store) -> list[Issue]:
    from ..ranking import issues_from_records

    return issues_from_records(store.open_issues())


_TYPE_MAP = {
    "unproved_lemma": "proof-gap",
    "circular_reasoning": "logical-error",
    "logical_gap": "proof-gap",
    "logical_leap": "proof-gap",
    "hypothesis_not_verified": "external-ref",
    "unjustified_claim": "proof-gap",
    "incomplete_step": "proof-gap",
    "cited_blackbox_crux": "external-ref",
    "unchecked_finite_claim": "missing-case",
}


def _issue_type(code: str) -> str:
    return _TYPE_MAP.get((code or "").lower(), "proof-gap")
