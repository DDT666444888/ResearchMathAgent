"""Issue severity and the ranked queue Q.

The paper (main.tex:371):

    "Each issue receives a base severity during discovery: P0 for a defect that
     invalidates the main conclusion or a necessary crux claim, P1 for a gap
     that blocks a required lemma or proof branch, P2 for a localized missing
     justification, citation, or verification, and P3 for a notation or
     presentation defect that does not affect correctness. The critic orders
     open issues first by severity and then, among issues at the same level, by
     dependency impact, measured by the number of unresolved downstream claims
     that depend on the affected claim."

Three orderings exist because the paper reports them as a ladder
(FIFO 3.8 -> severity 5.0 -> severity+impact 6.0, main.tex:930-934); here they
are one flag rather than three code paths.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .claims import ClaimGraph

PRIORITIES = ("P0", "P1", "P2", "P3")
_RANK = {p: i for i, p in enumerate(PRIORITIES)}

ORDER_MODES = ("fifo", "severity", "severity+impact")

# Issue codes that are correctness-critical wherever they land.
_ALWAYS_P0 = {"circular_reasoning", "false_statement", "counterexample"}
# Codes that never affect correctness.
_PRESENTATION = {"notation", "typo", "presentation", "clarity", "formatting",
                 "unclear_def", "unclear-def"}
# Codes that are localized by nature.
_LOCALIZED = {"missing_citation", "missing_citations", "unresolved_citations",
              "unchecked_finite_claim", "external_ref", "external-ref"}

_NOTATION_TEXT = re.compile(
    r"\b(notation|typo|typographic|wording|phrasing|formatting|readab)", re.IGNORECASE)
# Codes that name a substantive proof gap: the free-text notation heuristic must
# never downgrade these to P3, however their message happens to be phrased.
_SUBSTANTIVE = {"unproved_lemma", "cited_blackbox_crux", "logical_gap", "logical_leap",
                "hypothesis_not_verified", "incomplete_step", "proof_gap",
                "circular_reasoning", "false_statement", "counterexample",
                "unchecked_finite_claim"}


@dataclass
class Issue:
    """One entry of the queue Q."""

    id: str
    code: str
    message: str
    severity: str = "P2"
    claim_id: str | None = None
    detail: str = ""
    created_at: str = ""
    status: str = "open"
    impact: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return _RANK.get(self.severity, _RANK["P2"])

    def as_dict(self) -> dict:
        return {"id": self.id, "code": self.code, "message": self.message,
                "severity": self.severity, "claim_id": self.claim_id,
                "detail": self.detail, "impact": self.impact,
                "status": self.status}


def assign_severity(issue: Issue, graph: ClaimGraph | None = None) -> str:
    """P0-P3 for one issue, per the paper's definitions.

    Deterministic structure first: whether the main conclusion depends on the
    affected claim is a graph question, not a judgement call. An LLM tie-break
    belongs only where the graph is silent (T4.2).
    """
    code = (issue.code or "").lower().replace("-", "_")

    if code in _ALWAYS_P0:
        return "P0"
    # A presentation-typed defect is P3. But the free-text "notation/wording"
    # heuristic must NOT fire on a substantive defect: a logical_gap whose
    # message says "the wording hides a gap that breaks the main theorem" is a
    # P0, not a P3. So the text heuristic only applies when the code itself is
    # not a substantive-gap code.
    if code in _PRESENTATION:
        return "P3"
    if code not in _SUBSTANTIVE and _NOTATION_TEXT.search(issue.message or ""):
        return "P3"
    # The DEFECT TYPE caps severity before position is considered: the paper
    # puts "a localized missing justification, citation, or verification" at P2
    # regardless of where it sits. Otherwise a missing citation on a claim the
    # main theorem happens to use would outrank an unproved crux lemma.
    if code in _LOCALIZED:
        return "P2"

    if graph is not None and issue.claim_id:
        claim = graph.node(issue.claim_id)
        if claim is not None:
            # Circular reasoning through this claim invalidates the conclusion.
            if any(issue.claim_id in cycle for cycle in graph.cycles()):
                return "P0"
            # P0 = "invalidates the main conclusion or a necessary crux claim".
            # Note this asks about the HEADLINE result, not merely about being
            # a root: every dependency chain has a root, including an orphan
            # lemma nobody uses, so root-ness alone cannot separate P0 from P1.
            if graph.supports_main_result(issue.claim_id):
                return "P0"
            # P1 = "blocks a required lemma or proof branch": something rests
            # on it, but the headline result does not.
            if graph.dependents_of(issue.claim_id):
                return "P1"
            # A leaf nothing rests on yet.
            return "P2"
    # Unanchored gaps default to "blocks a required lemma": conservative, since
    # under-rating a real gap is worse than over-rating a cosmetic one.
    if code in {"unproved_lemma", "cited_blackbox_crux", "logical_gap",
                "hypothesis_not_verified", "incomplete_step", "proof_gap"}:
        return "P1"
    return "P2"


def compute_impact(issue: Issue, graph: ClaimGraph | None) -> int:
    """Dependency impact: unresolved claims that transitively rest on this one."""
    if graph is None or not issue.claim_id:
        return 0
    if graph.node(issue.claim_id) is None:
        return 0
    return graph.unresolved_downstream(issue.claim_id)


def rank(issues: list[Issue], graph: ClaimGraph | None = None,
         mode: str = "severity+impact", *, assign: bool = True) -> list[Issue]:
    """Order the open issues into the queue Q.

    mode:
      "fifo"             discovery order (the paper's weakest variant, 3.8)
      "severity"         severity only (5.0)
      "severity+impact"  severity, then dependency impact (6.0, default)
    """
    if mode not in ORDER_MODES:
        raise ValueError(f"unknown order mode: {mode}; expected one of {ORDER_MODES}")

    live = [i for i in issues if i.status != "resolved"]
    if assign:
        for issue in live:
            issue.severity = assign_severity(issue, graph)
            issue.impact = compute_impact(issue, graph)

    if mode == "fifo":
        return list(live)
    if mode == "severity":
        # Stable sort: within a severity, discovery order is preserved, so this
        # is genuinely "severity only" and not impact ordering by accident.
        return sorted(live, key=lambda i: i.rank)
    return sorted(live, key=lambda i: (i.rank, -i.impact))


def select(issues: list[Issue], budget: int, graph: ClaimGraph | None = None,
           mode: str = "severity+impact") -> list[Issue]:
    """Q[1:b] — the top-b issues this round will repair."""
    return rank(issues, graph, mode)[:max(0, budget)]


def issues_from_records(records) -> list[Issue]:
    """Build the queue from store I-component records."""
    out = []
    for record in records:
        meta = record.meta or {}
        out.append(Issue(
            id=record.id,
            code=str(meta.get("code") or meta.get("issue_type") or "proof_gap"),
            message=record.body,
            severity=str(meta.get("severity") or "P2"),
            claim_id=meta.get("claim_id"),
            detail=str(meta.get("detail", "")),
            created_at=record.created_at,
            status=str(meta.get("status", "open")),
            meta=meta,
        ))
    return out


def issues_from_verifier(raw: list[dict], prefix: str = "v") -> list[Issue]:
    """Build the queue from verifier-style {code,severity,message,detail} dicts."""
    out = []
    for i, item in enumerate(raw or []):
        out.append(Issue(
            id=f"{prefix}{i}",
            code=str(item.get("code", "proof_gap")),
            message=str(item.get("message", "")),
            severity=str(item.get("priority") or "P2"),
            claim_id=item.get("claim_id"),
            detail=str(item.get("detail", "")),
            meta=dict(item),
        ))
    return out
