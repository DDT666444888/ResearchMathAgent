"""Revise — a coordinated proof rewrite from the meeting's action plan.

main.tex:380-381:

    "The revision operation then receives the action plan together with the
     relevant proof state and performs a coordinated update of the working proof
     (Alg. line revise). This allows proof development to advance both through
     localized repairs and through broader strategic revisions."

This operation does not exist anywhere in the pre-existing code — nothing
consumed an action plan to edit the proof; `run_step_execution` was defined and
orphaned. Revise is deliberately the coordinated counterpart to the Solver's
localized patch: it may restructure across the whole proof, and its diff is
expected to be larger than a Solver revision's (the round loop can measure the
distinction). It writes a new Pi revision whose parent is the current proof and
whose provenance names the action plan it executed.
"""
from __future__ import annotations

from ..orchestrator import Query
from .base import Operation, register


@register
class ReviseOperation(Operation):
    """Run(Revise, (p, a), S, B) -> a new Pi revision executing plan a."""

    name = "revise"
    expects = "latex"       # revise returns a proof document, not JSON
    include_proof = True

    def instructions(self, ctx) -> str:
        plan = ctx.extra.get("action_plan") or {}
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.get("steps", [])))
        return (
            "Revise: carry out the action plan below as a coordinated update of "
            "the proof. You may restructure across the document, but keep every "
            "result that is already correct. Return ONLY the complete LaTeX "
            "document.\n\nACTION PLAN: " + str(plan.get("summary", "")) +
            ("\nSTEPS:\n" + steps if steps else "")
        )

    def query(self, ctx) -> Query:
        plan_id = ctx.extra.get("action_plan_id")
        return Query(
            text="Execute the meeting action plan against the current proof.",
            id=f"{ctx.store.problem_id}-revise-r{ctx.round}",
            links=[plan_id] if plan_id else [],
        )

    def parse(self, raw, ctx) -> str:
        from ..models import clean_latex_reply
        if isinstance(raw, str):
            return clean_latex_reply(raw)
        if isinstance(raw, dict):
            return clean_latex_reply(str(raw.get("tex", "")))
        return ""

    def writeback(self, tex: str, ctx) -> list[dict]:
        if not tex:
            return []
        proof = ctx.store.current_proof()
        return [{
            "component": "Pi", "kind": "proof_revision", "body": tex,
            "links": [ctx.extra["action_plan_id"]] if ctx.extra.get("action_plan_id") else [],
            "meta": {"produced_by": "revise", "mode": "coordinated_revision",
                     "parent_id": proof.id if proof else None,
                     "action_plan_id": ctx.extra.get("action_plan_id")},
        }]
