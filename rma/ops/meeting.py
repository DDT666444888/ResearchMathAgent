"""Meeting — coordinated strategy, emitting (rho, a, DeltaH).

main.tex:378-380:

    "The meeting operation therefore convenes a coordinator and several
     field-appropriate mathematician personas. They deliberate over the current
     proof, unresolved issues, prior attempts, and retrieved literature, and the
     coordinator synthesizes a meeting record rho, an action plan a, and new
     strategic insights DeltaH."

The gap this closes: the existing meeting machinery produces rho and a, but
DeltaH — the strategic-insight delta — was never emitted. Here synthesis returns
all three: rho is written to M, a is written to M as a linked action plan
(so Revise can consume it), and each insight in DeltaH is written to H (so
UpdateConcepts can fold it in).
"""
from __future__ import annotations

from ..models import parse_json_response
from ..orchestrator import Query
from .base import Operation, register


@register
class MeetingOperation(Operation):
    """Run(Meeting, p, S, B) -> (rho, a, DeltaH)."""

    name = "meeting"
    include_proof = True

    def instructions(self, ctx) -> str:
        return (
            "Meeting: a coordinator and field mathematicians review the current "
            "proof, open issues, and literature, then synthesize. Return ONLY a "
            'JSON object:\n'
            '{"record": "<meeting transcript / summary rho>", '
            '"action_plan": {"summary": "...", "steps": ["...", ...]}, '
            '"insights": ["<strategic insight>", ...]}'
        )

    def query(self, ctx) -> Query:
        queue = ctx.extra.get("queue") or []
        links = [i.id for i in queue[:ctx.config.issue_budget] if getattr(i, "id", None)]
        return Query(
            text=f"Hold a strategy meeting for {ctx.store.problem_id}: agree on the "
                 f"highest-priority gaps and a concrete action plan.",
            id=f"{ctx.store.problem_id}-meet-r{ctx.round}",
            links=links,
        )

    def parse(self, raw, ctx) -> dict:
        data = raw if isinstance(raw, dict) else parse_json_response(
            raw if isinstance(raw, str) else "")
        if not isinstance(data, dict):
            return {"record": "", "action_plan": {}, "insights": []}
        plan = data.get("action_plan") or {}
        if isinstance(plan, str):
            plan = {"summary": plan, "steps": []}
        elif isinstance(plan, list):
            # a bare list of steps -> treat as the plan's steps (avoids a
            # .get-on-list crash when the model returns action_plan as an array)
            plan = {"summary": "", "steps": plan}
        elif not isinstance(plan, dict):
            plan = {"summary": str(plan), "steps": []}
        return {
            "record": str(data.get("record", "")),
            "action_plan": {"summary": str(plan.get("summary", "")),
                            "steps": list(plan.get("steps", []))},
            "insights": [str(x) for x in (data.get("insights") or []) if str(x).strip()],
        }

    def writeback(self, synth: dict, ctx) -> list[dict]:
        out: list[dict] = []
        record = synth.get("record", "")
        plan = synth.get("action_plan") or {}
        insights = synth.get("insights") or []

        # rho -> M
        if record:
            out.append({"component": "M", "kind": "meeting_record", "body": record,
                        "meta": {"round": ctx.round}})
        # a -> M (a separate, higher-priority record Revise consumes)
        if plan.get("summary") or plan.get("steps"):
            out.append({"component": "M", "kind": "action_plan",
                        "body": plan.get("summary", ""),
                        "priority": 55,   # a plan outranks a transcript
                        "meta": {"round": ctx.round, "steps": plan.get("steps", []),
                                 "is_action_plan": True}})
        # DeltaH -> H (one record per insight, so UpdateConcepts can fold each in).
        # The "insights" ablation drops H: the meeting still runs and produces
        # rho + a, but its strategic-insight delta is not retained.
        if not ctx.config.ablated("insights"):
            for insight in insights:
                out.append({"component": "H", "kind": "insight", "body": insight,
                            "meta": {"round": ctx.round, "from": "meeting"}})

        ctx.extra["action_plan"] = plan
        ctx.extra["insights"] = insights
        return out

    def run(self, ctx, *, invoke=None, telemetry_path=None):
        result = super().run(ctx, invoke=invoke, telemetry_path=telemetry_path)
        # Capture the store id of the action-plan record so Revise can link to
        # the exact plan it executes.
        for record in result.written:
            if record.meta.get("is_action_plan"):
                ctx.extra["action_plan_id"] = record.id
                break
        return result
