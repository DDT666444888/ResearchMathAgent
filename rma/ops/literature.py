"""Literature — grounding the open issues in prior work.

main.tex:374-376:

    "the orchestrator invokes the Literature operation with the problem (p) and
     the current ranked issue set (Q). The operation searches for references
     relevant to the unresolved claims... For each useful source, it records the
     relevant theorem or technique, its assumptions, the claims it may support,
     and any conditions limiting its applicability."

So a literature record is not a bare citation: it carries the paper's four
fields — theorem_or_technique, assumptions, claims_supported, applicability_limits
— and links back to the issue ids that prompted the search, so the solver can
retrieve it next round as a linked record. The one regression that matters is
that L actually reaches the solver's context, which it never did before.
"""
from __future__ import annotations

from ..models import parse_json_response
from ..orchestrator import Query
from .base import Operation, register

# The four fields every source record must carry (main.tex:376).
REQUIRED_FIELDS = ("theorem_or_technique", "assumptions", "claims_supported",
                   "applicability_limits")


def validate_source(source: dict) -> bool:
    """A source is admissible only if it carries all four fields non-empty
    (claims_supported may be an empty list — a source can be background)."""
    if not isinstance(source, dict):
        return False
    for field in ("theorem_or_technique", "assumptions", "applicability_limits"):
        if not str(source.get(field, "")).strip():
            return False
    return "claims_supported" in source


@register
class LiteratureOperation(Operation):
    """Run(Literature, (p, Q), S, B) -> literature notes L keyed to Q."""

    name = "literature"
    include_proof = False   # literature is driven by the issue queue, not the proof

    def instructions(self, ctx) -> str:
        return (
            "Literature: for the unresolved claims in the query, find references "
            "(named theorems/techniques) that could close them. Return ONLY a JSON "
            'array; each element MUST carry all four keys:\n'
            '{"theorem_or_technique": "...", "assumptions": "...", '
            '"claims_supported": ["<issue id or claim>", ...], '
            '"applicability_limits": "...", "source": "<url or citation>"}'
        )

    def query(self, ctx) -> Query:
        queue = ctx.extra.get("queue") or []
        # top issues drive the search; link to them so the notes are retrievable
        top = queue[: ctx.config.issue_budget]
        lines = [f"- [{getattr(i, 'id', '?')}] {getattr(i, 'message', i)}" for i in top]
        links = [i.id for i in top if getattr(i, "id", None)]
        return Query(
            text="Find references for these open obligations:\n" + "\n".join(lines),
            id=f"{ctx.store.problem_id}-lit-r{ctx.round}",
            links=links,
        )

    def parse(self, raw, ctx) -> list[dict]:
        data = raw if isinstance(raw, list) else parse_json_response(
            raw if isinstance(raw, str) else "")
        if not isinstance(data, list):
            return []
        return [s for s in data if validate_source(s)]

    def writeback(self, sources: list[dict], ctx) -> list[dict]:
        queue = ctx.extra.get("queue") or []
        issue_ids = [i.id for i in queue if getattr(i, "id", None)]
        out = []
        for s in sources:
            supported = [str(c) for c in (s.get("claims_supported") or [])]
            # Link a note to the issues it names, else to the whole queried set.
            links = [c for c in supported if c in issue_ids] or issue_ids
            out.append({
                "component": "L", "kind": "literature_note",
                "body": f"{s['theorem_or_technique']} — {s['assumptions']}. "
                        f"Limits: {s['applicability_limits']}",
                "links": links,
                "meta": {"title": str(s.get("source") or s["theorem_or_technique"])[:80],
                         "url": s.get("source") if str(s.get("source", "")).startswith("http") else None,
                         "theorem_or_technique": s["theorem_or_technique"],
                         "assumptions": s["assumptions"],
                         "claims_supported": supported,
                         "applicability_limits": s["applicability_limits"]},
            })
        return out
