"""UpdateConcepts — incremental glossary update from the proof and DeltaH.

main.tex:382-384:

    "The concept component K maintains a structured glossary of the mathematical
     objects used in the proof... It is initialized from the problem statement
     and updated after each revision using the current proof and meeting
     insights."

The pre-existing `generate_concepts` ran once, from the problem statement, only
when the glossary was absent. Here UpdateConcepts is incremental: after each
revision it adds glossary entries for objects newly introduced in the proof and
folds in the round's insight delta DeltaH, without duplicating entries that
already exist.
"""
from __future__ import annotations

from ..models import parse_json_response
from ..orchestrator import Query
from .base import Operation, register


@register
class ConceptsOperation(Operation):
    """UpdateConcepts(S, pi, DeltaH) -> new glossary entries in K."""

    name = "concepts"
    include_proof = True

    def instructions(self, ctx) -> str:
        existing = ctx.extra.get("existing_concept_names") or []
        known = (" Already defined (do not repeat): " + ", ".join(existing[:40])) if existing else ""
        return (
            "UpdateConcepts: list the mathematical objects the current proof "
            "introduces that a reader would need defined. Return ONLY a JSON "
            'array; each element:\n'
            '{"name": "...", "definition": "...", "assumptions": "", "notation": "", '
            '"source": ""}.' + known
        )

    def query(self, ctx) -> Query:
        insights = ctx.extra.get("insights") or []
        note = ("\nRecent strategic insights to reflect: " + "; ".join(insights[:5])) if insights else ""
        return Query(text=f"Update the concept glossary for {ctx.store.problem_id}.{note}",
                     id=f"{ctx.store.problem_id}-concepts-r{ctx.round}")

    def run(self, ctx, *, invoke=None, telemetry_path=None):
        # Seed the "already defined" set so the model does not repeat entries;
        # reading the glossary is a write-back concern (dedup), not context.
        ctx.extra["existing_concept_names"] = [
            r.meta.get("name") for r in ctx.store.concepts if r.meta.get("name")]
        return super().run(ctx, invoke=invoke, telemetry_path=telemetry_path)

    def parse(self, raw, ctx) -> list[dict]:
        data = raw if isinstance(raw, list) else parse_json_response(
            raw if isinstance(raw, str) else "")
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if isinstance(item, dict) and str(item.get("name", "")).strip():
                out.append(item)
        return out

    def writeback(self, concepts: list[dict], ctx) -> list[dict]:
        existing = {(r.meta.get("name") or "").strip().lower() for r in ctx.store.concepts}
        out = []
        for c in concepts:
            name = str(c["name"]).strip()
            if name.lower() in existing:
                continue
            existing.add(name.lower())
            out.append({
                "component": "K", "kind": "concept",
                "body": str(c.get("definition", "")),
                "meta": {"name": name, "assumptions": c.get("assumptions", ""),
                         "notation": c.get("notation", ""), "source": c.get("source", ""),
                         "round": ctx.round},
            })
        return out
