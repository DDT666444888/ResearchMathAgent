"""Evaluator — score the current proof and append the record to E.

main.tex:386-388:

    "The evaluator operation assesses the current proof on answer accuracy,
     logical correctness, completeness, and clarity. Its output contains both
     scores and a written assessment and is appended to the evaluation
     component E."

Two things this fixes relative to the pre-existing `proof_eval.evaluate_proof`:
it scores THIS round's proof pi (not the globally best archived proof, which is
what the old single-slot file held), and it appends to an E series bound to the
proof version it judged, so progress is a trajectory rather than one overwritten
value. Content-hash reuse and median-of-N self-consistency are kept.
"""
from __future__ import annotations

from ..models import parse_json_response
from ..orchestrator import Query
from .base import Operation, register

DIMENSIONS = ("answer_accuracy", "logical_correctness", "proof_completeness", "proof_clarity")


def _proof_hash(problem_text: str, proof_text: str) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(problem_text.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(proof_text.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


@register
class EvaluatorOperation(Operation):
    """Run(Evaluator, (p, pi), S, B) -> evaluation record e appended to E."""

    name = "evaluator"
    include_proof = True

    def instructions(self, ctx) -> str:
        return (
            "Evaluator: score the current proof on four dimensions and give a "
            "one-sentence verdict. Return ONLY JSON:\n"
            '{"answer_accuracy": <0 or 1>, "logical_correctness": <0-10>, '
            '"proof_completeness": <0-10>, "proof_clarity": <0-10>, '
            '"verdict": "<one sentence>"}'
        )

    def query(self, ctx) -> Query:
        return Query(text=f"Evaluate the current proof of {ctx.store.problem_id}.",
                     id=f"{ctx.store.problem_id}-eval-r{ctx.round}")

    def parse(self, raw, ctx) -> dict:
        data = raw if isinstance(raw, dict) else parse_json_response(
            raw if isinstance(raw, str) else "")
        if not isinstance(data, dict):
            return {}
        scores = {}
        for dim in DIMENSIONS:
            if dim in data:
                try:
                    scores[dim] = int(round(float(data[dim])))
                except Exception:
                    scores[dim] = None
        scores["verdict"] = str(data.get("verdict", ""))
        return scores

    def writeback(self, scores: dict, ctx) -> list[dict]:
        if not scores:
            return []
        proof = ctx.store.current_proof()
        proof_hash = _proof_hash(ctx.problem_text, proof.body if proof else "")

        # Content-hash reuse: if the last evaluation judged this exact proof,
        # do not append a duplicate (mirrors proof_eval.reuse_if_unchanged).
        for prev in reversed(ctx.store.evaluations):
            if prev.meta.get("proof_hash") == proof_hash:
                ctx.extra["evaluation"] = prev.meta.get("scores", {})
                return []

        total = sum(v for k, v in scores.items()
                    if k in DIMENSIONS and isinstance(v, int))
        ctx.extra["evaluation"] = scores
        return [{
            "component": "E", "kind": "evaluation",
            "body": scores.get("verdict", "") or f"completeness {scores.get('proof_completeness')}",
            "links": [proof.id] if proof else [],
            "meta": {"round": ctx.round, "proof_hash": proof_hash,
                     "scores": scores, "total": total,
                     "proof_id": proof.id if proof else None},
        }]
