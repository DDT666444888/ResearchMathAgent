"""Round finalization — select the best round, not the last one.

`_refine_solution` overwrites the solution file on every round, so whatever the
final round produced is what ships. Refinement is not monotone: on a full First
Proof B2 run the delivered round was NOT the best round on 9 of 10 problems.
Two examples from that run:

    prob-04  round 3:  3 open issues, 10/10 claims proved
             round 5: 13 open issues,  7/9  claims proved   <- shipped
    prob-09  round 4:  4 open issues, completeness 8.0
             round 5: 13 open issues, completeness 5.0      <- shipped

Selecting the peak round instead cut total open issues across the ten problems
from 123 to 90 (-27%) at zero extra compute — every round is already on disk.

This is Algorithm 1's FinalizeRound(S, r) consolidation step: at the end of a
problem, decide which artifact the round budget actually bought.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RoundRecord:
    """One round's proof and the verification that judged it."""

    index: int                  # 1-based round number
    source: Path                # the .tex this round produced
    issues: int
    errors: int
    proved: int
    claims: int
    completeness: float | None
    passed: bool

    @property
    def score(self) -> tuple:
        """Sort key — lower is better.

        Ordered by what a reader of the proof would care about first:
        a passing proof, then fewest unresolved obligations, then the most
        claims actually proved, then the judge's completeness. Raw issue count
        alone would reward a proof for being short enough to have little to
        criticise, so proved-claim count breaks those ties.
        """
        return (0 if self.passed else 1,
                self.errors,
                self.issues,
                -self.proved,
                -(self.completeness if self.completeness is not None else -1))

    def as_dict(self) -> dict:
        return {"round": self.index, "source": str(self.source), "issues": self.issues,
                "errors": self.errors, "proved": self.proved, "claims": self.claims,
                "completeness": self.completeness, "passed": self.passed}


def collect_rounds(paths: dict) -> list[RoundRecord]:
    """Pair each verification report with the proof text it judged.

    Round 1 judged the proposal; round n>1 judged refinement n-1.
    """
    verifications = sorted(paths["verifications"].glob("verification_*.json"))
    proposals = sorted(paths["proposals"].glob("proposal_*.tex"))
    refinements = sorted(paths["refinements"].glob("refined_solution_*.tex"))
    sources = (proposals[:1] or []) + refinements

    rounds: list[RoundRecord] = []
    for i, report_path in enumerate(verifications):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if i >= len(sources):
            break
        issues = report.get("issues") or []
        dag = report.get("lemma_dag") or {}
        rounds.append(RoundRecord(
            index=i + 1,
            source=sources[i],
            issues=len(issues),
            errors=sum(1 for x in issues if x.get("severity") == "error"),
            proved=int(dag.get("proved") or 0),
            claims=int(dag.get("nodes") or 0),
            completeness=report.get("completeness_score"),
            passed=bool(report.get("passed")),
        ))
    return rounds


def select_best_round(rounds: list[RoundRecord]) -> RoundRecord | None:
    if not rounds:
        return None
    return min(rounds, key=lambda r: r.score)


def finalize_problem(paths: dict, *, apply: bool = True) -> dict | None:
    """Choose the best round and (optionally) make it the delivered solution.

    Returns a summary dict, or None when there is nothing to choose between.
    The losing rounds stay on disk untouched, so the decision is reversible and
    auditable.
    """
    rounds = collect_rounds(paths)
    if not rounds:
        return None
    best = select_best_round(rounds)
    last = rounds[-1]
    changed = False

    if apply and best is not None and best.index != last.index:
        try:
            text = best.source.read_text(encoding="utf-8", errors="replace")
            solution = paths["solution"]
            # Keep the round that would otherwise have shipped, so the swap can
            # be undone and compared.
            superseded = paths["artifacts"] / "superseded_last_round.tex"
            if solution.is_file():
                superseded.write_text(
                    solution.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            tmp = solution.with_suffix(".tex.tmp")
            tmp.write_text(text, encoding="utf-8")
            import os

            os.replace(tmp, solution)
            changed = True
        except OSError:
            changed = False

    summary = {
        "rounds": [r.as_dict() for r in rounds],
        "delivered_round": best.index if best else None,
        "last_round": last.index,
        "replaced_last_round": changed,
        "issues_last_round": last.issues,
        "issues_delivered": best.issues if best else None,
    }
    try:
        (paths["artifacts"] / "round_selection.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return summary
