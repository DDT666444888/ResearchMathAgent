r"""The Algorithm 1 round loop — solve_problem(p, cfg).

This transcribes Algorithm 1 (main.tex:197-347) directly:

    S <- InitializeStore(p)
    for r in 0 .. N_R-1:
        pi <- CurrentProof(S)
        (Q,S) <- Run(Critic, p, S, B)
        for iota in Q[1:b]:
            (_,S) <- Run(Solver, (p,iota), S, B)
        (_,S)        <- Run(Literature, (p,Q), S, B)
        ((rho,a,dH),S) <- Run(Meeting, p, S, B)
        (pi,S)       <- Run(Revise, (p,a), S, B)
        S            <- UpdateConcepts(S, pi, dH)
        (e,S)        <- Run(Evaluator, (p,pi), S, B)
        S            <- FinalizeRound(S, r)
        if Solved(pi,e) or Stalled(S): break
    return pi

The operations are the seven in rma/ops; the store is S; the budget B and round
count N_R come from RunConfig.

── Paper fidelity ────────────────────────────────────────────────────────────
`RunConfig.paper_faithful=True` runs Algorithm 1 EXACTLY as written above:
`pi <- CurrentProof(S)` is the last revision each round, and the run outputs the
final `pi` (\KwOut{proof pi}).

By DEFAULT (paper_faithful=False) we add two enhancements that are NOT in the
paper's pseudocode, each marked "ENHANCEMENT" at its site:

  * push_forward_best — each round refines the BEST proof so far, not the last,
    so a non-monotone (regressing) round cannot compound into the next.
  * best-of-rounds delivery — the delivered proof is the peak round, not
    whatever ran last.

Two further implementation details are faithful to the paper's *intent* but are
not literal pseudocode, and apply in BOTH modes: `_safe_op` keeps one failing
operation from crashing the whole run (a crash is a bug, not a semantics), and
Solved/Stalled are concrete implementations of the paper's qualitative
predicates (termination.py). Everything else — the seven operations, their
order, PrefixToBudget, and the Solved/Stalled/budget termination — matches the
paper in both modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import claims as _claims
from .config import RunConfig
from .ops import get_operation
from .ops.base import OpContext
from .store import ResearchStore
from .termination import RoundMetrics, stop_reason

# The unit sequence one round runs, in the paper's order. Recorded per round so
# the order is testable against Algorithm 1.
ROUND_UNIT_ORDER = ("critic", "solver", "literature", "meeting", "revise",
                    "update_concepts", "evaluator", "finalize")


@dataclass
class RoundResult:
    round: int
    units: list[str] = field(default_factory=list)
    metrics: RoundMetrics | None = None
    issues_opened: int = 0
    issues_closed: int = 0
    stop: str | None = None
    proof_id: str | None = None       # the Pi revision current at the end of this round


@dataclass
class SolveResult:
    problem_id: str
    rounds: list[RoundResult] = field(default_factory=list)
    stop_reason: str | None = None
    final_proof_id: str | None = None      # the last round's proof
    delivered_round: int | None = None     # the round whose proof is delivered
    delivered_proof_id: str | None = None  # best-of-rounds selection


def _round_score(rr: RoundResult) -> tuple:
    """Best-of-rounds key (lower is better), mirroring finalize.RoundRecord:
    the round that terminated solved wins, then fewest open critical issues,
    then the most proved terminal claims, then the highest completeness.
    Refinement is not monotone, so the last round is often not the best."""
    m = rr.metrics
    if m is None:
        return (1, 1_000, 0, 0.0)
    return (0 if rr.stop == "solved" else 1,
            m.open_critical,
            -m.proved_terminal_fraction,
            -(m.completeness if m.completeness is not None else -1))


def _safe_op(store: ResearchStore, name: str, round_idx: int, fn):
    """Run one operation, but never let its failure crash the whole run.

    A multi-round research loop makes many stochastic model calls; any one can
    fail transiently (a narration-only reply, a timeout, a malformed artifact).
    Letting that exception propagate discards every round of accumulated store
    progress. Instead we record the failure (as an `op_error`, tagged unit
    `round_loop` so it is not miscounted as the failed module's useful output)
    and continue with whatever the persistent store already holds."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - resilience boundary for the loop
        try:
            store.add("H", "op_error",
                      f"[op-error] {name} failed in round {round_idx}: "
                      f"{type(e).__name__}: {e}",
                      meta={"unit": "round_loop", "error": True,
                            "failed_op": name, "round": round_idx},
                      round=round_idx)
        except Exception:
            pass
        return None


def _critical_counts(store: ResearchStore) -> tuple[int, int]:
    """(open P0/P1, resolved P0/P1) from the issue store."""
    open_c = res_c = 0
    for issue in store.issues:
        sev = issue.meta.get("severity", "P2")
        if sev not in ("P0", "P1"):
            continue
        if issue.meta.get("status") == "resolved":
            res_c += 1
        else:
            open_c += 1
    return open_c, res_c


def solve_problem(
    store: ResearchStore,
    problem: dict,
    cfg: RunConfig | None = None,
    *,
    backend=None,
    critic_analyses: dict | None = None,
    telemetry_path=None,
) -> SolveResult:
    """Run Algorithm 1 for one problem over the store S.

    ``backend(unit, observation) -> reply`` is the only seam to a model; an
    offline test passes a FakeBackend. ``critic_analyses`` injects the critic's
    three analyses (the critic does not use ``backend`` — its analyses are its
    invoke). Both default to the real implementations.
    """
    cfg = cfg or RunConfig()
    result = SolveResult(problem_id=store.problem_id)
    history: list[RoundMetrics] = []
    prev_resolved_critical = 0
    best_rr: RoundResult | None = None
    best_proof_body: str | None = None

    for r in range(cfg.n_rounds):
        store.begin_round(r)

        # ENHANCEMENT (not in the paper; default on, disabled by paper_faithful).
        # Push-forward-from-best: start this round from the BEST proof so far, not
        # a proof a previous round regressed. Refinement is not monotone, so
        # without this a bad round compounds into the next and the loop goes
        # backwards (the diagnostic q6 run: open_critical 15->20->24, clarity
        # 7->4). Re-establishing the best proof as current makes the best score
        # monotone non-decreasing — later rounds can only help. In paper-faithful
        # mode push_forward_best is False, so pi = CurrentProof(S) = last revision.
        if (getattr(cfg, "push_forward_best", True) and best_proof_body is not None):
            cur = store.current_proof()
            if cur is None or cur.body != best_proof_body:
                store.add_proof_revision(
                    best_proof_body, produced_by="carry_forward", round=r,
                    meta={"carried_from_round": best_rr.round if best_rr else None,
                          "reason": "push_forward_from_best"})

        ctx = OpContext(store=store, config=cfg, problem=problem, round=r)
        rr = RoundResult(round=r)

        n_issues_before = len(store.issues)

        # Each op is run through _safe_op: a single operation failing (a transient
        # model error, a narration-only reply) must not crash the whole run and
        # discard every round of accumulated store progress. A failed op is
        # recorded and the round continues with whatever the store already holds.
        def _run(name, **kw):
            _safe_op(store, name, r, lambda: get_operation(name).run(
                ctx, telemetry_path=telemetry_path, **kw))
            rr.units.append(name)

        # (Q,S) <- Run(Critic, ...)
        _safe_op(store, "critic", r, lambda: get_operation("critic").run(
            ctx, analyses=critic_analyses, telemetry_path=telemetry_path))
        rr.units.append("critic")
        queue = ctx.extra.get("queue") or []

        # for iota in Q[:b]: Run(Solver, (p,iota), ...)
        for issue in queue[: cfg.issue_budget]:
            _safe_op(store, "solver", r, lambda issue=issue: get_operation("solver").run(
                ctx, issue=_issue_dict(issue), invoke=backend, telemetry_path=telemetry_path))
            rr.units.append("solver")

        # Run(Literature, (p,Q), ...)  — skip only if literature is ablated
        if not cfg.ablated("literature"):
            _run("literature", invoke=backend)

        # Run(Meeting, ...) -> (rho, a, dH) — skip if meeting ablated
        if not cfg.ablated("meeting"):
            _run("meeting", invoke=backend)
            # Run(Revise, (p,a), ...)
            _run("revise", invoke=backend)

        # UpdateConcepts(S, pi, dH) — skip if concepts ablated
        if not cfg.ablated("concepts"):
            _safe_op(store, "concepts", r, lambda: get_operation("concepts").run(
                ctx, invoke=backend, telemetry_path=telemetry_path))
            rr.units.append("update_concepts")

        # (e,S) <- Run(Evaluator, (p,pi), ...)
        _run("evaluator", invoke=backend)

        # FinalizeRound(S, r)
        metrics = _finalize_round(store, ctx, r, prev_resolved_critical)
        rr.units.append("finalize")
        rr.metrics = metrics
        rr.issues_opened = max(0, len(store.issues) - n_issues_before)
        _end_proof = store.current_proof()
        rr.proof_id = _end_proof.id if _end_proof else None
        history.append(metrics)

        # if Solved(pi,e) or Stalled(S): break
        reason = stop_reason(metrics, history, r, cfg.n_rounds,
                             completeness_min=_completeness_min(cfg))
        rr.stop = reason
        result.rounds.append(rr)

        # Update the best-so-far (same key as best-of-rounds delivery). This is
        # what the NEXT round carries forward, so the best proof never regresses.
        if best_rr is None or _round_score(rr) < _round_score(best_rr):
            best_rr = rr
            _bp = store.current_proof()
            best_proof_body = _bp.body if _bp else best_proof_body

        prev_resolved_critical = _critical_counts(store)[1]
        if reason in ("solved", "stalled"):
            result.stop_reason = reason
            break
        if reason == "budget_exhausted":
            result.stop_reason = "budget_exhausted"

    result.stop_reason = result.stop_reason or "budget_exhausted"

    proof = store.current_proof()
    result.final_proof_id = proof.id if proof else None

    if getattr(cfg, "paper_faithful", False):
        # Algorithm 1 outputs pi = CurrentProof(S) — the final proof (main.tex
        # \KwOut{proof pi}; there is no best-of-rounds selection in the paper).
        # Deliver exactly that in paper-faithful mode.
        if result.rounds:
            last = result.rounds[-1]
            result.delivered_round = last.round
            result.delivered_proof_id = last.proof_id or result.final_proof_id
    elif result.rounds:
        # ENHANCEMENT (not in the paper): best-of-rounds delivery. Refinement is
        # not monotone, so pick the peak round rather than whatever ran last.
        best = min(result.rounds, key=_round_score)
        result.delivered_round = best.round
        result.delivered_proof_id = best.proof_id or result.final_proof_id
    return result


def _finalize_round(store, ctx, round_idx, prev_resolved_critical) -> RoundMetrics:
    """Consolidate the round: compute metrics and write a round-summary record.

    This is FinalizeRound(S, r). Best-of-rounds *delivery* selection
    (rma.finalize) runs once at the end of the pipeline over the output tree; the
    store-level summary here is what Stalled/Solved read.
    """
    proof = store.current_proof()
    graph = _claims.parse_claims(proof.body if proof else "")
    open_crit, resolved_crit = _critical_counts(store)
    # Completeness must describe the SAME proof state as proved_terminal_fraction
    # and open_critical, which are read from the post-revision current proof. The
    # evaluator scores that post-revision proof; the critic's completeness_score
    # is from the START of the round (pre-solver, pre-revise), so it would judge
    # a different proof. Prefer the evaluator; fall back to the critic only when
    # the evaluator produced nothing.
    ev = ctx.extra.get("evaluation") or {}
    completeness = ev.get("proof_completeness")
    if completeness is None:
        completeness = ctx.extra.get("completeness_score")

    metrics = RoundMetrics(
        round=round_idx,
        completeness=float(completeness) if completeness is not None else None,
        proved_terminal_fraction=graph.proved_terminal_fraction(),
        open_critical=open_crit,
        closed_critical=max(0, resolved_crit - prev_resolved_critical),
        open_total=len(store.open_issues()),
    )
    store.add("E", "round_summary",
              f"round {round_idx}: completeness={metrics.completeness} "
              f"terminal_frac={metrics.proved_terminal_fraction} "
              f"open_critical={open_crit}",
              meta={"round": round_idx, "kind_detail": "round_summary",
                    "completeness": metrics.completeness,
                    "proved_terminal_fraction": metrics.proved_terminal_fraction,
                    "open_critical": open_crit, "closed_critical": metrics.closed_critical,
                    "open_total": metrics.open_total},
              round=round_idx)
    return metrics


def _issue_dict(issue) -> dict:
    return {"id": getattr(issue, "id", None), "code": getattr(issue, "code", ""),
            "message": getattr(issue, "message", ""), "severity": getattr(issue, "severity", "P2"),
            "claim_id": getattr(issue, "claim_id", None), "detail": getattr(issue, "detail", "")}


def _completeness_min(cfg) -> float:
    import os

    from .termination import DEFAULT_COMPLETENESS_MIN

    return float(os.environ.get("RMA_COMPLETENESS_MIN", DEFAULT_COMPLETENESS_MIN))
