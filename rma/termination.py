"""Termination predicates — Solved(pi, e) and Stalled(S).

Algorithm 1 ends a round early when `Solved(pi,e) or Stalled(S)` (main.tex:390):

    "The procedure terminates when Solved(pi,e) determines that the current
     proof is complete and contains no unresolved correctness-critical issues,
     when Stalled(S) detects insufficient progress across recent rounds, or when
     the round budget N_R is exhausted."

`Stalled` did not exist anywhere in the pre-existing code, and its absence was
visible on the First Proof B2 run: prob-03 reached completeness 9.0 with 2 open
issues at round 2, then spent three more rounds drifting backwards. A stall
check would have stopped at round 2 — saving compute AND delivering the better
proof. These predicates are the fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Completeness at/above this counts the proof "complete" for Solved. Matches the
# completeness gate's default (RMA_COMPLETENESS_MIN).
DEFAULT_COMPLETENESS_MIN = 8.0
CRITICAL_SEVERITIES = ("P0", "P1")


@dataclass
class RoundMetrics:
    """The signals a round leaves behind, consumed by the predicates."""

    round: int
    completeness: float | None
    proved_terminal_fraction: float
    open_critical: int          # open P0/P1 issues after the round
    closed_critical: int        # P0/P1 issues closed during the round
    open_total: int = 0
    extra: dict = field(default_factory=dict)


def solved(metrics: RoundMetrics, *, completeness_min: float = DEFAULT_COMPLETENESS_MIN) -> bool:
    """Solved(pi, e): complete, correct, and self-contained.

    All three must hold: the evaluator judges it complete, no correctness-
    critical (P0/P1) issue is open, and every terminal claim is proved (the
    argument bottoms out in nothing unproven).
    """
    if metrics.open_critical > 0:
        return False
    if metrics.proved_terminal_fraction < 1.0:
        return False
    if metrics.completeness is None:
        return False
    return metrics.completeness >= completeness_min


def stalled(history: list[RoundMetrics], *, window: int = 2, eps: float = 0.5) -> bool:
    """Stalled(S): no meaningful progress across the last `window` rounds.

    Fires only when ALL of these hold over the window: no P0/P1 issue was
    closed, completeness did not rise by more than `eps`, and the proved-
    terminal fraction did not increase. Any one of those improving means the
    system is still making progress, so it keeps going.

    Requires at least `window`+1 rounds of history, so it never fires before
    the system has had a chance to move.
    """
    if len(history) < window + 1:
        return False
    recent = history[-(window + 1):]

    # progress signal 1: the critical-issue backlog actually SHRANK across the
    # window. Net count is what matters, not gross closes: "closed one while the
    # critic opened six" is not progress — it is how the loop churns forever on a
    # proof whose gap backlog only grows (observed on q6: open_critical
    # 36 -> 41 -> 47 while completeness sat flat at 9). When the backlog is not
    # shrinking and nothing else is improving, the loop has stalled; it should
    # stop and deliver the best round rather than burn the remaining budget.
    opens = [m.open_critical for m in recent]
    if opens[-1] < opens[0]:
        return False

    # progress signal 2: completeness rose by more than eps across the window
    comps = [m.completeness for m in recent if m.completeness is not None]
    if len(comps) >= 2 and (comps[-1] - comps[0]) > eps:
        return False

    # progress signal 3: proved-terminal fraction increased across the window
    fracs = [m.proved_terminal_fraction for m in recent]
    if fracs[-1] > fracs[0] + 1e-9:
        return False

    return True


def stop_reason(metrics: RoundMetrics, history: list[RoundMetrics], round_idx: int,
                n_rounds: int, *, completeness_min: float = DEFAULT_COMPLETENESS_MIN) -> str | None:
    """Which terminal condition (if any) ends the loop now."""
    if solved(metrics, completeness_min=completeness_min):
        return "solved"
    if stalled(history):
        return "stalled"
    if round_idx >= n_rounds - 1:
        return "budget_exhausted"
    return None
