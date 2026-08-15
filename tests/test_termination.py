"""T5.3 — Solved(pi, e) and Stalled(S)."""
from __future__ import annotations

import unittest

from rma.termination import RoundMetrics, solved, stalled, stop_reason


def _m(round, comp, frac, open_crit, closed_crit=0):
    return RoundMetrics(round=round, completeness=comp, proved_terminal_fraction=frac,
                        open_critical=open_crit, closed_critical=closed_crit)


class SolvedTest(unittest.TestCase):
    def test_solved_when_complete_correct_selfcontained(self) -> None:
        self.assertTrue(solved(_m(3, 9.0, 1.0, 0)))

    def test_not_solved_when_p0_open(self) -> None:
        self.assertFalse(solved(_m(3, 9.0, 1.0, 1)))

    def test_not_solved_when_terminal_fraction_below_one(self) -> None:
        self.assertFalse(solved(_m(3, 9.0, 0.8, 0)))

    def test_not_solved_when_completeness_below_threshold(self) -> None:
        self.assertFalse(solved(_m(3, 6.0, 1.0, 0)))

    def test_not_solved_when_completeness_unknown(self) -> None:
        self.assertFalse(solved(_m(3, None, 1.0, 0)))

    def test_threshold_configurable(self) -> None:
        self.assertTrue(solved(_m(3, 6.0, 1.0, 0), completeness_min=6.0))


class StalledTest(unittest.TestCase):
    def test_not_stalled_before_enough_history(self) -> None:
        self.assertFalse(stalled([_m(0, 3.0, 0.5, 2)]))
        self.assertFalse(stalled([_m(0, 3.0, 0.5, 2), _m(1, 3.0, 0.5, 2)]))

    def test_stalled_after_two_flat_rounds(self) -> None:
        hist = [_m(0, 3.0, 0.5, 3), _m(1, 3.0, 0.5, 3), _m(2, 3.0, 0.5, 3)]
        self.assertTrue(stalled(hist))

    def test_not_stalled_when_a_critical_issue_closed(self) -> None:
        hist = [_m(0, 3.0, 0.5, 3), _m(1, 3.0, 0.5, 2, closed_crit=1), _m(2, 3.0, 0.5, 2)]
        self.assertFalse(stalled(hist))

    def test_not_stalled_while_completeness_rises(self) -> None:
        hist = [_m(0, 3.0, 0.5, 3), _m(1, 4.0, 0.5, 3), _m(2, 5.0, 0.5, 3)]
        self.assertFalse(stalled(hist))

    def test_not_stalled_while_terminal_fraction_rises(self) -> None:
        hist = [_m(0, 3.0, 0.5, 3), _m(1, 3.0, 0.6, 3), _m(2, 3.0, 0.7, 3)]
        self.assertFalse(stalled(hist))

    def test_growing_backlog_with_flat_quality_is_stalled(self) -> None:
        """The live q6 shape: the critic opens issues faster than the solver
        closes them, so the backlog GROWS while completeness sits flat. Closing
        a few each round must not count as progress — the loop should stall and
        deliver the best round instead of churning to budget exhaustion."""
        hist = [_m(0, 9.0, 1.0, 36, closed_crit=2),
                _m(1, 9.0, 1.0, 41, closed_crit=1),
                _m(2, 9.0, 1.0, 47, closed_crit=1)]
        self.assertTrue(stalled(hist))

    def test_net_backlog_shrinking_is_not_stalled(self) -> None:
        # backlog trending down across the window is real progress
        hist = [_m(0, 9.0, 1.0, 47), _m(1, 9.0, 1.0, 41), _m(2, 9.0, 1.0, 36)]
        self.assertFalse(stalled(hist))

    def test_the_b2_prob03_shape_would_have_stopped(self) -> None:
        """prob-03: near-solved at round 2, then three flat rounds. A stall
        check should fire once the plateau is two rounds deep."""
        hist = [_m(0, 5.0, 0.7, 1), _m(1, 9.0, 1.0, 0, closed_crit=1),  # the good round
                _m(2, 9.0, 1.0, 0), _m(3, 9.0, 1.0, 0)]
        # by round 3, two flat rounds since the last improvement -> stalled
        self.assertTrue(stalled(hist))


class JudgeDisagreementTest(unittest.TestCase):
    """A live q6 run showed the holistic evaluator (completeness 9) and the
    strict semantic judge (completeness 2) disagree sharply. The design mediates
    this: the semantic judge's findings become P1 issues, and Solved requires
    zero open P0/P1, so a high evaluator score cannot declare a proof solved
    while the strict judge's incomplete_steps are still open."""

    def test_high_completeness_does_not_solve_with_open_strict_issues(self) -> None:
        # evaluator says 9, but the semantic judge left 8 open P1 issues
        self.assertFalse(solved(_m(3, 9.0, 1.0, open_crit=8)))

    def test_solves_only_once_strict_issues_are_closed(self) -> None:
        self.assertTrue(solved(_m(3, 9.0, 1.0, open_crit=0)))


class StopReasonTest(unittest.TestCase):
    def test_solved_takes_priority(self) -> None:
        self.assertEqual(stop_reason(_m(2, 9.0, 1.0, 0), [], 2, 5), "solved")

    def test_budget_exhausted_on_last_round(self) -> None:
        self.assertEqual(stop_reason(_m(4, 3.0, 0.5, 2), [_m(4, 3.0, 0.5, 2)], 4, 5),
                         "budget_exhausted")

    def test_none_when_progressing_mid_run(self) -> None:
        self.assertIsNone(stop_reason(_m(1, 4.0, 0.6, 2), [_m(0, 3.0, 0.5, 2), _m(1, 4.0, 0.6, 2)],
                                      1, 5))


if __name__ == "__main__":
    unittest.main()
