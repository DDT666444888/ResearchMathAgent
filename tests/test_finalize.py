"""T5.2 — FinalizeRound must deliver the best round, not the last one.

Refinement is not monotone. On a full First Proof B2 run the delivered round
was not the best round on 9 of 10 problems; selecting the peak instead cut
total open issues from 123 to 90.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rma.finalize import RoundRecord, collect_rounds, finalize_problem, select_best_round


def _rec(index, issues, errors=0, proved=0, claims=0, completeness=None, passed=False):
    return RoundRecord(index=index, source=Path(f"r{index}.tex"), issues=issues,
                       errors=errors, proved=proved, claims=claims,
                       completeness=completeness, passed=passed)


class SelectionTest(unittest.TestCase):
    def test_fewest_errors_wins(self) -> None:
        best = select_best_round([_rec(1, 10, errors=8), _rec(2, 12, errors=2)])
        self.assertEqual(best.index, 2)

    def test_a_passing_round_always_wins(self) -> None:
        best = select_best_round([_rec(1, 0, errors=0, passed=True), _rec(2, 0, errors=0)])
        self.assertEqual(best.index, 1)

    def test_proved_claims_break_an_issue_tie(self) -> None:
        """Raw issue count alone rewards a proof for being too short to
        criticise; proved-claim count is the tiebreak."""
        best = select_best_round([_rec(1, 12, errors=12, proved=8, claims=8),
                                  _rec(2, 12, errors=12, proved=12, claims=14)])
        self.assertEqual(best.index, 2)

    def test_completeness_breaks_remaining_ties(self) -> None:
        best = select_best_round([_rec(1, 5, errors=5, proved=3, completeness=4.0),
                                  _rec(2, 5, errors=5, proved=3, completeness=8.0)])
        self.assertEqual(best.index, 2)

    def test_empty_input(self) -> None:
        self.assertIsNone(select_best_round([]))

    def test_last_round_wins_when_it_is_genuinely_best(self) -> None:
        best = select_best_round([_rec(1, 20, errors=20), _rec(2, 3, errors=3)])
        self.assertEqual(best.index, 2)


class FinalizeOnDiskTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.paths = {
            "artifacts": root / "artifacts",
            "proposals": root / "artifacts" / "proposals",
            "verifications": root / "artifacts" / "verifications",
            "refinements": root / "artifacts" / "refinements",
            "solution": root / "prob-01_solution.tex",
        }
        for key in ("artifacts", "proposals", "verifications", "refinements"):
            self.paths[key].mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _round(self, i: int, issues: int, proved: int, claims: int, text: str) -> None:
        if i == 1:
            (self.paths["proposals"] / "proposal_001.tex").write_text(text, encoding="utf-8")
        else:
            (self.paths["refinements"] / f"refined_solution_{i-1:03d}.tex").write_text(
                text, encoding="utf-8")
        (self.paths["verifications"] / f"verification_{i:03d}.json").write_text(json.dumps({
            "passed": False,
            "issues": [{"code": "logical_gap", "severity": "error"}] * issues,
            "lemma_dag": {"nodes": claims, "proved": proved},
            "completeness_score": 2.0,
        }), encoding="utf-8")
        self.paths["solution"].write_text(text, encoding="utf-8")  # as the loop would

    def test_pairs_each_verification_with_the_proof_it_judged(self) -> None:
        self._round(1, 16, 9, 13, "ROUND ONE")
        self._round(2, 14, 8, 10, "ROUND TWO")
        rounds = collect_rounds(self.paths)
        self.assertEqual([r.index for r in rounds], [1, 2])
        self.assertEqual(rounds[0].source.read_text(), "ROUND ONE")
        self.assertEqual(rounds[1].issues, 14)

    def test_best_round_replaces_the_delivered_solution(self) -> None:
        """The prob-04 shape: round 3 is far better than round 5."""
        self._round(1, 13, 7, 9, "ROUND ONE")
        self._round(2, 10, 8, 10, "ROUND TWO")
        self._round(3, 3, 10, 10, "ROUND THREE - THE GOOD ONE")
        self._round(4, 11, 8, 10, "ROUND FOUR")
        self._round(5, 13, 7, 9, "ROUND FIVE")

        summary = finalize_problem(self.paths)
        self.assertEqual(summary["delivered_round"], 3)
        self.assertEqual(summary["last_round"], 5)
        self.assertTrue(summary["replaced_last_round"])
        self.assertEqual(self.paths["solution"].read_text(), "ROUND THREE - THE GOOD ONE")

    def test_superseded_round_is_preserved(self) -> None:
        self._round(1, 3, 10, 10, "GOOD")
        self._round(2, 13, 7, 9, "WORSE")
        finalize_problem(self.paths)
        superseded = self.paths["artifacts"] / "superseded_last_round.tex"
        self.assertTrue(superseded.is_file(), "the swap must be reversible")
        self.assertEqual(superseded.read_text(), "WORSE")

    def test_no_swap_when_last_round_is_best(self) -> None:
        self._round(1, 13, 7, 9, "ROUND ONE")
        self._round(2, 3, 10, 10, "ROUND TWO")
        summary = finalize_problem(self.paths)
        self.assertFalse(summary["replaced_last_round"])
        self.assertEqual(self.paths["solution"].read_text(), "ROUND TWO")

    def test_selection_report_written(self) -> None:
        self._round(1, 3, 10, 10, "A")
        self._round(2, 9, 7, 9, "B")
        finalize_problem(self.paths)
        report = json.loads((self.paths["artifacts"] / "round_selection.json").read_text())
        self.assertEqual(len(report["rounds"]), 2)
        self.assertEqual(report["delivered_round"], 1)

    def test_apply_false_reports_without_changing_anything(self) -> None:
        self._round(1, 3, 10, 10, "GOOD")
        self._round(2, 13, 7, 9, "WORSE")
        summary = finalize_problem(self.paths, apply=False)
        self.assertEqual(summary["delivered_round"], 1)
        self.assertFalse(summary["replaced_last_round"])
        self.assertEqual(self.paths["solution"].read_text(), "WORSE")

    def test_single_round_is_a_noop(self) -> None:
        self._round(1, 13, 6, 9, "ONLY")
        summary = finalize_problem(self.paths)
        self.assertEqual(summary["delivered_round"], 1)
        self.assertFalse(summary["replaced_last_round"])

    def test_no_rounds_returns_none(self) -> None:
        self.assertIsNone(finalize_problem(self.paths))


if __name__ == "__main__":
    unittest.main()
