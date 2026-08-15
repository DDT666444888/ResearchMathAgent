"""T6.1 / T6.2 — the ablation matrix: every config runs, and none is a no-op."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rma.ablations import MATRIX, matrix_names, run_matrix
from rma.config import ABLATIONS


class MatrixTest(unittest.TestCase):
    def test_every_ablation_name_is_in_the_matrix(self) -> None:
        names = set(matrix_names())
        self.assertIn("full", names)
        for ab in ABLATIONS:
            self.assertIn(ab, names, f"{ab} missing from the ablation matrix")

    def test_matrix_is_full_plus_one_per_ablation(self) -> None:
        self.assertEqual(len(MATRIX), 1 + len(ABLATIONS))

    def test_every_config_completes_a_run(self) -> None:
        runs = run_matrix(n_rounds=3)
        self.assertEqual(len(runs), len(MATRIX))
        for r in runs:
            # a completed run has at least one round and a stop reason
            self.assertGreaterEqual(r.rounds, 1, f"{r.config} produced no rounds")
            self.assertIsNotNone(r.stop_reason, f"{r.config} did not terminate")

    def test_no_ablation_is_a_noop(self) -> None:
        """A flag that parses but changes nothing is a bug. Every ablation must
        alter the observable signal vector relative to full."""
        runs = {r.config: r for r in run_matrix(n_rounds=3)}
        full = runs["full"]
        offenders = [name for name in ABLATIONS
                     if runs[name].observable == full.observable]
        self.assertEqual(offenders, [], f"these ablations did nothing: {offenders}")

    def test_context_ablations_change_the_applied_mode(self) -> None:
        runs = {r.config: r for r in run_matrix(n_rounds=3)}
        self.assertIn("dump", _flatten(runs["context.dump"].observable["context_modes"]))
        self.assertIn("truncate", _flatten(runs["context.truncate"].observable["context_modes"]))
        self.assertIn("budget", _flatten(runs["full"].observable["context_modes"]))

    def test_order_ablations_change_the_applied_order(self) -> None:
        runs = {r.config: r for r in run_matrix(n_rounds=3)}
        self.assertIn("fifo", _flatten(runs["order.fifo"].observable["order_modes"]))
        self.assertIn("severity", _flatten(runs["order.severity"].observable["order_modes"]))

    def test_meeting_ablation_skips_meeting_and_revise(self) -> None:
        runs = {r.config: r for r in run_matrix(n_rounds=3)}
        units = set(runs["meeting"].observable["units"])
        self.assertNotIn("meeting", units)
        self.assertNotIn("revise", units)


class ReportTest(unittest.TestCase):
    def test_report_emits_figure_ready_metrics(self) -> None:
        from argparse import Namespace

        from rma.ablations import run_report_ablations

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "abl.json"
            rc = run_report_ablations(Namespace(backend="fake", rounds=3, out=str(out)))
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text())
            self.assertEqual(data["n_configs"], len(MATRIX))
            keys = set(data["configs"][0])
            for k in ("config", "completeness", "proved_terminal_fraction",
                      "issues_opened", "context_tokens_mean", "records", "differs_from_full"):
                self.assertIn(k, keys)
            # every non-full config is marked as differing
            noop = [c["config"] for c in data["configs"]
                    if c["config"] != "full" and not c["differs_from_full"]]
            self.assertEqual(noop, [])


def _flatten(modes_tuple) -> set:
    out = set()
    for m in modes_tuple:
        if isinstance(m, (tuple, list)):
            out.update(m)
        else:
            out.add(m)
    return out


if __name__ == "__main__":
    unittest.main()
