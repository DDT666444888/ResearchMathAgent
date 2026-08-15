"""T4.3 — Solver applies a localized patch, not a full rewrite.

The behaviour under test is the one the B2 run showed was missing: a repair
must change exactly its target region and leave the rest byte-identical. The
old full-rewrite refiner deleted established content between rounds.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rma.config import RunConfig
from rma.ops import get_operation
from rma.ops.base import OpContext
from rma.ops.solver import apply_patch, regions_outside_patch_unchanged
from rma.store import ResearchStore

PROOF = r"""\documentclass{article}\begin{document}
\begin{lemma}\label{a}The first bound holds.\end{lemma}
\begin{proof}It follows immediately from standard arguments.\end{proof}

\begin{lemma}\label{b}The second bound holds.\end{lemma}
\begin{proof}By a routine calculation.\end{proof}

\begin{theorem}\label{main}The result follows.\end{theorem}
\begin{proof}Combine \ref{a} and \ref{b}.\end{proof}
\end{document}"""


class ApplyPatchTest(unittest.TestCase):
    def test_patch_touches_only_target_region(self) -> None:
        patch = {
            "anchor_before": "",
            "replaced_text": "It follows immediately from standard arguments.",
            "replacement": "Expand the discriminant and bound each term by 1/n; "
                           "summing the n terms gives the claim.",
        }
        result = apply_patch(PROOF, patch)
        self.assertTrue(result.applied)
        self.assertIn("Expand the discriminant", result.tex)
        # Everything except the replaced sentence is byte-identical.
        self.assertIn("\\begin{lemma}\\label{b}The second bound holds.\\end{lemma}", result.tex)
        self.assertTrue(regions_outside_patch_unchanged(PROOF, result.tex, patch))

    def test_out_of_region_change_is_impossible_by_construction(self) -> None:
        patch = {"replaced_text": "routine calculation", "replacement": "direct computation"}
        result = apply_patch(PROOF, patch)
        self.assertTrue(result.applied)
        # exactly one substring changed; the two lemmas' statements are intact
        self.assertEqual(PROOF.replace("routine calculation", "direct computation"), result.tex)

    def test_ambiguous_anchor_is_rejected(self) -> None:
        # "holds." appears three times -> not uniquely locatable
        patch = {"replaced_text": "holds.", "replacement": "holds trivially."}
        result = apply_patch(PROOF, patch)
        self.assertFalse(result.applied)
        self.assertEqual(result.mode, "reject")
        self.assertEqual(result.tex, PROOF, "a rejected patch must not change the proof")

    def test_missing_span_is_rejected(self) -> None:
        patch = {"replaced_text": "this text is not in the proof", "replacement": "x"}
        self.assertFalse(apply_patch(PROOF, patch).applied)

    def test_new_lemma_appended_without_disturbing_existing_text(self) -> None:
        patch = {"replaced_text": "", "replacement": "",
                 "new_lemmas": [r"\begin{lemma}\label{aux}An auxiliary bound.\end{lemma}"
                                r"\begin{proof}Direct.\end{proof}"]}
        result = apply_patch(PROOF, patch)
        self.assertTrue(result.applied)
        self.assertIn("auxiliary bound", result.tex)
        # the original body, up to \end{document}, is preserved verbatim
        self.assertIn("Combine \\ref{a} and \\ref{b}.", result.tex)
        self.assertTrue(result.tex.rstrip().endswith("\\end{document}"))

    def test_empty_patch_is_a_noop(self) -> None:
        result = apply_patch(PROOF, {"replaced_text": "", "replacement": ""})
        self.assertFalse(result.applied)
        self.assertEqual(result.mode, "noop")

    def test_whitespace_tolerant_anchoring(self) -> None:
        """A model cannot reproduce a large proof's exact spacing; a patch that
        differs only in whitespace must still apply (else it always falls back
        to a full rewrite on big proofs)."""
        proof = "Lemma 1.\n\n  The  bound   holds\n  by Cauchy--Schwarz.\n\nQED"
        patch = {"replaced_text": "The bound holds by Cauchy--Schwarz.",
                 "replacement": "The bound holds by a three-line estimate."}
        r = apply_patch(proof, patch)
        self.assertTrue(r.applied)
        self.assertIn("three-line estimate", r.tex)
        self.assertTrue(r.tex.startswith("Lemma 1.") and r.tex.endswith("QED"))

    def test_anchor_disambiguates_a_repeated_span(self) -> None:
        proof = "A: holds here.\nB: holds here.\n"
        patch = {"anchor_before": "B: ", "replaced_text": "holds here.",
                 "replacement": "holds trivially."}
        r = apply_patch(proof, patch)
        self.assertTrue(r.applied)
        self.assertEqual(r.tex, "A: holds here.\nB: holds trivially.\n")


class SolverOperationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ResearchStore.open(self.root, "q6", "first_proof_1")
        self.store.add_proof_revision(PROOF, produced_by="proposer")
        self.ctx = OpContext(store=self.store, config=RunConfig(context_budget=60000),
                             problem={"title": "T"}, round=1)
        self.issue = {"id": "q6-I-1", "code": "logical_gap",
                      "message": "Lemma a's proof is hand-waved.", "severity": "P1",
                      "claim_id": "lemma-1"}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_accepted_patch_creates_a_new_revision(self) -> None:
        patch = {"replaced_text": "It follows immediately from standard arguments.",
                 "replacement": "A three-line estimate closes it.", "new_lemmas": []}
        solver = get_operation("solver")
        result = solver.run(self.ctx, issue=self.issue, invoke=lambda u, o: patch)
        self.assertTrue(result.artifact.applied)
        # the store now has 2 proof revisions, the new one links to the issue
        self.assertEqual(len(self.store.proofs), 2)
        latest = self.store.current_proof()
        self.assertIn("three-line estimate", latest.body)
        self.assertEqual(latest.meta["mode"], "patch")
        self.assertEqual(latest.meta["closed_issue"], "q6-I-1")

    def test_patch_revision_diff_is_small(self) -> None:
        patch = {"replaced_text": "By a routine calculation.",
                 "replacement": "By a routine calculation, detailed in three steps below."}
        solver = get_operation("solver")
        solver.run(self.ctx, issue=self.issue, invoke=lambda u, o: patch)
        before, after = self.store.proofs[0].body, self.store.proofs[1].body
        # only the one sentence differs
        self.assertEqual(before.replace("By a routine calculation.",
                                        "By a routine calculation, detailed in three steps below."),
                         after)

    def test_falls_back_after_two_failures_and_increments_counter(self) -> None:
        bad_patch = {"replaced_text": "not present anywhere", "replacement": "x"}
        full = r"\documentclass{article}\begin{document}REWRITTEN\end{document}"

        def _invoke(unit, obs):
            return full if unit == "solver_fallback" else bad_patch

        solver = get_operation("solver")
        before = solver.fallback_count
        result = solver.run(self.ctx, issue=self.issue, invoke=_invoke)
        self.assertEqual(solver.fallback_count, before + 1, "fallback not counted")
        self.assertTrue(self.ctx.extra.get("solver_fallback"))
        self.assertIn("REWRITTEN", self.store.current_proof().body)
        self.assertEqual(self.store.current_proof().meta["mode"], "full_rewrite")

    def test_rejected_patch_with_no_fallback_leaves_proof_unchanged(self) -> None:
        bad_patch = {"replaced_text": "not present", "replacement": "x"}
        solver = get_operation("solver")
        solver.run(self.ctx, issue=self.issue,
                   invoke=lambda u, o: bad_patch if u == "solver" else "")
        # no fallback tex -> proof stays at the single original revision
        self.assertEqual(len(self.store.proofs), 1)
        self.assertEqual(self.store.current_proof().body, PROOF)

    def test_solver_logged(self) -> None:
        from rma.orchestrator import read_telemetry

        patch = {"replaced_text": "By a routine calculation.", "replacement": "By calculation."}
        get_operation("solver").run(self.ctx, issue=self.issue, invoke=lambda u, o: patch)
        self.assertTrue(any(e["unit"] == "solver" for e in read_telemetry(self.store)))


if __name__ == "__main__":
    unittest.main()
